"""
Phase 3 单组运行脚本 - 支持并行
用法：
  CUDA_VISIBLE_DEVICES=0 python run_phase3_single.py --group A1 --eraser_init 0.0
  CUDA_VISIBLE_DEVICES=1 python run_phase3_single.py --group A2 --eraser_init -4.6
  CUDA_VISIBLE_DEVICES=0 python run_phase3_single.py --group B
"""

import torch
import torch.nn as nn
import json
import os
import sys
import random
import argparse
import numpy as np
from datetime import datetime
import time

sys.path.insert(0, '/data/biic')
sys.path.insert(0, '/data/biic/phase2')

from clifford_cl41 import GRADE_SLICES, N_BLADES
from rotor_utils import sandwich_product, exp_bivector, normalize_rotor
from eraser_ops import GradeAwareEraser
from token_to_ic import TokenToImmutableCore
from all_grade_decoder import AllGradeDecoder
from mutable_state import BIICLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEEDS = [42, 123, 456]
N_STEPS = 2000
N_CHANNELS = 64
N_LAYERS = 12
BATCH_SIZE = 16
SEQ_LEN = 64
LR = 1e-3
VOCAB_SIZE = 50257
D_HIDDEN = 256
LOG_EVERY = 10


class FixedBIICLoss(nn.Module):
    def __init__(self, vocab_size, n_channels):
        super().__init__()
        self.grade0_classifier = nn.Linear(n_channels, vocab_size)
        self.n_channels = n_channels

    def get_alpha(self, step, total_steps):
        frac = step / max(total_steps, 1)
        if frac < 0.1:
            return 1.0
        elif frac < 0.5:
            return 1.0 - (frac - 0.1) / 0.4 * 0.99
        else:
            return 0.01

    def forward(self, mv, main_logits, targets, step, total_steps):
        B, L, C, _ = mv.shape
        main_loss = nn.CrossEntropyLoss()(
            main_logits.reshape(-1, main_logits.size(-1)), targets.reshape(-1))
        alpha = self.get_alpha(step, total_steps)
        if alpha <= 0.01:
            return main_loss, {'main': main_loss.item(), 'alpha': alpha}
        g0_start, g0_end = GRADE_SLICES[0]
        grade0 = mv[..., g0_start:g0_end].squeeze(-1)
        aux1_logits = self.grade0_classifier(grade0)
        aux1_loss = nn.CrossEntropyLoss()(
            aux1_logits.reshape(-1, aux1_logits.size(-1)), targets.reshape(-1))
        variance_loss = torch.tensor(0.0, device=mv.device)
        for g in [0, 1, 2]:
            start, end = GRADE_SLICES[g]
            g_data = mv[..., start:end]
            channel_var = g_data.var(dim=-2).mean()
            variance_loss = variance_loss - channel_var.clamp(max=100.0)
        total_loss = main_loss + alpha * (0.6 * aux1_loss + 0.4 * variance_loss.clamp(-50, 50))
        return total_loss, {'main': main_loss.item(), 'aux1': aux1_loss.item(), 'var': variance_loss.item(), 'alpha': alpha}


class OrthogonalTokenBaseline(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.skew_params = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_model) * 0.01) for _ in range(n_layers)])
        self.n_layers = n_layers
        self.d_model = d_model
        self.output = nn.Linear(d_model, vocab_size)

    def get_orthogonal_matrix(self, A):
        A_skew = A - A.T
        I = torch.eye(self.d_model, device=A.device)
        return torch.linalg.solve(I + A_skew, I - A_skew)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        for i in range(self.n_layers):
            Q = self.get_orthogonal_matrix(self.skew_params[i])
            x = x @ Q.T
            x = torch.tanh(x)
            x = x @ Q
        return self.output(x)

    def get_initial_embed(self, token_ids):
        return self.embed(token_ids)


class LinearBaseline(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.output = nn.Linear(d_model, vocab_size)
        self.n_layers = n_layers
        self.d_model = d_model

    def forward(self, token_ids):
        x = self.embed(token_ids)
        for layer, norm in zip(self.layers, self.norms):
            x = torch.relu(norm(layer(x)))
        return self.output(x)

    def get_initial_embed(self, token_ids):
        return self.embed(token_ids)


class DimMatchedBaseline(nn.Module):
    def __init__(self, vocab_size, d_model=2048, n_layers=6):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.skew_params = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_model) * 0.001) for _ in range(n_layers)])
        self.n_layers = n_layers
        self.d_model = d_model
        self.output = nn.Linear(d_model, vocab_size)

    def get_orthogonal_matrix(self, A):
        A_skew = A - A.T
        I = torch.eye(self.d_model, device=A.device)
        return torch.linalg.solve(I + A_skew, I - A_skew)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        for i in range(self.n_layers):
            Q = self.get_orthogonal_matrix(self.skew_params[i])
            x = x @ Q.T
            x = torch.tanh(x)
            x = x @ Q
        return self.output(x)

    def get_initial_embed(self, token_ids):
        return self.embed(token_ids)


def train_biic(seed, eraser_init=0.0, group_name=''):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    encoder = TokenToImmutableCore(VOCAB_SIZE, N_CHANNELS).to(device)
    layers = nn.ModuleList([
        BIICLayer(N_CHANNELS, use_residual=True, use_eraser=True) for _ in range(N_LAYERS)
    ]).to(device)
    for layer in layers:
        if hasattr(layer.mutable_layer, 'eraser'):
            with torch.no_grad():
                layer.mutable_layer.eraser.decay_logits.fill_(eraser_init)
    decoder = AllGradeDecoder(VOCAB_SIZE, N_CHANNELS, D_HIDDEN).to(device)
    loss_fn = FixedBIICLoss(VOCAB_SIZE, N_CHANNELS).to(device)

    all_params = list(encoder.parameters()) + list(layers.parameters()) + list(decoder.parameters()) + list(loss_fn.parameters())
    optimizer = torch.optim.Adam(all_params, lr=LR)

    loss_curve = []
    g0_start, g0_end = GRADE_SLICES[0]

    for step in range(N_STEPS):
        token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
        inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
        optimizer.zero_grad()
        mv = encoder(inputs)
        mv_prior = mv.clone().detach()
        for layer in layers:
            mv = layer(mv, mv_prior)
        logits = decoder(mv)
        loss, ld = loss_fn(mv, logits, targets, step, N_STEPS)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        if step % LOG_EVERY == 0:
            loss_curve.append({"step": step, "loss": loss.item(), "main": ld["main"], "alpha": ld["alpha"]})
        if step % 200 == 0:
            print("  %s seed=%d Step %d: loss=%.4f main=%.4f alpha=%.3f" % (
                group_name, seed, step, loss.item(), ld["main"], ld["alpha"]), flush=True)

    # Eval: grade-0 retention
    retention = {}
    test_ids = torch.randint(0, VOCAB_SIZE, (8, SEQ_LEN)).to(device)
    with torch.no_grad():
        mv_init = encoder(test_ids)
        grade0_init = mv_init[..., g0_start:g0_end].clone()
        mv_eval = mv_init.clone()
        mv_prior_eval = mv_init.clone()
        for n in [5, 10, 20, 50, 100]:
            for layer in layers:
                mv_eval = layer(mv_eval, mv_prior_eval)
            g0_cur = mv_eval[..., g0_start:g0_end]
            sim = nn.CosineSimilarity(dim=-1)(
                grade0_init.reshape(-1, grade0_init.size(-1)),
                g0_cur.reshape(-1, g0_cur.size(-1))).mean().item()
            retention[n] = sim

    # Eval: entropy
    entropy_history = []
    with torch.no_grad():
        mv_ent = encoder(test_ids)
        mv_prior_ent = mv_ent.clone()
        g1_start = GRADE_SLICES[1][0]
        g4_end = GRADE_SLICES[4][1]
        for i in range(20):
            for layer in layers:
                mv_ent = layer(mv_ent, mv_prior_ent)
            ent = mv_ent[..., g1_start:g4_end].norm(dim=-1).var().item()
            entropy_history.append(ent)

    return {"loss_curve": loss_curve, "final_loss": loss_curve[-1]["loss"],
            "final_main_loss": loss_curve[-1]["main"], "retention": retention,
            "entropy_history": entropy_history}


def train_simple_orthogonal(seed, group_name='B'):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = OrthogonalTokenBaseline(VOCAB_SIZE, d_model=64, n_layers=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_curve = []
    for step in range(N_STEPS):
        token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
        inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
        optimizer.zero_grad()
        logits = model(inputs)
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % LOG_EVERY == 0:
            loss_curve.append({"step": step, "loss": loss.item()})
        if step % 200 == 0:
            print("  %s seed=%d Step %d: loss=%.4f" % (group_name, seed, step, loss.item()), flush=True)

    retention = {}
    test_ids = torch.randint(0, VOCAB_SIZE, (8, SEQ_LEN)).to(device)
    with torch.no_grad():
        init_embed = model.get_initial_embed(test_ids)
        for n in [5, 10, 20, 50, 100]:
            x = init_embed.clone()
            for i in range(min(n, model.n_layers)):
                Q = model.get_orthogonal_matrix(model.skew_params[i])
                x = x @ Q.T
                x = torch.tanh(x)
                x = x @ Q
            sim = nn.CosineSimilarity(dim=-1)(
                init_embed.reshape(-1, init_embed.size(-1)),
                x.reshape(-1, x.size(-1))).mean().item()
            retention[n] = sim
    return {"loss_curve": loss_curve, "final_loss": loss_curve[-1]["loss"], "retention": retention}


def train_linear(seed, group_name='C'):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = LinearBaseline(VOCAB_SIZE, d_model=64, n_layers=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_curve = []
    for step in range(N_STEPS):
        token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
        inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
        optimizer.zero_grad()
        logits = model(inputs)
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % LOG_EVERY == 0:
            loss_curve.append({"step": step, "loss": loss.item()})
        if step % 200 == 0:
            print("  %s seed=%d Step %d: loss=%.4f" % (group_name, seed, step, loss.item()), flush=True)
    return {"loss_curve": loss_curve, "final_loss": loss_curve[-1]["loss"]}


def train_biic_grade0_only(seed, group_name='D'):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    encoder = TokenToImmutableCore(VOCAB_SIZE, N_CHANNELS).to(device)
    layers = nn.ModuleList([
        BIICLayer(N_CHANNELS, use_residual=True, use_eraser=True) for _ in range(N_LAYERS)
    ]).to(device)
    for layer in layers:
        if hasattr(layer.mutable_layer, 'eraser'):
            with torch.no_grad():
                layer.mutable_layer.eraser.decay_logits.fill_(0.0)
    decoder = AllGradeDecoder(VOCAB_SIZE, N_CHANNELS, D_HIDDEN).to(device)
    all_params = list(encoder.parameters()) + list(layers.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.Adam(all_params, lr=LR)
    loss_curve = []
    for step in range(N_STEPS):
        token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
        inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
        optimizer.zero_grad()
        mv = encoder(inputs)
        mv_prior = mv.clone().detach()
        for layer in layers:
            mv = layer(mv, mv_prior)
        logits = decoder(mv, active_grades=[0])
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()
        if step % LOG_EVERY == 0:
            loss_curve.append({"step": step, "loss": loss.item()})
        if step % 200 == 0:
            print("  %s seed=%d Step %d: loss=%.4f" % (group_name, seed, step, loss.item()), flush=True)
    return {"loss_curve": loss_curve, "final_loss": loss_curve[-1]["loss"]}


def train_dim_matched(seed, group_name='E'):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = DimMatchedBaseline(VOCAB_SIZE, d_model=2048, n_layers=6).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_curve = []
    for step in range(N_STEPS):
        token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
        inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
        optimizer.zero_grad()
        logits = model(inputs)
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % LOG_EVERY == 0:
            loss_curve.append({"step": step, "loss": loss.item()})
        if step % 200 == 0:
            print("  %s seed=%d Step %d: loss=%.4f" % (group_name, seed, step, loss.item()), flush=True)
    retention = {}
    test_ids = torch.randint(0, VOCAB_SIZE, (8, SEQ_LEN)).to(device)
    with torch.no_grad():
        init_embed = model.get_initial_embed(test_ids)
        for n in [5, 10, 20, 50, 100]:
            x = init_embed.clone()
            for i in range(min(n, model.n_layers)):
                Q = model.get_orthogonal_matrix(model.skew_params[i])
                x = x @ Q.T
                x = torch.tanh(x)
                x = x @ Q
            sim = nn.CosineSimilarity(dim=-1)(
                init_embed.reshape(-1, init_embed.size(-1)),
                x.reshape(-1, x.size(-1))).mean().item()
            retention[n] = sim
    return {"loss_curve": loss_curve, "final_loss": loss_curve[-1]["loss"], "retention": retention}


def aggregate_seeds(all_runs):
    final_losses = [r["final_loss"] for r in all_runs]
    agg = {"final_loss_mean": float(np.mean(final_losses)), "final_loss_std": float(np.std(final_losses)),
           "final_losses": final_losses, "n_seeds": len(all_runs)}
    if "retention" in all_runs[0] and all_runs[0]["retention"]:
        ret_agg = {}
        for n in all_runs[0]["retention"]:
            vals = [r["retention"][n] for r in all_runs if n in r.get("retention", {})]
            ret_agg[str(n)] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
        agg["retention"] = ret_agg
    if "entropy_history" in all_runs[0] and all_runs[0]["entropy_history"]:
        ent_arrays = [r["entropy_history"] for r in all_runs]
        min_len = min(len(e) for e in ent_arrays)
        ent_mean = [float(np.mean([e[i] for e in ent_arrays])) for i in range(min_len)]
        agg["entropy_mean"] = ent_mean
    return agg


def run_group(group, eraser_init=0.0, seeds=SEEDS):
    print("\n" + "=" * 60)
    print("Group %s (seeds=%s)" % (group, seeds))
    print("=" * 60)
    t0 = time.time()

    runs = []
    for seed in seeds:
        if group == 'A1':
            r = train_biic(seed, eraser_init=0.0, group_name='A1')
        elif group == 'A2':
            r = train_biic(seed, eraser_init=-4.6, group_name='A2')
        elif group == 'B':
            r = train_simple_orthogonal(seed, group_name='B')
        elif group == 'C':
            r = train_linear(seed, group_name='C')
        elif group == 'D':
            r = train_biic_grade0_only(seed, group_name='D')
        elif group == 'E':
            r = train_dim_matched(seed, group_name='E')
        else:
            raise ValueError("Unknown group: %s" % group)
        runs.append(r)

    elapsed = time.time() - t0
    agg = aggregate_seeds(runs)

    result = {
        "group": group,
        "desc": {"A1": "BIIC Full, Eraser=0.5", "A2": "BIIC Weak Eraser=0.01",
                 "B": "Orthogonal Token + tanh (H1)", "C": "Linear + LayerNorm (lower bound)",
                 "D": "BIIC grade-0 only (H2 ablation)", "E": "2048-dim Embedding + Orthogonal (H2)"}[group],
        "seeds": seeds,
        "n_steps": N_STEPS,
        "elapsed_minutes": elapsed / 60,
        "aggregate": agg,
        "runs": runs,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "hardware": "RTX 4090",
        "config": {"n_channels": N_CHANNELS, "n_layers": N_LAYERS, "batch_size": BATCH_SIZE,
                   "seq_len": SEQ_LEN, "lr": LR, "vocab_size": VOCAB_SIZE},
    }

    print("\nGroup %s done in %.1f min" % (group, elapsed / 60))
    print("  Final loss: %.4f +/- %.4f" % (agg["final_loss_mean"], agg["final_loss_std"]))

    # Save
    os.makedirs('/data/biic/phase3/logs', exist_ok=True)
    os.makedirs('/data/biic/results', exist_ok=True)
    save_path = "/data/biic/phase3/logs/phase3_%s_%s.json" % (group, datetime.now().strftime("%Y%m%d_%H%M%S"))
    with open(save_path, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print("  Saved:", save_path)

    # Also save to results dir
    with open("/data/biic/results/phase3_%s.json" % group, 'w') as f:
        json.dump(result, f, indent=2, default=str)

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--group', type=str, required=True, choices=['A1', 'A2', 'B', 'C', 'D', 'E'])
    parser.add_argument('--eraser_init', type=float, default=0.0)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--n_steps', type=int, default=2000)
    args = parser.parse_args()

    N_STEPS = args.n_steps
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    run_group(args.group, eraser_init=args.eraser_init, seeds=args.seeds)
