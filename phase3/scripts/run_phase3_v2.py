"""
Phase 3 主实验脚本 v2 - 论文级数据输出
修复：
1. variance loss加clamp防爆炸
2. 每10步记录loss_curve
3. 3个种子，计算均值±std
4. 完整结构化JSON输出
"""

import torch
import torch.nn as nn
import json
import os
import sys
import random
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
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

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
    """修复版BIICLoss - variance项加clamp防爆炸"""
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

        # 修复：variance loss加clamp防爆炸
        variance_loss = torch.tensor(0.0, device=mv.device)
        for g in [0, 1, 2]:
            start, end = GRADE_SLICES[g]
            g_data = mv[..., start:end]
            channel_var = g_data.var(dim=-2).mean()
            variance_loss = variance_loss - channel_var.clamp(max=100.0)

        total_loss = main_loss + alpha * (0.6 * aux1_loss + 0.4 * variance_loss.clamp(-50, 50))
        return total_loss, {
            'main': main_loss.item(),
            'aux1': aux1_loss.item(),
            'var': variance_loss.item(),
            'alpha': alpha,
        }


class OrthogonalTokenBaseline(nn.Module):
    """组B: Token + 正交变换 + tanh"""
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
    """组C: Token + 线性变换 + LayerNorm"""
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
    """组E: 2048维embedding + 正交变换"""
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
    """训练BIIC模型，返回完整指标"""
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

    # 评测：grade-0保持率
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

    # 评测：信息熵变化
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

    return {
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1]["loss"],
        "final_main_loss": loss_curve[-1]["main"],
        "retention": retention,
        "entropy_history": entropy_history,
    }


def train_simple(model_class, seed, group_name='', **kwargs):
    """训练简单模型，返回完整指标"""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    model = model_class(VOCAB_SIZE, **kwargs).to(device)
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

    # 评测：embedding保持率
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

    return {
        "loss_curve": loss_curve,
        "final_loss": loss_curve[-1]["loss"],
        "retention": retention,
    }


def train_biic_grade0_only(seed):
    """组D: BIIC只用grade-0解码"""
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
                print("  D seed=%d Step %d: loss=%.4f" % (seed, step, loss.item()), flush=True)

    return {"loss_curve": loss_curve, "final_loss": loss_curve[-1]["loss"]}


def aggregate_seeds(all_runs):
    """聚合多种子结果"""
    final_losses = [r["final_loss"] for r in all_runs]
    agg = {
        "final_loss_mean": float(np.mean(final_losses)),
        "final_loss_std": float(np.std(final_losses)),
        "final_losses": final_losses,
        "n_seeds": len(all_runs),
    }
    # 聚合retention
    if "retention" in all_runs[0] and all_runs[0]["retention"]:
        ret_agg = {}
        for n in all_runs[0]["retention"]:
            vals = [r["retention"][n] for r in all_runs if n in r.get("retention", {})]
            ret_agg[str(n)] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
        agg["retention"] = ret_agg
    # 聚合entropy
    if "entropy_history" in all_runs[0] and all_runs[0]["entropy_history"]:
        ent_arrays = [r["entropy_history"] for r in all_runs]
        min_len = min(len(e) for e in ent_arrays)
        ent_mean = [float(np.mean([e[i] for e in ent_arrays])) for i in range(min_len)]
        agg["entropy_mean"] = ent_mean
    return agg


def run_all():
    os.makedirs('/data/biic/phase3/logs', exist_ok=True)
    all_results = {}
    t_start = time.time()

    # === Group A1: BIIC + Strong Eraser ===
    print("\n" + "=" * 60)
    print("Group A1: BIIC Full (Eraser=0.5)")
    print("=" * 60)
    a1_runs = []
    for seed in SEEDS:
        r = train_biic(seed, eraser_init=0.0, group_name='A1')
        a1_runs.append(r)
    all_results['A1'] = {"desc": "BIIC Full, Eraser=0.5", "runs": a1_runs, "aggregate": aggregate_seeds(a1_runs)}

    # === Group A2: BIIC + Weak Eraser ===
    print("\n" + "=" * 60)
    print("Group A2: BIIC Weak Eraser (0.01)")
    print("=" * 60)
    a2_runs = []
    for seed in SEEDS:
        r = train_biic(seed, eraser_init=-4.6, group_name='A2')
        a2_runs.append(r)
    all_results['A2'] = {"desc": "BIIC Weak Eraser=0.01", "runs": a2_runs, "aggregate": aggregate_seeds(a2_runs)}

    # === Group B: Orthogonal Token ===
    print("\n" + "=" * 60)
    print("Group B: Orthogonal Token (H1)")
    print("=" * 60)
    b_runs = []
    for seed in SEEDS:
        r = train_simple(OrthogonalTokenBaseline, seed, group_name='B', d_model=64, n_layers=12)
        b_runs.append(r)
    all_results['B'] = {"desc": "Orthogonal Token + tanh (H1)", "runs": b_runs, "aggregate": aggregate_seeds(b_runs)}

    # === Group C: Linear Baseline ===
    print("\n" + "=" * 60)
    print("Group C: Linear Baseline (lower bound)")
    print("=" * 60)
    c_runs = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model_c = LinearBaseline(VOCAB_SIZE, d_model=64, n_layers=12).to(device)
        optimizer_c = torch.optim.Adam(model_c.parameters(), lr=LR)
        loss_curve_c = []
        for step in range(N_STEPS):
            token_ids = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(device)
            inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
            optimizer_c.zero_grad()
            logits = model_c(inputs)
            loss = nn.CrossEntropyLoss()(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_c.parameters(), 1.0)
            optimizer_c.step()
            if step % LOG_EVERY == 0:
                loss_curve_c.append({"step": step, "loss": loss.item()})
            if step % 200 == 0:
                print("  C seed=%d Step %d: loss=%.4f" % (seed, step, loss.item()), flush=True)
        c_runs.append({"loss_curve": loss_curve_c, "final_loss": loss_curve_c[-1]["loss"]})
    all_results['C'] = {"desc": "Linear + LayerNorm (lower bound)", "runs": c_runs, "aggregate": aggregate_seeds(c_runs)}

    # === Group D: BIIC grade-0 only ===
    print("\n" + "=" * 60)
    print("Group D: BIIC grade-0 only (H2)")
    print("=" * 60)
    d_runs = []
    for seed in SEEDS:
        r = train_biic_grade0_only(seed)
        d_runs.append(r)
    all_results['D'] = {"desc": "BIIC grade-0 only (H2 ablation)", "runs": d_runs, "aggregate": aggregate_seeds(d_runs)}

    # === Group E: 2048-dim Embedding ===
    print("\n" + "=" * 60)
    print("Group E: 2048-dim Embedding (H2 dim-matched)")
    print("=" * 60)
    e_runs = []
    for seed in SEEDS:
        r = train_simple(DimMatchedBaseline, seed, group_name='E', d_model=2048, n_layers=6)
        e_runs.append(r)
    all_results['E'] = {"desc": "2048-dim Embedding + Orthogonal (H2)", "runs": e_runs, "aggregate": aggregate_seeds(e_runs)}

    # === Summary ===
    t_total = time.time() - t_start
    print("\n" + "=" * 60)
    print("PHASE 3 RESULTS SUMMARY")
    print("=" * 60)
    print("Total time: %.1f minutes" % (t_total / 60))

    print("\n--- Final Loss (mean +/- std across %d seeds) ---" % len(SEEDS))
    for group, data in all_results.items():
        agg = data["aggregate"]
        print("  %s: %.4f +/- %.4f | %s" % (group, agg["final_loss_mean"], agg["final_loss_std"], data["desc"]))

    print("\n--- Hypothesis Tests ---")
    a1_mean = all_results['A1']['aggregate']['final_loss_mean']
    b_mean = all_results['B']['aggregate']['final_loss_mean']
    d_mean = all_results['D']['aggregate']['final_loss_mean']
    e_mean = all_results['E']['aggregate']['final_loss_mean']

    print("  H1 (Geometry): A1=%.4f vs B=%.4f" % (a1_mean, b_mean))
    if a1_mean < b_mean * 0.95:
        print("    -> BIIC wins: geometric structure has intrinsic value")
    else:
        print("    -> Comparable or orthogonal wins")

    print("  H2 (Equivariance): A1=%.4f vs D=%.4f vs E=%.4f" % (a1_mean, d_mean, e_mean))
    if a1_mean < d_mean and a1_mean < e_mean:
        print("    -> Equivariant structure has independent value")
    else:
        print("    -> No clear equivariance advantage")

    # H3: Eraser
    a1_ent = all_results['A1']['aggregate'].get('entropy_mean', [])
    a2_ent = all_results['A2']['aggregate'].get('entropy_mean', [])
    if a1_ent and a2_ent:
        print("  H3 (Eraser): A1 entropy[0]=%.4f->[end]=%.4f, A2 entropy[0]=%.4f->[end]=%.4f" % (
            a1_ent[0], a1_ent[-1], a2_ent[0], a2_ent[-1]))

    # Save
    meta = {
        "phase": 3,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_time_minutes": t_total / 60,
        "hardware": "RTX 4090",
        "seeds": SEEDS,
        "config": {
            "n_channels": N_CHANNELS, "n_layers": N_LAYERS, "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN, "n_steps": N_STEPS, "lr": LR, "vocab_size": VOCAB_SIZE,
        },
    }

    save_data = {"meta": meta}
    for group, data in all_results.items():
        save_data[group] = {
            "desc": data["desc"],
            "aggregate": data["aggregate"],
            "runs": [{k: v for k, v in r.items() if k != "loss_curve"} for r in data["runs"]],
            "loss_curves": [r["loss_curve"] for r in data["runs"]],
        }

    save_path = "/data/biic/phase3/logs/phase3_results_%s.json" % datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(save_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print("\nResults saved:", save_path)

    # Also save a compact summary
    summary_path = "/data/biic/results/phase3_results.json"
    with open(summary_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print("Also saved:", summary_path)

    return all_results


if __name__ == "__main__":
    results = run_all()
