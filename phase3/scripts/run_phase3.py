"""
Phase 3 主实验脚本 - 信息穿透性对比实验
7组实验：A1/A2/B/C/D/E + 汇总分析
"""

import torch
import torch.nn as nn
import json
import os
import sys
import random
import numpy as np
from datetime import datetime

sys.path.insert(0, '/data/biic')
sys.path.insert(0, '/data/biic/phase2')

from clifford_cl41 import GRADE_SLICES, N_BLADES
from rotor_utils import sandwich_product, exp_bivector, normalize_rotor
from eraser_ops import GradeAwareEraser
from token_to_ic import TokenToImmutableCore
from all_grade_decoder import AllGradeDecoder
from mutable_state import BIICLayer
from biic_loss import BIICLoss

sys.path.insert(0, '/data/biic/phase3')
from config import CONFIG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


# ═══════════════════════════════════════════════════════════
# 模型构建
# ═══════════════════════════════════════════════════════════

def build_biic_model(eraser_init=0.0, n_layers=12, n_channels=64):
    encoder = TokenToImmutableCore(CONFIG['vocab_size'], n_channels).to(device)
    layers = nn.ModuleList([
        BIICLayer(n_channels, use_residual=True, use_eraser=True)
        for _ in range(n_layers)
    ]).to(device)
    # 修复Eraser衰减率初始化
    for layer in layers:
        if hasattr(layer.mutable_layer, 'eraser'):
            with torch.no_grad():
                layer.mutable_layer.eraser.decay_logits.fill_(eraser_init)
    decoder = AllGradeDecoder(CONFIG['vocab_size'], n_channels, CONFIG['d_hidden']).to(device)
    return encoder, layers, decoder


class OrthogonalTokenBaseline(nn.Module):
    """组B: Token + 正交变换 + tanh (H1检验)"""
    def __init__(self, vocab_size, d_model=64, n_layers=12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.skew_params = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_model) * 0.01)
            for _ in range(n_layers)
        ])
        self.n_layers = n_layers
        self.d_model = d_model
        self.output = nn.Linear(d_model, vocab_size)

    def get_orthogonal_matrix(self, A):
        A_skew = A - A.T
        I = torch.eye(self.d_model, device=A.device)
        Q = torch.linalg.solve(I + A_skew, I - A_skew)
        return Q

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


class DimMatchedBaseline(nn.Module):
    """组E: 2048维普通embedding + 正交变换 (H2维度匹配)"""
    def __init__(self, vocab_size, d_model=2048, n_layers=12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.skew_params = nn.ParameterList([
            nn.Parameter(torch.randn(d_model, d_model) * 0.001)
            for _ in range(n_layers)
        ])
        self.n_layers = n_layers
        self.d_model = d_model
        self.output = nn.Linear(d_model, vocab_size)

    def get_orthogonal_matrix(self, A):
        A_skew = A - A.T
        I = torch.eye(self.d_model, device=A.device)
        Q = torch.linalg.solve(I + A_skew, I - A_skew)
        return Q

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
    """组C: Token + 随机线性变换 (下界)"""
    def __init__(self, vocab_size, d_model=64, n_layers=12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(n_layers)])
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        for layer, norm in zip(self.layers, self.norms):
            x = torch.relu(norm(layer(x)))
        return self.output(x)

    def get_initial_embed(self, token_ids):
        return self.embed(token_ids)


# ═══════════════════════════════════════════════════════════
# 训练函数
# ═══════════════════════════════════════════════════════════

def train_biic(encoder, layers, decoder, n_steps, group_name):
    print(f"\n{'='*50}")
    print(f"Training Group {group_name} ({n_steps} steps)")
    print(f"{'='*50}")

    loss_fn = BIICLoss(CONFIG['vocab_size'], CONFIG['n_channels']).to(device)
    all_params = (list(encoder.parameters()) + list(layers.parameters()) +
                  list(decoder.parameters()) + list(loss_fn.parameters()))
    optimizer = torch.optim.Adam(all_params, lr=CONFIG['lr'])

    losses = []
    for step in range(n_steps):
        token_ids = torch.randint(0, CONFIG['vocab_size'], (CONFIG['batch_size'], 64)).to(device)
        inputs = token_ids[:, :-1]
        targets = token_ids[:, 1:]

        optimizer.zero_grad()
        mv = encoder(inputs)
        mv_prior = mv.clone().detach()
        for layer in layers:
            mv = layer(mv, mv_prior)
        logits = decoder(mv)
        loss, loss_dict = loss_fn(mv, logits, targets, step, n_steps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, CONFIG['grad_clip'])
        optimizer.step()
        losses.append(loss.item())

        if step % 200 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}, alpha={loss_dict['alpha']:.3f}", flush=True)

    print(f"  Done: {losses[0]:.4f} -> {losses[-1]:.4f}")
    return losses


def train_simple(model, n_steps, group_name):
    print(f"\n{'='*50}")
    print(f"Training Group {group_name} ({n_steps} steps)")
    print(f"{'='*50}")

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['lr'])
    losses = []

    for step in range(n_steps):
        token_ids = torch.randint(0, CONFIG['vocab_size'], (CONFIG['batch_size'], 64)).to(device)
        inputs = token_ids[:, :-1]
        targets = token_ids[:, 1:]

        optimizer.zero_grad()
        logits = model(inputs)
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, CONFIG['vocab_size']), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
        optimizer.step()
        losses.append(loss.item())

        if step % 200 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}", flush=True)

    print(f"  Done: {losses[0]:.4f} -> {losses[-1]:.4f}")
    return losses


# ═══════════════════════════════════════════════════════════
# 评测函数
# ═══════════════════════════════════════════════════════════

def measure_grade0_retention(encoder, layers, n_steps_list=[0, 5, 10, 20, 50, 100]):
    """测量grade-0在不同推理深度后的保持率"""
    results = {}
    test_ids = torch.randint(0, CONFIG['vocab_size'], (8, 64)).to(device)

    with torch.no_grad():
        mv_init = encoder(test_ids)
        g0_start, g0_end = GRADE_SLICES[0]
        grade0_init = mv_init[..., g0_start:g0_end].clone()

        mv = mv_init.clone()
        mv_prior = mv_init.clone()
        prev_n = 0

        for n in n_steps_list:
            additional = n - prev_n
            for _ in range(additional):
                for layer in layers:
                    mv = layer(mv, mv_prior)
            prev_n = n

            grade0_current = mv[..., g0_start:g0_end]
            sim = nn.CosineSimilarity(dim=-1)(
                grade0_init.reshape(-1, grade0_init.size(-1)),
                grade0_current.reshape(-1, grade0_current.size(-1))
            ).mean().item()
            l2_err = (grade0_init - grade0_current).norm().item()
            results[n] = {'cosine_sim': sim, 'l2_error': l2_err}
            print(f"    N={n}: cosine_sim={sim:.6f}, l2_err={l2_err:.2e}")

    return results


def measure_embedding_retention(model, n_steps_list=[0, 5, 10, 20, 50, 100]):
    """测量普通embedding在正交变换后的保持率"""
    results = {}
    test_ids = torch.randint(0, CONFIG['vocab_size'], (8, 64)).to(device)

    with torch.no_grad():
        init_embed = model.get_initial_embed(test_ids)
        x = init_embed.clone()

        for n in n_steps_list:
            # 对于正交模型，逐层变换
            x_temp = init_embed.clone()
            for i in range(min(n, model.n_layers)):
                Q = model.get_orthogonal_matrix(model.skew_params[i])
                x_temp = x_temp @ Q.T
                x_temp = torch.tanh(x_temp)
                x_temp = x_temp @ Q

            sim = nn.CosineSimilarity(dim=-1)(
                init_embed.reshape(-1, init_embed.size(-1)),
                x_temp.reshape(-1, x_temp.size(-1))
            ).mean().item()
            results[n] = {'cosine_sim': sim}
            print(f"    N={n}: cosine_sim={sim:.6f}")

    return results


def measure_entropy_change(encoder, layers, n_layers=50):
    """测量等变分量的信息熵随推理深度的变化"""
    test_ids = torch.randint(0, CONFIG['vocab_size'], (8, 64)).to(device)
    entropy_history = []

    with torch.no_grad():
        mv = encoder(test_ids)
        mv_prior = mv.clone()
        g1_start = GRADE_SLICES[1][0]
        g4_end = GRADE_SLICES[4][1]

        for i in range(n_layers):
            for layer in layers:
                mv = layer(mv, mv_prior)
            equivariant = mv[..., g1_start:g4_end]
            entropy_proxy = equivariant.norm(dim=-1).var().item()
            entropy_history.append(entropy_proxy)

    return entropy_history


# ═══════════════════════════════════════════════════════════
# 主实验
# ═══════════════════════════════════════════════════════════

def run_all():
    all_results = {}
    os.makedirs(CONFIG['log_path'], exist_ok=True)
    os.makedirs(CONFIG['save_path'], exist_ok=True)

    N_STEPS = CONFIG['n_steps']

    # ─── 组A1: BIIC + 充分Eraser (衰减率=0.5) ───
    print("\n" + "="*60)
    print("Group A1: BIIC Full (Eraser init=0.5)")
    print("="*60)
    enc_a1, layers_a1, dec_a1 = build_biic_model(eraser_init=0.0)
    losses_a1 = train_biic(enc_a1, layers_a1, dec_a1, N_STEPS, 'A1')
    print("  Measuring grade-0 retention...")
    ret_a1 = measure_grade0_retention(enc_a1, layers_a1)
    print("  Measuring entropy...")
    ent_a1 = measure_entropy_change(enc_a1, layers_a1, n_layers=20)
    all_results['A1'] = {'losses': losses_a1, 'retention': ret_a1, 'entropy': ent_a1,
                         'desc': 'BIIC Full, Eraser=0.5'}

    # ─── 组A2: BIIC + 弱Eraser (衰减率=0.01) ───
    print("\n" + "="*60)
    print("Group A2: BIIC Weak Eraser (init=0.01)")
    print("="*60)
    enc_a2, layers_a2, dec_a2 = build_biic_model(eraser_init=-4.6)
    losses_a2 = train_biic(enc_a2, layers_a2, dec_a2, N_STEPS, 'A2')
    print("  Measuring grade-0 retention...")
    ret_a2 = measure_grade0_retention(enc_a2, layers_a2)
    print("  Measuring entropy...")
    ent_a2 = measure_entropy_change(enc_a2, layers_a2, n_layers=20)
    all_results['A2'] = {'losses': losses_a2, 'retention': ret_a2, 'entropy': ent_a2,
                         'desc': 'BIIC Weak Eraser=0.01'}

    # ─── 组B: 正交Token (H1检验) ───
    print("\n" + "="*60)
    print("Group B: Orthogonal Token (H1 test)")
    print("="*60)
    model_b = OrthogonalTokenBaseline(CONFIG['vocab_size'], d_model=64, n_layers=12).to(device)
    losses_b = train_simple(model_b, N_STEPS, 'B')
    print("  Measuring retention...")
    ret_b = measure_embedding_retention(model_b)
    all_results['B'] = {'losses': losses_b, 'retention': ret_b,
                        'desc': 'Orthogonal Token + tanh (H1)'}

    # ─── 组C: 随机线性变换 (下界) ───
    print("\n" + "="*60)
    print("Group C: Linear Baseline (lower bound)")
    print("="*60)
    model_c = LinearBaseline(CONFIG['vocab_size'], d_model=64, n_layers=12).to(device)
    losses_c = train_simple(model_c, N_STEPS, 'C')
    all_results['C'] = {'losses': losses_c, 'desc': 'Linear + LayerNorm (lower bound)'}

    # ─── 组D: BIIC只用grade-0解码 (H2消融) ───
    print("\n" + "="*60)
    print("Group D: BIIC grade-0 only decode (H2 ablation)")
    print("="*60)
    enc_d, layers_d, _ = build_biic_model(eraser_init=0.0)
    dec_d = AllGradeDecoder(CONFIG['vocab_size'], CONFIG['n_channels'], CONFIG['d_hidden']).to(device)
    # 训练时只用grade-0
    optimizer_d = torch.optim.Adam(
        list(enc_d.parameters()) + list(layers_d.parameters()) + list(dec_d.parameters()), lr=CONFIG['lr'])
    losses_d = []
    for step in range(N_STEPS):
        token_ids = torch.randint(0, CONFIG['vocab_size'], (CONFIG['batch_size'], 64)).to(device)
        inputs, targets = token_ids[:, :-1], token_ids[:, 1:]
        optimizer_d.zero_grad()
        mv = enc_d(inputs)
        mv_prior = mv.clone().detach()
        for layer in layers_d:
            mv = layer(mv, mv_prior)
        logits = dec_d(mv, active_grades=[0])
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, CONFIG['vocab_size']), targets.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(enc_d.parameters()) + list(layers_d.parameters()) + list(dec_d.parameters()), 1.0)
        optimizer_d.step()
        losses_d.append(loss.item())
        if step % 200 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}", flush=True)
    print(f"  Done: {losses_d[0]:.4f} -> {losses_d[-1]:.4f}")
    all_results['D'] = {'losses': losses_d, 'desc': 'BIIC grade-0 only (H2 ablation)'}

    # ─── 组E: 2048维普通Embedding (H2维度匹配) ───
    print("\n" + "="*60)
    print("Group E: 2048-dim Embedding (H2 dim-matched)")
    print("="*60)
    model_e = DimMatchedBaseline(CONFIG['vocab_size'], d_model=2048, n_layers=6).to(device)
    losses_e = train_simple(model_e, N_STEPS, 'E')
    all_results['E'] = {'losses': losses_e, 'desc': '2048-dim Embedding + Orthogonal (H2)'}

    # ═══════════════════════════════════════════════════════════
    # 汇总
    # ═══════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("PHASE 3 RESULTS SUMMARY")
    print("="*60)

    print("\n--- Final Loss Comparison ---")
    for group, r in all_results.items():
        if r['losses']:
            init_l, final_l = r['losses'][0], r['losses'][-1]
            impr = (init_l - final_l) / init_l * 100
            print(f"  {group}: {init_l:.4f} -> {final_l:.4f} ({impr:.1f}% improvement) | {r['desc']}")

    print("\n--- Grade-0 Retention (BIIC groups) ---")
    for group in ['A1', 'A2']:
        if group in all_results and 'retention' in all_results[group]:
            ret = all_results[group]['retention']
            for n, v in ret.items():
                print(f"  {group} N={n}: cosine_sim={v['cosine_sim']:.6f}")

    print("\n--- Embedding Retention (Baseline groups) ---")
    if 'B' in all_results and 'retention' in all_results['B']:
        for n, v in all_results['B']['retention'].items():
            print(f"  B N={n}: cosine_sim={v['cosine_sim']:.6f}")

    print("\n--- Entropy Change (Eraser effectiveness) ---")
    for group in ['A1', 'A2']:
        if group in all_results and 'entropy' in all_results[group]:
            ent = all_results[group]['entropy']
            if ent:
                print(f"  {group}: {ent[0]:.4f} -> {ent[-1]:.4f} (ratio={ent[-1]/(ent[0]+1e-10):.2f})")

    print("\n--- Hypothesis Tests ---")
    loss_a1 = all_results['A1']['losses'][-1]
    loss_b = all_results['B']['losses'][-1]
    loss_c = all_results['C']['losses'][-1]
    loss_d = all_results['D']['losses'][-1]
    loss_e = all_results['E']['losses'][-1]

    print(f"  H1 (Geometry vs Orthogonal): A1={loss_a1:.4f} vs B={loss_b:.4f}")
    if loss_a1 < loss_b * 0.95:
        print(f"    -> BIIC wins: geometric structure has intrinsic value")
    elif loss_a1 > loss_b * 1.05:
        print(f"    -> Orthogonal wins: advantage is from orthogonal constraint")
    else:
        print(f"    -> Comparable: no clear winner")

    print(f"  H2 (Equivariance vs Dimension): A1={loss_a1:.4f} vs D={loss_d:.4f} vs E={loss_e:.4f}")
    if loss_a1 < loss_d and loss_a1 < loss_e:
        print(f"    -> Equivariant structure has independent value")
    elif loss_a1 < loss_d:
        print(f"    -> Equivariant grades help, but dimension may explain part")
    else:
        print(f"    -> No clear equivariance advantage")

    print(f"  H3 (Eraser): A1(strong)={all_results['A1']['losses'][-1]:.4f} vs A2(weak)={all_results['A2']['losses'][-1]:.4f}")
    if 'entropy' in all_results['A1'] and 'entropy' in all_results['A2']:
        e1, e2 = all_results['A1']['entropy'], all_results['A2']['entropy']
        if e1 and e2:
            if e1[-1] < e2[-1]:
                print(f"    -> Strong Eraser controls entropy better")
            else:
                print(f"    -> Eraser strength has limited effect")

    # 保存结果
    save_file = os.path.join(CONFIG['log_path'], f"phase3_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    save_data = {}
    for group, r in all_results.items():
        save_data[group] = {
            'losses_first10': r['losses'][:10],
            'losses_last10': r['losses'][-10:],
            'final_loss': r['losses'][-1],
            'init_loss': r['losses'][0],
            'desc': r['desc'],
        }
        if 'retention' in r:
            save_data[group]['retention'] = {str(k): v for k, v in r['retention'].items()}
        if 'entropy' in r:
            save_data[group]['entropy_first5'] = r['entropy'][:5]
            save_data[group]['entropy_last5'] = r['entropy'][-5:]

    with open(save_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved: {save_file}")

    return all_results


if __name__ == "__main__":
    results = run_all()
