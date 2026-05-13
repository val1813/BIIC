# /data/biic/experiments/probing.py

import torch
import torch.nn as nn
import sys
import json
import os
import numpy as np

sys.path.append('/data/biic')
sys.path.append('/data/biic/phase2')

from token_to_ic import TokenToImmutableCore
from mutable_state import BIICLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"设备: {device}")

# ─── 数据生成 ───

def generate_structured_data(n_samples=3000, seq_len=32, vocab_size=200):
    """
    生成有明确句法结构的toy数据集
    
    词汇表分区（让结构更清晰）：
      0-49:   名词（POS=0）
      50-99:  动词（POS=1）
      100-149: 形容词（POS=2）
      150-199: 其他（POS=3）
    
    依存关系（相邻token对）：
      名词+动词 → 主谓（DEP=0）
      动词+名词 → 谓宾（DEP=1）
      形容词+名词 → 修饰（DEP=2）
      其他组合 → 无关（DEP=3）
    """
    token_ids = torch.randint(0, vocab_size, (n_samples, seq_len))
    
    # POS标签
    pos = torch.zeros_like(token_ids)
    pos[(token_ids >= 50) & (token_ids < 100)] = 1
    pos[(token_ids >= 100) & (token_ids < 150)] = 2
    pos[token_ids >= 150] = 3
    
    # 依存关系标签（相邻对）
    dep = torch.zeros(n_samples, seq_len - 1, dtype=torch.long) + 3
    for i in range(seq_len - 1):
        p1 = pos[:, i]
        p2 = pos[:, i + 1]
        dep[(p1 == 0) & (p2 == 1), i] = 0  # 名词+动词
        dep[(p1 == 1) & (p2 == 0), i] = 1  # 动词+名词
        dep[(p1 == 2) & (p2 == 0), i] = 2  # 形容词+名词
    
    return token_ids, pos, dep


def get_grade_features(mv, grade):
    """提取指定grade的特征并展平"""
    grade_idx = {
        0: [0],
        1: [1, 2, 4, 8, 16],
        2: [3, 5, 6, 9, 10, 12, 17, 18, 20, 24],
        3: [7, 11, 13, 14, 19, 21, 22, 25, 26, 28],
        4: [15, 23, 27, 29, 30],
        5: [31]
    }
    data = mv[..., grade_idx[grade]]  # [B, L, C, dim]
    B, L, C, d = data.shape
    return data.reshape(B, L, C * d)  # [B, L, C*dim]


def train_linear_probe(X_train, y_train, X_val, y_val, n_classes, n_epochs=200):
    """训练线性探针，返回验证集准确率"""
    probe = nn.Linear(X_train.shape[-1], n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=5e-3)
    
    best_acc = 0
    for epoch in range(n_epochs):
        probe.train()
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(probe(X_train), y_train)
        loss.backward()
        opt.step()
        
        if (epoch + 1) % 50 == 0:
            probe.eval()
            with torch.no_grad():
                acc = (probe(X_val).argmax(-1) == y_val).float().mean().item()
            best_acc = max(best_acc, acc)
    
    return best_acc


# ─── 主实验 ───

def run_probing(n_steps_list=[0, 5, 10, 20, 50]):
    """
    完整Probing实验
    
    对比三种基线：
    1. 随机初始化的grade-2（纯噪声基线）
    2. 训练后的grade-0（不变核，上限参考）
    3. 训练后的grade-2（我们想检验的）
    4. 训练后的grade-2（经过N层推理后）
    """
    vocab_size = 200
    n_channels = 8
    
    print("="*60)
    print("BIIC Probing实验")
    print("检验grade-2是噪声还是有结构")
    print("="*60)
    
    # 生成数据
    token_ids, pos_labels, dep_labels = generate_structured_data(
        n_samples=3000, seq_len=32, vocab_size=vocab_size
    )
    token_ids = token_ids.to(device)
    
    # 80/20划分
    n = len(token_ids)
    split = int(n * 0.8)
    
    # 构建模型
    encoder = TokenToImmutableCore(vocab_size, n_channels).to(device)
    layers = nn.ModuleList([
        BIICLayer(n_channels, use_residual=True, use_eraser=True)
        for _ in range(50)
    ]).to(device)
    
    results = {
        'pos_probing': {},
        'dep_probing': {},
        'random_baseline_pos': 0.25,
        'random_baseline_dep': 0.25,
    }
    
    # ── 随机基线：随机初始化的grade-2 ──
    print("\n[基线] 随机初始化grade-2的POS探针准确率:")
    with torch.no_grad():
        mv_random = torch.randn(n, 32, n_channels, 32, device=device)
    g2_random = get_grade_features(mv_random, 2).reshape(-1, n_channels * 10).cpu()
    pos_flat = pos_labels.reshape(-1)
    
    split_feat = int(len(g2_random) * 0.8)
    random_acc = train_linear_probe(
        g2_random[:split_feat], pos_flat[:split_feat],
        g2_random[split_feat:], pos_flat[split_feat:],
        n_classes=4
    )
    results['random_grade2_pos'] = random_acc
    print(f"  随机grade-2 POS准确率: {random_acc:.3f}（理论随机基线: 0.25）")
    
    # ── 主实验：不同推理深度的grade-0和grade-2 ──
    print("\n[主实验] 各推理深度的探针准确率:")
    print(f"{'N步':>6} | {'grade-0 POS':>12} | {'grade-2 POS':>12} | {'grade-2 DEP':>12} | 信号")
    print("-" * 65)
    
    for n_steps in n_steps_list:
        with torch.no_grad():
            mv = encoder(token_ids)
            ic_g0 = mv[..., 0].detach()
            
            for i in range(n_steps):
                mv = layers[i](mv, ic_g0)
        
        # grade-0 POS探针
        g0_feat = get_grade_features(mv.detach(), 0)  # [B, L, C]
        B, L, C = g0_feat.shape
        g0_flat = g0_feat.reshape(B*L, C).cpu()
        pos_flat = pos_labels.reshape(-1)
        split_f = int(len(g0_flat) * 0.8)
        
        g0_pos_acc = train_linear_probe(
            g0_flat[:split_f], pos_flat[:split_f],
            g0_flat[split_f:], pos_flat[split_f:],
            n_classes=4
        )
        
        # grade-2 POS探针
        g2_feat = get_grade_features(mv.detach(), 2)  # [B, L, C*10]
        B, L, D = g2_feat.shape
        g2_flat = g2_feat.reshape(B*L, D).cpu()
        
        g2_pos_acc = train_linear_probe(
            g2_flat[:split_f], pos_flat[:split_f],
            g2_flat[split_f:], pos_flat[split_f:],
            n_classes=4
        )
        
        # grade-2 DEP探针（相邻token对）
        dep_flat = dep_labels.reshape(-1)
        g2_pairs = torch.cat([g2_feat[:, :-1, :], g2_feat[:, 1:, :]], dim=-1)
        B2, L2, D2 = g2_pairs.shape
        g2_pairs_flat = g2_pairs.reshape(B2*L2, D2).cpu()
        split_d = int(len(g2_pairs_flat) * 0.8)
        
        g2_dep_acc = train_linear_probe(
            g2_pairs_flat[:split_d], dep_flat[:split_d],
            g2_pairs_flat[split_d:], dep_flat[split_d:],
            n_classes=4
        )
        
        # 判断信号强度
        signal = "🔴 噪声" if g2_pos_acc < 0.35 else "🟡 弱信号" if g2_pos_acc < 0.55 else "🟢 强信号"
        
        print(f"{n_steps:>6} | {g0_pos_acc:>12.3f} | {g2_pos_acc:>12.3f} | {g2_dep_acc:>12.3f} | {signal}")
        
        results['pos_probing'][n_steps] = {
            'grade0': g0_pos_acc,
            'grade2': g2_pos_acc,
        }
        results['dep_probing'][n_steps] = g2_dep_acc
    
    # ── 结论 ──
    max_g2_pos = max(results['pos_probing'][n]['grade2'] for n in n_steps_list)
    max_dep = max(results['dep_probing'].values())
    
    print("\n" + "="*60)
    print("结论：")
    
    if max_g2_pos < 0.35:
        conclusion = "grade-2是纯噪声，需要更强的辅助损失引导"
        action = "在实验2中加入grade-2预测相邻token关系的辅助损失"
    elif max_g2_pos < 0.55:
        conclusion = "grade-2有弱信号，跨token通道能激活它"
        action = "实验2：直接加局部窗口注意力，应该能看到改善"
    else:
        conclusion = "grade-2已有强信号！加跨token通道应该效果显著"
        action = "实验2：优先级最高，预计改善明显"
    
    print(f"  grade-2 POS最高准确率: {max_g2_pos:.3f}")
    print(f"  grade-2 DEP最高准确率: {max_dep:.3f}")
    print(f"  结论: {conclusion}")
    print(f"  下一步: {action}")
    
    # 保存结果
    os.makedirs('/data/biic/experiments/results', exist_ok=True)
    with open('/data/biic/experiments/results/probing_results.json', 'w') as f:
        json.dump({
            'conclusion': conclusion,
            'action': action,
            'max_grade2_pos': max_g2_pos,
            'max_grade2_dep': max_dep,
            'random_grade2_pos': random_acc,
            'pos_probing': {str(k): v for k, v in results['pos_probing'].items()},
            'dep_probing': {str(k): v for k, v in results['dep_probing'].items()},
        }, f, indent=2)
    
    print(f"\n结果已保存: /data/biic/experiments/results/probing_results.json")
    return results


if __name__ == '__main__':
    run_probing()
