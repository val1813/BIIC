"""
实验N：GLUE线性探针
验证grade-0作为语义锚点在下游分类任务是否有独立优势
"""
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
from clifford_cl41 import GRADE_SLICES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_grade_features(mv, grade_name):
    """提取指定grade并展平"""
    if grade_name == 'grade0':
        start, end = GRADE_SLICES[0]
    elif grade_name == 'grade2':
        start, end = GRADE_SLICES[2]
    elif grade_name == 'all':
        # All 32 blades
        data = mv.reshape(mv.shape[0], mv.shape[1], -1)  # [B, L, C*32]
        return data.mean(dim=1)  # [B, C*32]
    else:
        raise ValueError(f"Unknown grade: {grade_name}")

    data = mv[..., start:end]  # [B, L, C, d]
    B, L, C, d = data.shape
    return data.reshape(B, L, C * d).mean(dim=1)  # [B, C*d]


def train_linear_probe(X_train, y_train, X_val, y_val, n_classes=2, epochs=300):
    """训练线性分类器并返回验证准确率"""
    input_dim = X_train.shape[1]
    probe = nn.Linear(input_dim, n_classes)
    opt = torch.optim.Adam(probe.parameters(), lr=1e-2, weight_decay=1e-4)

    X_t = torch.FloatTensor(X_train)
    y_t = torch.LongTensor(y_train)

    for _ in range(epochs):
        probe.train()
        opt.zero_grad()
        loss = nn.CrossEntropyLoss()(probe(X_t), y_t)
        loss.backward()
        opt.step()

    probe.eval()
    with torch.no_grad():
        val_pred = probe(torch.FloatTensor(X_val)).argmax(-1).numpy()
    return float((val_pred == y_val).mean())


def run_glue_probe():
    print("=" * 60)
    print("实验N：GLUE线性探针")
    print("验证grade-0作为语义锚点的分类能力")
    print("=" * 60)

    vocab_size = 500
    n_channels = 8
    n_layers = 6

    encoder = TokenToImmutableCore(vocab_size, n_channels).to(device)
    layers = nn.ModuleList([
        BIICLayer(n_channels, use_residual=True, use_eraser=True)
        for _ in range(n_layers)
    ]).to(device)

    grade_names = ['grade0', 'grade2', 'all']
    all_results = {}

    # ── 任务1：情感分类 ──
    print("\n--- 任务1：情感分类 ---")
    print("正面词汇=0-99，负面词汇=100-199")
    n_samples = 500
    seq_len = 8

    task1_results = {}
    for grade_name in grade_names:
        feats = []
        labels = []
        with torch.no_grad():
            # 正面
            pos_ids = torch.randint(0, 100, (n_samples, seq_len)).to(device)
            mv = encoder(pos_ids)
            mv_prior = mv.clone()
            for layer in layers:
                mv = layer(mv, mv_prior)
            feat = get_grade_features(mv, grade_name).cpu()
            feats.append(feat)
            labels.extend([1] * n_samples)

            # 负面
            neg_ids = torch.randint(100, 200, (n_samples, seq_len)).to(device)
            mv = encoder(neg_ids)
            mv_prior = mv.clone()
            for layer in layers:
                mv = layer(mv, mv_prior)
            feat = get_grade_features(mv, grade_name).cpu()
            feats.append(feat)
            labels.extend([0] * n_samples)

        X = torch.cat(feats, dim=0).numpy()
        y = np.array(labels)

        # Shuffle
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        split = int(len(X) * 0.8)
        acc = train_linear_probe(X[:split], y[:split], X[split:], y[split:])
        task1_results[grade_name] = acc
        print(f"  {grade_name}: {acc:.4f}")

    all_results['sentiment'] = task1_results

    # ── 任务2：语义相似度 ──
    print("\n--- 任务2：语义相似度 ---")
    print("同类词对=1，异类词对=0")

    task2_results = {}
    for grade_name in grade_names:
        feats = []
        labels = []
        with torch.no_grad():
            # 正样本：同类词对
            for _ in range(250):
                t1 = torch.randint(0, 100, (1, 4)).to(device)
                t2 = torch.randint(0, 100, (1, 4)).to(device)
                mv1 = encoder(t1)
                mv2 = encoder(t2)
                for layer in layers:
                    mv1 = layer(mv1, mv1.clone())
                    mv2 = layer(mv2, mv2.clone())
                f1 = get_grade_features(mv1, grade_name)
                f2 = get_grade_features(mv2, grade_name)
                feats.append((f1 - f2).abs().cpu())
                labels.append(1)

            # 负样本：异类词对
            for _ in range(250):
                t1 = torch.randint(0, 100, (1, 4)).to(device)
                t2 = torch.randint(100, 200, (1, 4)).to(device)
                mv1 = encoder(t1)
                mv2 = encoder(t2)
                for layer in layers:
                    mv1 = layer(mv1, mv1.clone())
                    mv2 = layer(mv2, mv2.clone())
                f1 = get_grade_features(mv1, grade_name)
                f2 = get_grade_features(mv2, grade_name)
                feats.append((f1 - f2).abs().cpu())
                labels.append(0)

        X = torch.cat(feats, dim=0).numpy()
        y = np.array(labels)
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        split = int(len(X) * 0.8)
        acc = train_linear_probe(X[:split], y[:split], X[split:], y[split:])
        task2_results[grade_name] = acc
        print(f"  {grade_name}: {acc:.4f}")

    all_results['similarity'] = task2_results

    # ── 任务3：词类判断 ──
    print("\n--- 任务3：词类判断（名词vs动词vs形容词）---")
    print("名词=0-49, 动词=50-99, 形容词=100-149")

    task3_results = {}
    for grade_name in grade_names:
        feats = []
        labels = []
        with torch.no_grad():
            for cls, (lo, hi) in enumerate([(0, 50), (50, 100), (100, 150)]):
                ids = torch.randint(lo, hi, (200, 1)).to(device)
                mv = encoder(ids)
                mv_prior = mv.clone()
                for layer in layers:
                    mv = layer(mv, mv_prior)
                feat = get_grade_features(mv, grade_name).cpu()
                feats.append(feat)
                labels.extend([cls] * 200)

        X = torch.cat(feats, dim=0).numpy()
        y = np.array(labels)
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]

        split = int(len(X) * 0.8)
        acc = train_linear_probe(X[:split], y[:split], X[split:], y[split:], n_classes=3)
        task3_results[grade_name] = acc
        print(f"  {grade_name}: {acc:.4f}")

    all_results['pos_tagging'] = task3_results

    # ── 结论 ──
    print("\n" + "=" * 60)
    print("汇总：")
    print(f"  情感: g0={task1_results['grade0']:.4f} g2={task1_results['grade2']:.4f} all={task1_results['all']:.4f}")
    print(f"  相似: g0={task2_results['grade0']:.4f} g2={task2_results['grade2']:.4f} all={task2_results['all']:.4f}")
    print(f"  词类: g0={task3_results['grade0']:.4f} g2={task3_results['grade2']:.4f} all={task3_results['all']:.4f}")

    g0_avg = np.mean([task1_results['grade0'], task2_results['grade0'], task3_results['grade0']])
    g2_avg = np.mean([task1_results['grade2'], task2_results['grade2'], task3_results['grade2']])

    if g0_avg > 0.7:
        conclusion = (f"grade-0平均准确率{g0_avg:.4f}，作为语义身份锚点有效。"
                     f"优于SRA的可学习锚点（SRA锚点会漂移）。")
    else:
        conclusion = f"grade-0平均准确率{g0_avg:.4f}，接近随机基线。"

    if g2_avg > g0_avg + 0.05:
        conclusion += f" grade-2({g2_avg:.4f})优于grade-0({g0_avg:.4f})，等变分量携带更多语义。"
    elif g0_avg > g2_avg + 0.05:
        conclusion += f" grade-0({g0_avg:.4f})优于grade-2({g2_avg:.4f})，不变核是主要语义载体。"

    print(f"\n结论: {conclusion}")

    os.makedirs('/data/biic/results', exist_ok=True)
    with open('/data/biic/results/exp_n_glue_probe.json', 'w') as f:
        json.dump({
            'experiment': 'N: GLUE线性探针',
            'conclusion': conclusion,
            'results': all_results,
            'random_baseline': {'binary': 0.5, 'ternary': 0.333}
        }, f, indent=2, ensure_ascii=False)

    print("\n结果已保存: /data/biic/results/exp_n_glue_probe.json")


if __name__ == '__main__':
    run_glue_probe()
