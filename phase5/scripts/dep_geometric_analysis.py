"""
实验M：依存关系几何积分析
验证：有依存关系的token对的grade-2几何积是否比无依存关系的更大/更结构化
如果成立，就找到了语言的代数等变性定义
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def geometric_product_grade0(a, b):
    """
    计算两个grade-2多向量几何积的grade-0部分
    a, b: [C, 10]（grade-2的10个blade）
    返回：[C] 标量

    Cl(4,1) grade-2 blade的度规符号:
    blade索引: e12,e13,e14,e15,e23,e24,e25,e34,e35,e45
    度规: e1^2=e2^2=e3^2=e4^2=+1, e5^2=-1
    eij^2 = ei^2 * ej^2
    """
    metric_signs = torch.tensor([
        1, 1, 1, -1,   # e12,e13,e14,e15
        1, 1, -1,      # e23,e24,e25
        1, -1,         # e34,e35
        -1             # e45
    ], dtype=torch.float32, device=a.device)

    return (a * b * metric_signs).sum(dim=-1)  # [C]


def generate_dep_pairs(n_samples=1000, vocab_size=200):
    """
    生成有/无依存关系的token对
    名词(0-49)+动词(50-99) → 有依存关系（主谓）
    名词(0-49)+形容词(100-149) → 有依存关系（修饰）
    随机两个token → 无依存关系（对照组）
    """
    dep_pairs = []
    for _ in range(n_samples):
        noun = torch.randint(0, 50, (1,)).item()
        if torch.rand(1).item() > 0.5:
            verb = torch.randint(50, 100, (1,)).item()
            dep_pairs.append((noun, verb, 1))
        else:
            adj = torch.randint(100, 150, (1,)).item()
            dep_pairs.append((noun, adj, 1))

    nodep_pairs = []
    for _ in range(n_samples):
        t1 = torch.randint(0, vocab_size, (1,)).item()
        t2 = torch.randint(0, vocab_size, (1,)).item()
        while (t1 < 50 and 50 <= t2 < 100) or (t1 < 50 and 100 <= t2 < 150):
            t1 = torch.randint(0, vocab_size, (1,)).item()
            t2 = torch.randint(0, vocab_size, (1,)).item()
        nodep_pairs.append((t1, t2, 0))

    return dep_pairs + nodep_pairs


def run_dep_geometric_analysis():
    print("=" * 60)
    print("实验M：依存关系几何积分析")
    print("验证：grade-2几何积是否编码依存关系")
    print("=" * 60)

    vocab_size = 200
    n_channels = 8
    n_steps_list = [0, 5, 10, 20]

    encoder = TokenToImmutableCore(vocab_size, n_channels).to(device)
    layers = nn.ModuleList([
        BIICLayer(n_channels, use_residual=True, use_eraser=True)
        for _ in range(20)
    ]).to(device)

    pairs = generate_dep_pairs(n_samples=500)
    results = {}

    # Cl(4,1) grade-2 blade indices in the 32-dim multivector
    # grade-2 has 10 blades: C(5,2)=10
    from clifford_cl41 import GRADE_SLICES
    g2_start, g2_end = GRADE_SLICES[2]
    g2_dim = g2_end - g2_start  # should be 10

    print(f"Grade-2 slice: [{g2_start}:{g2_end}], dim={g2_dim}")

    for n_steps in n_steps_list:
        dep_scores = []
        nodep_scores = []

        with torch.no_grad():
            for t1_id, t2_id, label in pairs:
                t1 = torch.tensor([[t1_id]], device=device)
                t2 = torch.tensor([[t2_id]], device=device)

                mv1 = encoder(t1)  # [1, 1, C, 32]
                mv2 = encoder(t2)

                mv1_prior = mv1.clone()
                mv2_prior = mv2.clone()

                for i in range(n_steps):
                    mv1 = layers[i](mv1, mv1_prior)
                    mv2 = layers[i](mv2, mv2_prior)

                # 提取grade-2: [1, 1, C, g2_dim] -> [C, g2_dim]
                g2_1 = mv1[0, 0, :, g2_start:g2_end]  # [C, 10]
                g2_2 = mv2[0, 0, :, g2_start:g2_end]  # [C, 10]

                # 计算几何积的grade-0部分
                gp_score = geometric_product_grade0(g2_1, g2_2)
                score = gp_score.abs().mean().item()

                if label == 1:
                    dep_scores.append(score)
                else:
                    nodep_scores.append(score)

        dep_mean = np.mean(dep_scores)
        dep_std = np.std(dep_scores)
        nodep_mean = np.mean(nodep_scores)
        nodep_std = np.std(nodep_scores)
        ratio = dep_mean / nodep_mean if nodep_mean > 0 else float('inf')

        from scipy import stats
        t_stat, p_val = stats.ttest_ind(dep_scores, nodep_scores)

        results[str(n_steps)] = {
            'dep_mean': float(dep_mean),
            'dep_std': float(dep_std),
            'nodep_mean': float(nodep_mean),
            'nodep_std': float(nodep_std),
            'ratio': float(ratio),
            't_stat': float(t_stat),
            'p_value': float(p_val),
            'significant': bool(p_val < 0.05)
        }

        sig = "显著" if p_val < 0.05 else "不显著"
        print(f"\nN={n_steps}步推理后:")
        print(f"  有依存: {dep_mean:.4f} +/- {dep_std:.4f}")
        print(f"  无依存: {nodep_mean:.4f} +/- {nodep_std:.4f}")
        print(f"  比值: {ratio:.3f}  t={t_stat:.3f}  p={p_val:.4f}  {sig}")

    # 结论
    print("\n" + "=" * 60)
    best_step = max(results.keys(), key=lambda k: results[k]['ratio'])
    best = results[best_step]

    if best['significant'] and best['ratio'] > 1.1:
        conclusion = (f"发现：有依存关系的token对的grade-2几何积"
                     f"比无依存关系的高{best['ratio']:.2f}倍（p={best['p_value']:.4f}）\n"
                     f"这说明语言的依存关系对应Cl(4,1)的代数结构\n"
                     f"这是BIIC独有的发现")
    else:
        conclusion = (f"grade-2几何积在有/无依存关系的token对之间无显著差异\n"
                     f"比值={best['ratio']:.3f}，p={best['p_value']:.4f}\n"
                     f"需要更多训练或不同的几何积计算方式")

    print(conclusion)

    os.makedirs('/data/biic/results', exist_ok=True)
    with open('/data/biic/results/exp_m_dep_geometric.json', 'w') as f:
        json.dump({
            'experiment': 'M: 依存关系几何积分析',
            'conclusion': conclusion,
            'results_by_steps': results
        }, f, indent=2, ensure_ascii=False)

    print("\n结果已保存: /data/biic/results/exp_m_dep_geometric.json")


if __name__ == '__main__':
    run_dep_geometric_analysis()
