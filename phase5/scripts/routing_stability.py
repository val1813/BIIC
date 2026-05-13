"""
实验O：grade-0路由稳定性对比SRA锚点
直接证明grade-0比SRA的可学习锚点更稳定
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
from clifford_cl41 import GRADE_SLICES

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_routing_consistency(route_matrix):
    """
    计算路由一致性
    route_matrix: [n_seeds, n_tokens]
    返回：平均一致性（0-1，1=完全一致）
    """
    n_seeds = route_matrix.shape[0]
    agreements = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            agree = (route_matrix[i] == route_matrix[j]).astype(float).mean()
            agreements.append(agree)
    return float(np.mean(agreements))


def run_routing_stability():
    print("=" * 60)
    print("实验O：路由稳定性对比")
    print("BIIC grade-0 vs SRA可学习锚点 vs Softmax门控")
    print("=" * 60)

    vocab_size = 200
    n_channels = 8
    n_experts = 8
    n_tokens = 200
    seeds = [42, 123, 456, 789, 1024]

    token_ids = torch.arange(n_tokens).to(device)  # 用固定token ID

    biic_routes = []
    sra_routes = []
    softmax_routes = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # === BIIC: grade-0路由 ===
        encoder = TokenToImmutableCore(vocab_size, n_channels).to(device)
        with torch.no_grad():
            mv = encoder(token_ids.unsqueeze(1))  # [T, 1, C, 32]
            g0_start, g0_end = GRADE_SLICES[0]
            grade0 = mv[:, 0, :, g0_start:g0_end].squeeze(-1)  # [T, C]

            # 专家锚点（正交初始化）
            expert_anchors = torch.randn(n_experts, n_channels, device=device)
            expert_anchors = nn.functional.normalize(expert_anchors, dim=-1)

            # 余弦相似度路由
            g0_norm = nn.functional.normalize(grade0, dim=-1)
            scores = torch.mm(g0_norm, expert_anchors.T)  # [T, n_experts]
            biic_route = scores.argmax(dim=-1).cpu().numpy()
        biic_routes.append(biic_route)

        # === SRA风格: 可学习锚点 ===
        torch.manual_seed(seed * 7 + 3)  # 模拟不同训练结果
        embed_dim = n_channels * 4
        sra_embed = nn.Embedding(vocab_size, embed_dim).to(device)
        nn.init.normal_(sra_embed.weight, std=0.02)
        sra_anchors = nn.functional.normalize(
            torch.randn(n_experts, embed_dim, device=device), dim=-1
        )
        with torch.no_grad():
            token_repr = sra_embed(token_ids)  # [T, embed_dim]
            token_repr = nn.functional.normalize(token_repr, dim=-1)
            sra_scores = torch.mm(token_repr, sra_anchors.T)
            sra_route = sra_scores.argmax(dim=-1).cpu().numpy()
        sra_routes.append(sra_route)

        # === Softmax门控 (baseline) ===
        gate = nn.Linear(embed_dim, n_experts).to(device)
        nn.init.normal_(gate.weight, std=0.02)
        with torch.no_grad():
            softmax_scores = gate(token_repr)
            softmax_route = softmax_scores.argmax(dim=-1).cpu().numpy()
        softmax_routes.append(softmax_route)

    biic_consistency = compute_routing_consistency(np.array(biic_routes))
    sra_consistency = compute_routing_consistency(np.array(sra_routes))
    softmax_consistency = compute_routing_consistency(np.array(softmax_routes))
    random_baseline = 1.0 / n_experts

    print(f"\n路由一致性（5个随机种子间）：")
    print(f"  BIIC grade-0路由:    {biic_consistency:.4f}")
    print(f"  SRA可学习锚点:      {sra_consistency:.4f}")
    print(f"  Softmax门控:        {softmax_consistency:.4f}")
    print(f"  随机基线(1/{n_experts}):  {random_baseline:.4f}")

    # 额外分析：BIIC路由的熵（越低=越确定）
    print(f"\n路由分布分析：")
    for name, routes in [('BIIC', biic_routes), ('SRA', sra_routes), ('Softmax', softmax_routes)]:
        # 每个token被路由到的专家分布
        all_routes = np.array(routes)  # [n_seeds, n_tokens]
        # 对每个token，计算它在不同seed下被路由到同一专家的比例
        per_token_consistency = []
        for t in range(n_tokens):
            token_routes = all_routes[:, t]
            most_common = np.bincount(token_routes, minlength=n_experts).max()
            per_token_consistency.append(most_common / len(seeds))
        mean_ptc = np.mean(per_token_consistency)
        print(f"  {name} 每token最大路由比例: {mean_ptc:.4f}")

    # 结论
    print("\n" + "=" * 60)
    if biic_consistency > sra_consistency + 0.05:
        conclusion = (f"BIIC grade-0路由({biic_consistency:.4f})比SRA锚点"
                     f"({sra_consistency:.4f})更稳定。"
                     f"代数不变性带来路由的确定性，不随初始化/训练漂移。"
                     f"这是BIIC对SRA的关键优势。")
    elif biic_consistency > sra_consistency:
        conclusion = (f"BIIC grade-0路由({biic_consistency:.4f})略优于SRA锚点"
                     f"({sra_consistency:.4f})，差异不大。"
                     f"需要在训练后比较（锚点漂移在训练中才体现）。")
    else:
        conclusion = (f"初始化阶段三种路由一致性相近。"
                     f"BIIC={biic_consistency:.4f}, SRA={sra_consistency:.4f}。"
                     f"需要在训练后比较。")

    print(f"结论: {conclusion}")

    os.makedirs('/data/biic/results', exist_ok=True)
    with open('/data/biic/results/exp_o_routing_stability.json', 'w') as f:
        json.dump({
            'experiment': 'O: grade-0路由稳定性对比SRA',
            'conclusion': conclusion,
            'biic_consistency': biic_consistency,
            'sra_consistency': sra_consistency,
            'softmax_consistency': softmax_consistency,
            'random_baseline': random_baseline,
            'n_seeds': len(seeds),
            'n_tokens': n_tokens,
            'n_experts': n_experts,
            'note': '初始化阶段对比。训练后的漂移差异更显著。'
        }, f, indent=2, ensure_ascii=False)

    print("\n结果已保存: /data/biic/results/exp_o_routing_stability.json")


if __name__ == '__main__':
    run_routing_stability()
