"""
Eraser Operator - 受控衰减算子
DNA类比：去甲基化酶（不是全清，是受控衰减到先验态）

核心功能：
- 向先验衰减：ρ_{t+1} = (1-λ)ρ_t + λ·ρ_prior
- 随机重置（训练时极低概率，防止状态僵化）
- 可学习衰减率（每个维度独立）
"""

import torch
import torch.nn as nn


class EraserOperator(nn.Module):
    """
    受控衰减Eraser算子

    DNA类比：去甲基化酶
    - 不是全清，是受控衰减到语义先验
    - 每个维度有独立的衰减率（可学习）
    - 极低概率随机重置（防止状态僵化）
    """

    def __init__(self, state_dim, n_channels):
        """
        state_dim: 状态维度（密度矩阵的大小）
        n_channels: 通道数
        """
        super().__init__()
        self.state_dim = state_dim
        self.n_channels = n_channels

        # 每个通道每个维度的可学习衰减率（核心参数）
        # 初始化为较小值（衰减慢），让模型自己学习何时快速遗忘
        # 用logit空间存储，sigmoid后映射到(0,1)
        self.decay_logits = nn.Parameter(
            torch.ones(n_channels, state_dim) * (-4.6)  # sigmoid(-4.6) ≈ 0.01
        )

        # 随机重置概率（极低，防止状态僵化）
        self.reset_logit = nn.Parameter(torch.tensor(-7.0))  # sigmoid(-7) ≈ 0.001

    def forward(self, rho, rho_prior, step=None):
        """
        rho: 当前可变态 [B, L, C, D] 或 [B, L, C, D, D]
        rho_prior: 不变核提供的语义先验，形状同rho
        step: 当前训练步数（可选，用于调度）

        返回: 衰减后的状态，形状同rho
        """
        # 计算衰减率 λ ∈ (0, 1)
        decay = torch.sigmoid(self.decay_logits)  # [C, D]

        # 扩展decay到匹配rho的形状
        if rho.dim() == 4:
            # rho: [B, L, C, D]
            decay_expanded = decay.unsqueeze(0).unsqueeze(0)  # [1, 1, C, D]
        elif rho.dim() == 5:
            # rho: [B, L, C, D, D] (密度矩阵)
            decay_expanded = decay.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, C, D, 1]
        else:
            raise ValueError(f"Unexpected rho shape: {rho.shape}")

        # 模式1：向先验衰减（主要模式）
        # ρ_{t+1} = (1-λ)ρ_t + λ·ρ_prior
        rho_decayed = (1 - decay_expanded) * rho + decay_expanded * rho_prior

        # 模式2：随机重置（极低概率，仅训练时）
        if self.training:
            reset_prob = torch.sigmoid(self.reset_logit)
            reset_mask = torch.bernoulli(
                torch.full(rho.shape[:3], reset_prob.item(), device=rho.device)
            ).bool()  # [B, L, C]

            # 扩展mask到完整维度
            if rho.dim() == 4:
                reset_mask_expanded = reset_mask.unsqueeze(-1)  # [B, L, C, 1]
            else:
                reset_mask_expanded = reset_mask.unsqueeze(-1).unsqueeze(-1)  # [B, L, C, 1, 1]

            rho_decayed = torch.where(reset_mask_expanded, rho_prior, rho_decayed)

        return rho_decayed

    def get_decay_rates(self):
        """返回当前衰减率（用于监控和分析）"""
        return torch.sigmoid(self.decay_logits).detach()

    def get_reset_prob(self):
        """返回当前重置概率"""
        return torch.sigmoid(self.reset_logit).detach().item()


class GradeAwareEraser(nn.Module):
    """
    Grade感知的Eraser - 对不同grade施加不同衰减策略

    关键设计：
    - grade-0（不变核）：衰减率=0，绝对不动
    - grade-1~4（可变态）：可学习衰减率
    - grade-5（伪标量）：衰减率=0，保持不变
    """

    def __init__(self, n_channels, n_blades=32):
        super().__init__()
        self.n_channels = n_channels
        self.n_blades = n_blades

        # 只对grade 1-4的分量施加衰减（共30个分量）
        # grade 0: index 0 (1个) - 不衰减
        # grade 1: index 1-5 (5个) - 可衰减
        # grade 2: index 6-15 (10个) - 可衰减
        # grade 3: index 16-25 (10个) - 可衰减
        # grade 4: index 26-30 (5个) - 可衰减
        # grade 5: index 31 (1个) - 不衰减

        # 可衰减的分量：index 1-30 (30个)
        self.decay_logits = nn.Parameter(
            torch.ones(n_channels, 30) * (-4.6)
        )

    def forward(self, mv, mv_prior):
        """
        mv: [..., C, 32] 多向量
        mv_prior: [..., C, 32] 先验多向量
        返回: [..., C, 32] 衰减后的多向量
        """
        decay = torch.sigmoid(self.decay_logits)  # [C, 30]

        result = mv.clone()

        # grade 0 和 grade 5 不动
        # grade 1-4 (index 1:31) 施加衰减
        mutable_slice = slice(1, 31)

        # 扩展decay到匹配形状
        batch_dims = mv.dim() - 2  # 除了C和32之外的维度数
        decay_expanded = decay
        for _ in range(batch_dims):
            decay_expanded = decay_expanded.unsqueeze(0)
        # decay_expanded: [1, ..., 1, C, 30]

        result[..., mutable_slice] = (
            (1 - decay_expanded) * mv[..., mutable_slice]
            + decay_expanded * mv_prior[..., mutable_slice]
        )

        return result

    def get_decay_rates(self):
        """返回衰减率"""
        return torch.sigmoid(self.decay_logits).detach()


if __name__ == "__main__":
    print("=== Eraser Operator Test ===")

    # 基础Eraser测试
    eraser = EraserOperator(state_dim=16, n_channels=8)
    print(f"Decay rates range: [{eraser.get_decay_rates().min():.4f}, {eraser.get_decay_rates().max():.4f}]")
    print(f"Reset prob: {eraser.get_reset_prob():.6f}")

    # 模拟衰减过程
    B, L, C, D = 2, 4, 8, 16
    rho = torch.randn(B, L, C, D)
    rho_prior = torch.zeros(B, L, C, D)  # 先验为零

    print(f"\nBefore eraser: rho norm = {rho.norm():.4f}")
    rho_after = eraser(rho, rho_prior)
    print(f"After eraser:  rho norm = {rho_after.norm():.4f}")

    # 多步衰减
    rho_current = rho.clone()
    for step in range(100):
        rho_current = eraser(rho_current, rho_prior)
    print(f"After 100 steps: rho norm = {rho_current.norm():.4f}")
    print(f"Converged to prior: {torch.allclose(rho_current, rho_prior, atol=0.1)}")

    # Grade-aware Eraser测试
    print("\n=== Grade-Aware Eraser ===")
    ga_eraser = GradeAwareEraser(n_channels=8)
    mv = torch.randn(2, 4, 8, 32)
    mv_prior = torch.zeros_like(mv)

    mv_after = ga_eraser(mv, mv_prior)
    # grade-0 应该不变
    g0_diff = (mv_after[..., 0] - mv[..., 0]).abs().max().item()
    # grade-5 应该不变
    g5_diff = (mv_after[..., 31] - mv[..., 31]).abs().max().item()
    print(f"Grade-0 change: {g0_diff:.2e} (should be 0)")
    print(f"Grade-5 change: {g5_diff:.2e} (should be 0)")
    print(f"Grade-1~4 changed: {(mv_after[..., 1:31] - mv[..., 1:31]).abs().max().item():.4f}")
