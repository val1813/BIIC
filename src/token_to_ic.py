"""
TokenToImmutableCore - BPE token到不变核的编码器

设计要点：
1. 各grade用不同scale初始化（防止初始化同质化）
2. grade-0用小scale（引导学到稳定的语义核心）
3. 各grade通过独立线性层产生（防止梯度混淆）
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/data/biic')
from clifford_cl41 import GRADE_SLICES, N_BLADES


class TokenToImmutableCore(nn.Module):
    """
    BPE token -> 不变核（多通道Cl(4,1)多向量）
    """

    def __init__(self, vocab_size=50257, n_channels=8):
        super().__init__()
        self.n_channels = n_channels
        self.vocab_size = vocab_size

        # 共享基础嵌入
        self.base_embed = nn.Embedding(vocab_size, 128)

        # 各grade独立的投影层（物理分离）
        self.to_grade0 = nn.Linear(128, n_channels * 1)
        self.to_grade1 = nn.Linear(128, n_channels * 5)
        self.to_grade2 = nn.Linear(128, n_channels * 10)
        self.to_grade3 = nn.Linear(128, n_channels * 10)
        self.to_grade4 = nn.Linear(128, n_channels * 5)
        self.to_grade5 = nn.Linear(128, n_channels * 1)

        # 差异化初始化
        nn.init.normal_(self.to_grade0.weight, std=0.01)
        nn.init.normal_(self.to_grade1.weight, std=0.1)
        nn.init.normal_(self.to_grade2.weight, std=0.05)
        nn.init.normal_(self.to_grade3.weight, std=0.1)
        nn.init.normal_(self.to_grade4.weight, std=0.05)
        nn.init.normal_(self.to_grade5.weight, std=0.01)

    def forward(self, token_ids):
        """
        token_ids: [B, L] 整数
        返回: [B, L, C, 32] 多通道多向量
        """
        B, L = token_ids.shape
        C = self.n_channels

        feat = self.base_embed(token_ids)  # [B, L, 128]

        # 各grade分别投影
        g0 = self.to_grade0(feat).view(B, L, C, 1)
        g1 = self.to_grade1(feat).view(B, L, C, 5)
        g2 = self.to_grade2(feat).view(B, L, C, 10)
        g3 = self.to_grade3(feat).view(B, L, C, 10)
        g4 = self.to_grade4(feat).view(B, L, C, 5)
        g5 = self.to_grade5(feat).view(B, L, C, 1)

        # 组装多向量 [B, L, C, 32]
        mv = torch.cat([g0, g1, g2, g3, g4, g5], dim=-1)  # [B, L, C, 32]

        return mv

    def get_grade_norms(self, mv):
        """返回各grade的平均L2范数，用于监控grade分离情况"""
        norms = {}
        for g in range(6):
            start, end = GRADE_SLICES[g]
            norms[g] = mv[..., start:end].norm(dim=-1).mean().item()
        return norms
