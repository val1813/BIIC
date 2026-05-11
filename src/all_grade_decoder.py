"""
AllGradeDecoder - 全grade解码器（核心原创贡献）

设计原则：
- 各grade独立投影到d_hidden
- 可学习的grade重要性权重（grade_gates）
- 支持active_grades参数，用于消融实验

Cl(4,1)各grade维度：
  grade-0: 1维  (blade索引: [0])
  grade-1: 5维  (blade索引: [1:6])
  grade-2: 10维 (blade索引: [6:16])
  grade-3: 10维 (blade索引: [16:26])
  grade-4: 5维  (blade索引: [26:31])
  grade-5: 1维  (blade索引: [31])

注意：这里用的是Phase 1 clifford_cl41.py中GRADE_SLICES定义的连续索引
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/data/biic')
from clifford_cl41 import GRADE_SLICES, N_BLADES


class AllGradeDecoder(nn.Module):
    """
    全grade解码器

    从多通道Cl(4,1)多向量中提取所有grade的信息，
    通过可学习的gate融合后输出logits
    """

    GRADE_DIMS = {0: 1, 1: 5, 2: 10, 3: 10, 4: 5, 5: 1}

    def __init__(self, vocab_size, n_channels=8, d_hidden=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_channels = n_channels
        self.d_hidden = d_hidden

        # 各grade独立的投影层
        self.grade_projectors = nn.ModuleDict()
        for g, dim in self.GRADE_DIMS.items():
            # 输入: [B, L, C * dim]，输出: [B, L, d_hidden]
            self.grade_projectors[str(g)] = nn.Sequential(
                nn.Linear(n_channels * dim, d_hidden),
                nn.GELU(),
                nn.Linear(d_hidden, d_hidden),
            )

        # 可学习的grade重要性门控
        # 初始化为0（sigmoid=0.5），让模型自己学习哪些grade重要
        self.grade_gates = nn.ParameterDict({
            str(g): nn.Parameter(torch.zeros(1)) for g in range(6)
        })

        # 输出投影
        self.output_proj = nn.Sequential(
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, vocab_size),
        )

    def forward(self, mv, active_grades=None):
        """
        mv: [B, L, C, 32] 多通道多向量
        active_grades: 可选，指定使用哪些grade（用于消融实验）
                       None表示使用全部grade
        返回: [B, L, vocab_size]
        """
        B, L, C, D = mv.shape
        assert D == N_BLADES, f"Expected {N_BLADES} blades, got {D}"

        if active_grades is None:
            active_grades = list(range(6))

        fused = None

        for g in active_grades:
            start, end = GRADE_SLICES[g]
            # 提取该grade的分量: [B, L, C, grade_dim]
            g_data = mv[..., start:end]
            # 展平通道维度: [B, L, C * grade_dim]
            g_flat = g_data.reshape(B, L, -1)

            # 投影到d_hidden
            projected = self.grade_projectors[str(g)](g_flat)  # [B, L, d_hidden]

            # 门控
            gate = torch.sigmoid(self.grade_gates[str(g)])

            if fused is None:
                fused = gate * projected
            else:
                fused = fused + gate * projected

        if fused is None:
            fused = torch.zeros(B, L, self.d_hidden, device=mv.device)

        logits = self.output_proj(fused)  # [B, L, vocab_size]
        return logits

    def get_grade_weights(self):
        """返回各grade的实际权重，用于分析"""
        return {int(g): torch.sigmoid(self.grade_gates[g]).item()
                for g in self.grade_gates}
