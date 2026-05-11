"""
BIICLoss - 分阶段辅助损失

目的：引导grade分工，防止所有信息挤入同一个grade

辅助损失1：grade-0预测原始token ID -> 引导grade-0学到token的核心语义
辅助损失2：grade方差约束 -> 防止grade坍缩

退火策略：
  前10%：alpha=1.0（强引导）
  10%-50%：线性退火到0.01
  50%后：alpha=0.01（永久结构正则化，不退到0）
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/data/biic')
from clifford_cl41 import GRADE_SLICES


class BIICLoss(nn.Module):

    def __init__(self, vocab_size, n_channels=8):
        super().__init__()
        # 辅助分类头：grade-0 -> token ID
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
        """
        mv: [B, L, C, 32] 编码器输出的多向量
        main_logits: [B, L, vocab_size] 主解码器输出
        targets: [B, L] 目标token ID
        """
        B, L, C, _ = mv.shape

        # 主损失
        main_loss = nn.CrossEntropyLoss()(
            main_logits.reshape(-1, main_logits.size(-1)),
            targets.reshape(-1)
        )

        alpha = self.get_alpha(step, total_steps)

        if alpha <= 0.01:
            return main_loss, {'main': main_loss.item(), 'alpha': alpha}

        # 辅助损失1：grade-0预测token ID
        g0_start, g0_end = GRADE_SLICES[0]
        grade0 = mv[..., g0_start:g0_end].squeeze(-1)  # [B, L, C]
        aux1_logits = self.grade0_classifier(grade0)  # [B, L, vocab_size]
        aux1_loss = nn.CrossEntropyLoss()(
            aux1_logits.reshape(-1, aux1_logits.size(-1)),
            targets.reshape(-1)
        )

        # 辅助损失2：grade方差约束（防坍缩）
        variance_loss = torch.tensor(0.0, device=mv.device)
        for g in [0, 1, 2]:
            start, end = GRADE_SLICES[g]
            g_data = mv[..., start:end]  # [B, L, C, dim]
            # 最大化通道间方差
            channel_var = g_data.var(dim=-2).mean()
            variance_loss = variance_loss - channel_var

        total_loss = main_loss + alpha * (0.6 * aux1_loss + 0.4 * variance_loss)

        return total_loss, {
            'main': main_loss.item(),
            'aux1_grade0_clf': aux1_loss.item(),
            'aux2_variance': variance_loss.item(),
            'alpha': alpha
        }
