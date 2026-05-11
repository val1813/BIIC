"""
MutableState - 可变态推理层

包含：
- SimpleWriter: 通过旋转子对可变态做等变变换
- MutableStateLayer: Writer + GradeAwareEraser
- BIICLayer: 完整推理层（可配置残差和Eraser）
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/data/biic')
from clifford_cl41 import geometric_product_fast, GRADE_SLICES, N_BLADES
from rotor_utils import normalize_rotor, sandwich_product, exp_bivector
from eraser_ops import GradeAwareEraser


class SimpleWriter(nn.Module):
    """
    Writer算子：根据可学习参数对可变态做等变变换

    实现：每个通道有独立的双向量参数，生成旋转子做sandwich积
    """

    def __init__(self, n_channels, n_bivector=10):
        super().__init__()
        self.n_channels = n_channels
        # 可学习的双向量参数（生成旋转子）
        self.bivector_params = nn.Parameter(
            torch.randn(n_channels, n_bivector) * 0.01
        )

    def forward(self, mv):
        """
        mv: [B, L, C, 32]
        对每个通道应用独立的旋转变换
        """
        B, L, C, D = mv.shape
        results = []

        bv_start, bv_end = GRADE_SLICES[2]  # grade-2: index 6:16

        for c in range(C):
            # 构建双向量多向量
            bv_mv = torch.zeros(1, N_BLADES, device=mv.device, dtype=mv.dtype)
            bv_mv[0, bv_start:bv_end] = self.bivector_params[c]

            # 生成旋转子 R = exp(B/2)
            R = exp_bivector(bv_mv * 0.5, n_terms=8)
            R = normalize_rotor(R)

            # 对该通道所有位置做sandwich积
            mv_c = mv[:, :, c, :]  # [B, L, 32]
            B_size, L_size = mv_c.shape[:2]
            mv_c_flat = mv_c.reshape(-1, N_BLADES)  # [B*L, 32]
            R_expanded = R.expand(mv_c_flat.shape[0], -1)  # [B*L, 32]

            mv_c_transformed = sandwich_product(R_expanded, mv_c_flat)
            results.append(mv_c_transformed.reshape(B_size, L_size, N_BLADES))

        # 用stack+permute避免inplace操作
        return torch.stack(results, dim=2)  # [B, L, C, 32]


class MutableStateLayer(nn.Module):
    """
    可变态的单层更新
    顺序：Writer -> GradeAwareEraser
    """

    def __init__(self, n_channels):
        super().__init__()
        self.writer = SimpleWriter(n_channels)
        self.eraser = GradeAwareEraser(n_channels)

    def forward(self, mv, mv_prior):
        """
        mv: [B, L, C, 32] 当前多向量状态
        mv_prior: [B, L, C, 32] 不变核（作为Eraser的先验）
        """
        # 1. Writer：等变变换
        mv = self.writer(mv)

        # 2. GradeAwareEraser：衰减grade 1-4，grade-0/5不动
        mv = self.eraser(mv, mv_prior)

        return mv


class BIICLayer(nn.Module):
    """
    完整的BIIC推理层
    可配置：是否使用残差、是否使用Eraser
    """

    def __init__(self, n_channels, use_residual=True, use_eraser=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_eraser = use_eraser
        self.n_channels = n_channels
        self.mutable_layer = MutableStateLayer(n_channels)

    def forward(self, mv, mv_prior=None):
        """
        mv: [B, L, C, 32]
        mv_prior: [B, L, C, 32] 不变核（Eraser先验）
        """
        if self.use_eraser and mv_prior is not None:
            mv_new = self.mutable_layer(mv, mv_prior)
        else:
            mv_new = self.mutable_layer.writer(mv)

        if self.use_residual:
            # 残差连接：只对grade 1-4加残差，grade-0/5保持不变
            result = mv.clone()
            g1_start = GRADE_SLICES[1][0]
            g4_end = GRADE_SLICES[4][1]
            result[..., g1_start:g4_end] = (
                mv[..., g1_start:g4_end] + mv_new[..., g1_start:g4_end]
            ) * 0.5  # 平均而非直接加，防止爆炸
            return result
        else:
            return mv_new
