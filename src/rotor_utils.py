"""
Rotor Utilities for Cl(4,1)
旋转子相关工具：生成、变换、验证

旋转子 R 是偶数grade元素（grade 0+2+4），满足 R·R̃ = 1
通过双向量的指数映射生成：R = exp(B/2)，B是grade-2元素
Sandwich积：x' = R·x·R̃ 实现等变变换
"""

import torch
import numpy as np
from clifford_cl41 import (
    geometric_product_fast, reverse_multivector, grade_project,
    grade_extract, assemble_multivector, multivector_norm_sq,
    N_BLADES, N_DIM, GRADE_SLICES, METRIC
)


def generate_bivector(params):
    """从10个参数生成grade-2双向量
    params: [..., 10]
    返回: [..., 32] 只有grade-2非零的多向量
    """
    batch_shape = params.shape[:-1]
    device = params.device
    dtype = params.dtype

    mv = torch.zeros(*batch_shape, N_BLADES, device=device, dtype=dtype)
    start, end = GRADE_SLICES[2]
    mv[..., start:end] = params
    return mv


def bivector_norm_sq(B):
    """计算双向量的范数平方
    对于Cl(4,1)中的双向量，B·B̃ 的标量部分
    """
    B_rev = reverse_multivector(B)  # grade-2 reverse = -B
    product = geometric_product_fast(B, B_rev)
    return product[..., 0]  # 标量部分


def exp_bivector(B, n_terms=16):
    """双向量的指数映射（Taylor展开）
    R = exp(B) = 1 + B + B²/2! + B³/3! + ...

    B: [..., 32] grade-2多向量
    n_terms: Taylor展开项数
    返回: [..., 32] 旋转子
    """
    device = B.device
    dtype = B.dtype
    batch_shape = B.shape[:-1]

    # 初始化：R = 1 (标量)
    result = torch.zeros(*batch_shape, N_BLADES, device=device, dtype=dtype)
    result[..., 0] = 1.0

    # Taylor展开
    term = torch.zeros(*batch_shape, N_BLADES, device=device, dtype=dtype)
    term[..., 0] = 1.0  # B^0 = 1

    for k in range(1, n_terms):
        term = geometric_product_fast(term, B) / k
        result = result + term

    return result


def generate_pure_rotor(bivector_params, scale=0.1):
    """从双向量参数生成纯旋转子
    R = exp(B/2)，其中B是grade-2元素

    bivector_params: [..., 10] 双向量的10个分量
    scale: 缩放因子，控制旋转角度大小
    返回: [..., 32] 归一化旋转子
    """
    B = generate_bivector(bivector_params * scale)
    R = exp_bivector(B)
    R = normalize_rotor(R)
    return R


def sandwich_product(R, x, stabilize=True):
    """Sandwich积: x' = R*x*R_rev
    R: [..., 32] 旋转子
    x: [..., 32] 被变换的多向量
    stabilize: 是否稳定化（防止Cl(4,1)负度规导致的范数爆炸）
    返回: [..., 32] 变换后的多向量
    """
    R_rev = reverse_multivector(R)
    # x' = R * x * R_rev
    Rx = geometric_product_fast(R, x)
    RxR_rev = geometric_product_fast(Rx, R_rev)

    if stabilize:
        RxR_rev = _stabilize_multivector(RxR_rev, x)

    return RxR_rev


def _stabilize_multivector(x_new, x_old):
    """稳定化：防止Cl(4,1)负度规导致的范数爆炸
    对每个grade分别保范，保持等变性
    grade-0和grade-5是不变量，不动
    grade 1-4 各自独立保范
    """
    result = x_new.clone()

    for grade in [1, 2, 3, 4]:
        start, end = GRADE_SLICES[grade]
        old_norm = x_old[..., start:end].norm(dim=-1, keepdim=True)
        new_norm = x_new[..., start:end].norm(dim=-1, keepdim=True)
        scale = old_norm / (new_norm + 1e-10)
        result[..., start:end] = x_new[..., start:end] * scale

    return result


def rotor_reverse(R):
    """计算旋转子的逆 R̃
    对于归一化旋转子，R̃ = R的reversion
    """
    return reverse_multivector(R)


def normalize_rotor(R):
    """归一化旋转子，使 R·R̃ = 1
    防止数值漂移导致旋转子退化
    """
    norm_sq = multivector_norm_sq(R)  # [...] 标量
    # norm_sq 应该接近1，取绝对值防止负数（数值误差）
    norm = torch.sqrt(torch.abs(norm_sq) + 1e-10)
    return R / norm.unsqueeze(-1)


def verify_no_reflection(R, tol=1e-4):
    """验证旋转子不含奇数grade分量（纯旋转无反射）
    纯旋转子只有偶数grade: grade 0, 2, 4
    如果有grade 1, 3, 5 分量，说明包含反射

    R: [..., 32]
    返回: bool tensor [...] True表示是纯旋转子
    """
    odd_energy = torch.zeros(R.shape[:-1], device=R.device)

    for grade in [1, 3, 5]:
        start, end = GRADE_SLICES[grade]
        odd_energy += (R[..., start:end] ** 2).sum(dim=-1)

    total_energy = (R ** 2).sum(dim=-1)
    odd_ratio = odd_energy / (total_energy + 1e-10)

    return odd_ratio < tol


def verify_rotor_identity(R, tol=1e-4):
    """验证 R·R̃ ≈ 1
    R: [..., 32]
    返回: bool tensor, True表示满足条件
    """
    R_rev = reverse_multivector(R)
    product = geometric_product_fast(R, R_rev)

    # 应该是纯标量=1
    scalar_part = product[..., 0]
    non_scalar_energy = (product[..., 1:] ** 2).sum(dim=-1)

    scalar_ok = torch.abs(scalar_part - 1.0) < tol
    non_scalar_ok = non_scalar_energy < tol

    return scalar_ok & non_scalar_ok


def random_rotor(batch_shape=(), scale=0.3, device=None):
    """生成随机旋转子（用于测试）
    batch_shape: 批次形状
    scale: 旋转角度大小
    """
    params = torch.randn(*batch_shape, 10, device=device) * scale
    return generate_pure_rotor(params, scale=1.0)


if __name__ == "__main__":
    print("=== Rotor Utils Test ===")

    # 生成随机旋转子
    R = random_rotor(batch_shape=(4,), scale=0.5)
    print(f"Rotor shape: {R.shape}")

    # 验证 R·R̃ = 1
    is_valid = verify_rotor_identity(R)
    print(f"R·R̃ = 1: {is_valid.all().item()}")

    # 验证无反射
    is_pure = verify_no_reflection(R)
    print(f"No reflection: {is_pure.all().item()}")

    # 测试sandwich积
    x = torch.randn(4, N_BLADES)
    x_transformed = sandwich_product(R, x)
    print(f"Sandwich product shape: {x_transformed.shape}")

    # 验证grade-0不变性
    x_g0 = grade_extract(x, 0)
    x_t_g0 = grade_extract(x_transformed, 0)
    g0_error = (x_g0 - x_t_g0).abs().max().item()
    print(f"Grade-0 invariance error: {g0_error:.2e}")
