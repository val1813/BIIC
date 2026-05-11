"""
Cl(4,1) Clifford Algebra - PyTorch Native Implementation
度规签名: (+,+,+,+,-), 即 e1²=e2²=e3²=e4²=+1, e5²=-1
2^5 = 32 个基底元素

Grade分布:
  grade 0: 1 (标量)
  grade 1: 5 (向量 e1..e5)
  grade 2: 10 (双向量 e12,e13,...,e45)
  grade 3: 10 (三向量)
  grade 4: 5 (四向量)
  grade 5: 1 (伪标量 e12345)
"""

import torch
import numpy as np
from itertools import combinations
from functools import lru_cache


# ═══════════════════════════════════════════════════════════
# 基底定义
# ═══════════════════════════════════════════════════════════

N_DIM = 5
N_BLADES = 32  # 2^5
METRIC = [1, 1, 1, 1, -1]  # e1²=+1, ..., e4²=+1, e5²=-1


def _canonical_basis():
    """生成所有32个基底的索引表示
    每个基底用一个frozenset表示其包含的向量索引
    例如: e1 -> {0}, e12 -> {0,1}, e125 -> {0,1,4}
    """
    bases = []
    for grade in range(N_DIM + 1):
        for combo in combinations(range(N_DIM), grade):
            bases.append(tuple(sorted(combo)))
    return bases


BASIS = _canonical_basis()  # 32个基底
BASIS_TO_IDX = {b: i for i, b in enumerate(BASIS)}


def _grade_of(basis_tuple):
    """返回基底的grade"""
    return len(basis_tuple)


# 预计算grade范围
GRADE_SLICES = {}
idx = 0
for g in range(N_DIM + 1):
    count = len([b for b in BASIS if len(b) == g])
    GRADE_SLICES[g] = (idx, idx + count)
    idx += count


def _sign_of_product(a_tuple, b_tuple):
    """计算两个基底相乘的符号和结果基底
    使用冒泡排序计算交换次数，处理平方项
    返回 (sign, result_tuple) 或 (0, None) 如果结果为零
    """
    # 合并两个基底的索引列表
    combined = list(a_tuple) + list(b_tuple)
    sign = 1

    # 冒泡排序，每次交换翻转符号
    n = len(combined)
    for i in range(n):
        for j in range(n - 1 - i):
            if combined[j] > combined[j + 1]:
                combined[j], combined[j + 1] = combined[j + 1], combined[j]
                sign *= -1
            elif combined[j] == combined[j + 1]:
                # 相同索引相邻，用度规消除
                metric_val = METRIC[combined[j]]
                sign *= metric_val
                combined[j] = -1  # 标记删除
                combined[j + 1] = -1
                break
        else:
            continue
        break

    # 递归处理直到没有重复
    # 更简洁的实现：直接用完整算法
    return _compute_product_sign(list(a_tuple), list(b_tuple))


def _compute_product_sign(a_list, b_list):
    """完整的几何积符号计算"""
    combined = a_list + b_list
    sign = 1

    # 冒泡排序到有序，计算交换次数
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(combined) - 1:
            if combined[i] > combined[i + 1]:
                combined[i], combined[i + 1] = combined[i + 1], combined[i]
                sign *= -1
                changed = True
            elif combined[i] == combined[i + 1]:
                # e_i * e_i = metric[i]
                sign *= METRIC[combined[i]]
                combined.pop(i)
                combined.pop(i)
                changed = True
                continue
            i += 1

    result_tuple = tuple(combined)
    return sign, result_tuple


# ═══════════════════════════════════════════════════════════
# 乘法表预计算
# ═══════════════════════════════════════════════════════════

def build_multiplication_table():
    """预计算32x32乘法表
    返回:
      signs: [32, 32] 符号矩阵 (+1/-1/0)
      indices: [32, 32] 结果基底索引
    """
    signs = np.zeros((N_BLADES, N_BLADES), dtype=np.int8)
    indices = np.zeros((N_BLADES, N_BLADES), dtype=np.int32)

    for i, bi in enumerate(BASIS):
        for j, bj in enumerate(BASIS):
            s, result = _compute_product_sign(list(bi), list(bj))
            if result in BASIS_TO_IDX:
                signs[i, j] = s
                indices[i, j] = BASIS_TO_IDX[result]
            else:
                signs[i, j] = 0
                indices[i, j] = 0

    return signs, indices


# 全局预计算乘法表
_MULT_SIGNS, _MULT_INDICES = build_multiplication_table()
MULT_SIGNS = torch.tensor(_MULT_SIGNS, dtype=torch.float32)
MULT_INDICES = torch.tensor(_MULT_INDICES, dtype=torch.long)


# ═══════════════════════════════════════════════════════════
# 核心运算
# ═══════════════════════════════════════════════════════════

def geometric_product(a, b):
    """计算两个多向量的几何积
    a, b: [..., 32] 多向量
    返回: [..., 32] 几何积结果

    使用预计算乘法表，避免循环
    """
    # a: [..., 32], b: [..., 32]
    batch_shape = a.shape[:-1]
    device = a.device

    signs = MULT_SIGNS.to(device)    # [32, 32]
    indices = MULT_INDICES.to(device)  # [32, 32]

    # a[..., i] * b[..., j] * sign[i,j] 累加到 result[..., indices[i,j]]
    # 展开计算
    a_flat = a.reshape(-1, N_BLADES)  # [B, 32]
    b_flat = b.reshape(-1, N_BLADES)  # [B, 32]
    B = a_flat.shape[0]

    result = torch.zeros(B, N_BLADES, device=device, dtype=a.dtype)

    for i in range(N_BLADES):
        for j in range(N_BLADES):
            s = signs[i, j].item()
            if s != 0:
                idx = indices[i, j].item()
                result[:, idx] += s * a_flat[:, i] * b_flat[:, j]

    return result.reshape(*batch_shape, N_BLADES)


def geometric_product_fast(a, b):
    """优化版几何积 - 使用向量化操作
    a, b: [..., 32]
    """
    device = a.device
    batch_shape = a.shape[:-1]

    signs = MULT_SIGNS.to(device)      # [32, 32]
    indices = MULT_INDICES.to(device)   # [32, 32]

    a_flat = a.reshape(-1, N_BLADES)  # [B, 32]
    b_flat = b.reshape(-1, N_BLADES)  # [B, 32]
    B = a_flat.shape[0]

    # 计算所有 a_i * b_j 的外积
    outer = a_flat.unsqueeze(2) * b_flat.unsqueeze(1)  # [B, 32, 32]
    # 乘以符号
    outer = outer * signs.unsqueeze(0)  # [B, 32, 32]

    # 按目标索引累加
    result = torch.zeros(B, N_BLADES, device=device, dtype=a.dtype)
    indices_flat = indices.reshape(-1)  # [1024]
    outer_flat = outer.reshape(B, -1)   # [B, 1024]

    result.scatter_add_(1, indices_flat.unsqueeze(0).expand(B, -1), outer_flat)

    return result.reshape(*batch_shape, N_BLADES)


def grade_project(mv, grade):
    """从多向量中提取特定grade的分量
    mv: [..., 32]
    grade: 0-5
    返回: [..., 32] 只保留指定grade的分量，其余为0
    """
    start, end = GRADE_SLICES[grade]
    result = torch.zeros_like(mv)
    result[..., start:end] = mv[..., start:end]
    return result


def grade_extract(mv, grade):
    """提取特定grade的分量值（不补零）
    mv: [..., 32]
    grade: 0-5
    返回: [..., n_components] 该grade的分量
    """
    start, end = GRADE_SLICES[grade]
    return mv[..., start:end]


def assemble_multivector(components_dict):
    """从各grade分量组装完整多向量
    components_dict: {grade: tensor} 每个tensor形状为 [..., n_components_of_grade]
    返回: [..., 32]
    """
    # 确定batch shape
    sample = next(iter(components_dict.values()))
    batch_shape = sample.shape[:-1]
    device = sample.device
    dtype = sample.dtype

    mv = torch.zeros(*batch_shape, N_BLADES, device=device, dtype=dtype)

    for grade, values in components_dict.items():
        start, end = GRADE_SLICES[grade]
        expected_size = end - start
        assert values.shape[-1] == expected_size, \
            f"Grade {grade} expects {expected_size} components, got {values.shape[-1]}"
        mv[..., start:end] = values

    return mv


def reverse_multivector(mv):
    """计算多向量的逆（reversion）
    grade k 的分量乘以 (-1)^(k*(k-1)/2)
    grade 0: +1
    grade 1: +1
    grade 2: -1
    grade 3: -1
    grade 4: +1
    grade 5: +1
    """
    result = mv.clone()
    rev_signs = [1, 1, -1, -1, 1, 1]

    for grade in range(N_DIM + 1):
        start, end = GRADE_SLICES[grade]
        result[..., start:end] *= rev_signs[grade]

    return result


def multivector_norm_sq(mv):
    """计算多向量的范数平方: mv * ~mv 的标量部分"""
    rev = reverse_multivector(mv)
    product = geometric_product_fast(mv, rev)
    return product[..., 0]  # grade-0 分量


if __name__ == "__main__":
    print("=== Cl(4,1) Clifford Algebra ===")
    print(f"Dimension: {N_DIM}, Blades: {N_BLADES}")
    print(f"Metric: {METRIC}")
    print(f"Grade slices: {GRADE_SLICES}")

    # 验证基底平方
    for i in range(N_DIM):
        ei = torch.zeros(N_BLADES)
        ei[i + 1] = 1.0  # grade-1 从索引1开始
        ei_sq = geometric_product_fast(ei.unsqueeze(0), ei.unsqueeze(0))
        print(f"  e{i+1}² = {ei_sq[0, 0].item():.1f} (expected: {METRIC[i]})")

    print("\nAll basis elements:")
    for i, b in enumerate(BASIS):
        grade = len(b)
        name = "1" if not b else "e" + "".join(str(x+1) for x in b)
        print(f"  [{i:2d}] grade-{grade}: {name}")
