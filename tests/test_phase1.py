"""
Phase 1 测试套件 - BIIC数学验证
10个测试，全部PASS才能进入Phase 2

测试1-7: Clifford代数和旋转子基础
测试8-10: Eraser算子验证
"""

import torch
import numpy as np
import sys

from clifford_cl41 import (
    geometric_product_fast, grade_project, grade_extract,
    assemble_multivector, reverse_multivector, multivector_norm_sq,
    N_BLADES, N_DIM, GRADE_SLICES, METRIC, BASIS
)
from rotor_utils import (
    generate_pure_rotor, sandwich_product, rotor_reverse,
    normalize_rotor, verify_no_reflection, verify_rotor_identity,
    random_rotor, generate_bivector
)
from eraser_ops import EraserOperator, GradeAwareEraser


def test_header(num, name):
    print(f"\n{'='*60}")
    print(f"  Test {num}: {name}")
    print(f"{'='*60}")


def report(passed, details=""):
    status = "PASS" if passed else "FAIL"
    color = "\033[92m" if passed else "\033[91m"
    reset = "\033[0m"
    print(f"  Result: {color}{status}{reset}")
    if details:
        print(f"  {details}")
    return passed


# ═══════════════════════════════════════════════════════════
# Test 1: Cl(4,1) 乘法表正确性
# ═══════════════════════════════════════════════════════════

def test_1_multiplication_table():
    test_header(1, "Cl(4,1) Multiplication Table Correctness")
    print("  Verifying: e_i^2 = metric[i] for all basis vectors")

    all_ok = True
    max_error = 0.0

    for i in range(N_DIM):
        # 构造基底向量 e_{i+1}
        ei = torch.zeros(1, N_BLADES)
        start = GRADE_SLICES[1][0]  # grade-1起始索引
        ei[0, start + i] = 1.0

        # 计算 e_i^2
        ei_sq = geometric_product_fast(ei, ei)
        scalar_val = ei_sq[0, 0].item()
        expected = METRIC[i]
        error = abs(scalar_val - expected)
        max_error = max(max_error, error)

        # 非标量部分应为0
        non_scalar_error = ei_sq[0, 1:].abs().max().item()
        max_error = max(max_error, non_scalar_error)

        status = "OK" if error < 1e-5 else "FAIL"
        print(f"    e{i+1}^2 = {scalar_val:.6f} (expected {expected}) [{status}]")

        if error >= 1e-5 or non_scalar_error >= 1e-5:
            all_ok = False

    # 额外验证：e_i * e_j = -e_j * e_i (i≠j)
    print("  Verifying: e_i*e_j = -e_j*e_i for i≠j")
    for i in range(N_DIM):
        for j in range(i + 1, N_DIM):
            ei = torch.zeros(1, N_BLADES)
            ej = torch.zeros(1, N_BLADES)
            start = GRADE_SLICES[1][0]
            ei[0, start + i] = 1.0
            ej[0, start + j] = 1.0

            eij = geometric_product_fast(ei, ej)
            eji = geometric_product_fast(ej, ei)
            anticommute_error = (eij + eji).abs().max().item()
            max_error = max(max_error, anticommute_error)

            if anticommute_error >= 1e-5:
                all_ok = False
                print(f"    e{i+1}*e{j+1} + e{j+1}*e{i+1} error: {anticommute_error:.2e}")

    print(f"  Anti-commutativity: max error = {max_error:.2e}")
    return report(all_ok and max_error < 1e-5, f"Max error: {max_error:.2e}")


# ═══════════════════════════════════════════════════════════
# Test 2: 旋转子验证（无反射）
# ═══════════════════════════════════════════════════════════

def test_2_rotor_verification():
    test_header(2, "Rotor Verification (No Reflection)")
    print("  Verifying: R*R_rev = 1, no odd-grade components")

    n_tests = 50
    all_identity = True
    all_pure = True
    max_identity_error = 0.0
    max_odd_ratio = 0.0

    for _ in range(n_tests):
        R = random_rotor(batch_shape=(), scale=0.5)

        # 验证 R*R_rev = 1
        R_rev = reverse_multivector(R.unsqueeze(0))
        product = geometric_product_fast(R.unsqueeze(0), R_rev)
        scalar_error = abs(product[0, 0].item() - 1.0)
        non_scalar_error = product[0, 1:].abs().max().item()
        identity_error = max(scalar_error, non_scalar_error)
        max_identity_error = max(max_identity_error, identity_error)

        if identity_error >= 1e-5:
            all_identity = False

        # 验证无奇数grade
        is_pure = verify_no_reflection(R.unsqueeze(0), tol=1e-4)
        if not is_pure.all():
            all_pure = False

        # 计算奇数grade能量比
        odd_energy = 0.0
        for g in [1, 3, 5]:
            s, e = GRADE_SLICES[g]
            odd_energy += (R[s:e] ** 2).sum().item()
        total_energy = (R ** 2).sum().item()
        odd_ratio = odd_energy / (total_energy + 1e-10)
        max_odd_ratio = max(max_odd_ratio, odd_ratio)

    print(f"  R*R_rev = 1: max error = {max_identity_error:.2e}")
    print(f"  Odd-grade ratio: max = {max_odd_ratio:.2e}")

    passed = all_identity and all_pure and max_identity_error < 1e-5
    return report(passed, f"Identity error: {max_identity_error:.2e}, Odd ratio: {max_odd_ratio:.2e}")


# ═══════════════════════════════════════════════════════════
# Test 3: Grade-0 不变性（核心测试）
# ═══════════════════════════════════════════════════════════

def test_3_grade0_invariance():
    test_header(3, "Grade-0 Invariance (CORE TEST)")
    print("  Verifying: grade-0 unchanged after 100 sandwich transforms")

    n_trials = 10
    n_transforms = 100
    max_error = 0.0

    for trial in range(n_trials):
        # 创建随机多向量
        x = torch.randn(1, N_BLADES)
        original_g0 = grade_extract(x, 0).clone()

        # 连续100次sandwich变换
        current = x.clone()
        for _ in range(n_transforms):
            R = random_rotor(batch_shape=(1,), scale=0.2)
            current = sandwich_product(R, current)

        # 检查grade-0是否不变
        final_g0 = grade_extract(current, 0)
        error = (final_g0 - original_g0).abs().max().item()
        max_error = max(max_error, error)

    print(f"  After {n_transforms} transforms x {n_trials} trials")
    print(f"  Max grade-0 error: {max_error:.2e}")

    passed = max_error < 1e-4
    return report(passed, f"Max error: {max_error:.2e} (threshold: 1e-4)")


# ═══════════════════════════════════════════════════════════
# Test 4: Grade-5 不变性
# ═══════════════════════════════════════════════════════════

def test_4_grade5_invariance():
    test_header(4, "Grade-5 Invariance (Pseudoscalar)")
    print("  Verifying: grade-5 unchanged under pure rotors")

    n_trials = 10
    n_transforms = 100
    max_error = 0.0

    for trial in range(n_trials):
        x = torch.randn(1, N_BLADES)
        original_g5 = grade_extract(x, 5).clone()

        current = x.clone()
        for _ in range(n_transforms):
            R = random_rotor(batch_shape=(1,), scale=0.2)
            current = sandwich_product(R, current)

        final_g5 = grade_extract(current, 5)
        error = (final_g5 - original_g5).abs().max().item()
        max_error = max(max_error, error)

    print(f"  After {n_transforms} transforms x {n_trials} trials")
    print(f"  Max grade-5 error: {max_error:.2e}")

    passed = max_error < 1e-4
    return report(passed, f"Max error: {max_error:.2e} (threshold: 1e-4)")


# ═══════════════════════════════════════════════════════════
# Test 5: Grade-1 等变性
# ═══════════════════════════════════════════════════════════

def test_5_grade1_equivariance():
    test_header(5, "Grade-1 Equivariance")
    print("  Verifying: project(R*x*R_rev, 1) == R*project(x, 1)*R_rev")

    n_trials = 20
    max_error = 0.0

    for _ in range(n_trials):
        x = torch.randn(1, N_BLADES)
        R = random_rotor(batch_shape=(1,), scale=0.5)

        # 方法1：先变换完整多向量，再提取grade-1
        x_transformed = sandwich_product(R, x)
        g1_after_transform = grade_project(x_transformed, 1)

        # 方法2：先提取grade-1，再变换
        g1_before = grade_project(x, 1)
        g1_transformed = sandwich_product(R, g1_before)

        # 两种方法应该给出相同结果
        error = (g1_after_transform - g1_transformed).abs().max().item()
        max_error = max(max_error, error)

    print(f"  Max equivariance error: {max_error:.2e}")

    passed = max_error < 1e-4
    return report(passed, f"Max error: {max_error:.2e} (threshold: 1e-4)")


# ═══════════════════════════════════════════════════════════
# Test 6: 梯度流测试
# ═══════════════════════════════════════════════════════════

def test_6_gradient_flow():
    test_header(6, "Gradient Flow (10-layer chain)")
    print("  Verifying: gradient ratio between first and last layer in [0.1, 10]")

    n_layers = 10

    # 创建可学习的输入
    x = torch.randn(1, N_BLADES, requires_grad=True)

    # 10层sandwich变换链
    current = x
    rotors = []
    for i in range(n_layers):
        params = torch.randn(1, 10) * 0.2
        params.requires_grad_(True)
        R = generate_pure_rotor(params, scale=1.0)
        rotors.append(params)
        current = sandwich_product(R, current)

    # 计算loss并反向传播
    loss = current.sum()
    loss.backward()

    # 检查梯度
    grad_input = x.grad.norm().item()
    grad_first_rotor = rotors[0].grad.norm().item() if rotors[0].grad is not None else 0
    grad_last_rotor = rotors[-1].grad.norm().item() if rotors[-1].grad is not None else 0

    print(f"  Input gradient norm: {grad_input:.4f}")
    print(f"  First rotor gradient norm: {grad_first_rotor:.4f}")
    print(f"  Last rotor gradient norm: {grad_last_rotor:.4f}")

    if grad_first_rotor > 0 and grad_last_rotor > 0:
        ratio = grad_first_rotor / grad_last_rotor
        print(f"  Gradient ratio (first/last): {ratio:.4f}")
        passed = 0.1 <= ratio <= 10.0
    else:
        ratio = 0
        passed = False
        print("  WARNING: Zero gradient detected")

    return report(passed, f"Ratio: {ratio:.4f} (acceptable: [0.1, 10])")


# ═══════════════════════════════════════════════════════════
# Test 7: 多通道独立性
# ═══════════════════════════════════════════════════════════

def test_7_multichannel_independence():
    test_header(7, "Multi-channel Independence (C=8)")
    print("  Verifying: grade-0 of each channel is independent")

    C = 8  # 8个通道

    # 创建8通道多向量 [C, 32]
    x = torch.randn(C, N_BLADES)
    original_g0 = grade_extract(x, 0).clone()  # [C, 1]

    # 只对通道0施加变换
    R = random_rotor(batch_shape=(1,), scale=0.5)
    x_ch0_transformed = sandwich_product(R, x[0:1])

    # 其他通道不变
    x_modified = x.clone()
    x_modified[0:1] = x_ch0_transformed

    # 检查通道1-7的grade-0是否不变
    max_error = 0.0
    for ch in range(1, C):
        g0_original = original_g0[ch]
        g0_after = grade_extract(x_modified[ch:ch+1], 0)
        error = (g0_after - g0_original).abs().max().item()
        max_error = max(max_error, error)

    # 通道0的grade-0应该不变（sandwich积保持grade-0）
    g0_ch0_after = grade_extract(x_modified[0:1], 0)
    g0_ch0_error = (g0_ch0_after - original_g0[0:1]).abs().max().item()

    print(f"  Other channels grade-0 max change: {max_error:.2e}")
    print(f"  Channel 0 grade-0 invariance: {g0_ch0_error:.2e}")

    passed = max_error < 1e-10 and g0_ch0_error < 1e-5
    return report(passed, f"Cross-channel leakage: {max_error:.2e}")


# ═══════════════════════════════════════════════════════════
# Test 8: Eraser 衰减收敛性
# ═══════════════════════════════════════════════════════════

def test_8_eraser_convergence():
    test_header(8, "Eraser Decay Convergence")
    print("  Verifying: after N steps, rho converges to rho_prior")

    eraser = EraserOperator(state_dim=16, n_channels=8)
    eraser.eval()  # 关闭随机重置

    # 手动设置较大的衰减率以加速收敛测试
    with torch.no_grad():
        eraser.decay_logits.fill_(0.0)  # sigmoid(0) = 0.5

    B, L, C, D = 2, 4, 8, 16
    rho = torch.randn(B, L, C, D)
    rho_prior = torch.randn(B, L, C, D) * 0.1  # 先验

    # 多步衰减
    rho_current = rho.clone()
    n_steps = 200

    for step in range(n_steps):
        rho_current = eraser(rho_current, rho_prior)

    # 检查是否收敛到先验
    convergence_error = (rho_current - rho_prior).abs().max().item()
    print(f"  After {n_steps} steps with decay=0.5:")
    print(f"  Max |rho - rho_prior| = {convergence_error:.2e}")

    # 理论上 (1-0.5)^200 ≈ 6.2e-61，应该完全收敛
    passed = convergence_error < 1e-5
    return report(passed, f"Convergence error: {convergence_error:.2e}")


# ═══════════════════════════════════════════════════════════
# Test 9: 不变核不受Eraser影响
# ═══════════════════════════════════════════════════════════

def test_9_immutable_core_preserved():
    test_header(9, "Immutable Core Not Affected by Eraser")
    print("  Verifying: grade-0 unchanged after Eraser operations")

    ga_eraser = GradeAwareEraser(n_channels=8)
    ga_eraser.eval()

    # 设置较大衰减率
    with torch.no_grad():
        ga_eraser.decay_logits.fill_(2.0)  # sigmoid(2) ≈ 0.88

    # 创建多向量
    B, L, C = 2, 4, 8
    mv = torch.randn(B, L, C, 32)
    mv_prior = torch.randn(B, L, C, 32) * 0.1

    original_g0 = mv[..., 0].clone()  # grade-0
    original_g5 = mv[..., 31].clone()  # grade-5

    # 多次Eraser操作
    mv_current = mv.clone()
    for _ in range(50):
        mv_current = ga_eraser(mv_current, mv_prior)

    # grade-0 和 grade-5 应该完全不变
    g0_error = (mv_current[..., 0] - original_g0).abs().max().item()
    g5_error = (mv_current[..., 31] - original_g5).abs().max().item()

    # grade 1-4 应该已经衰减
    mutable_change = (mv_current[..., 1:31] - mv[..., 1:31]).abs().max().item()

    print(f"  Grade-0 change: {g0_error:.2e} (should be 0)")
    print(f"  Grade-5 change: {g5_error:.2e} (should be 0)")
    print(f"  Grade 1-4 change: {mutable_change:.4f} (should be > 0)")

    passed = g0_error < 1e-10 and g5_error < 1e-10 and mutable_change > 0.01
    return report(passed, f"Grade-0 error: {g0_error:.2e}, Grade-5 error: {g5_error:.2e}")


# ═══════════════════════════════════════════════════════════
# Test 10: 衰减率梯度健康
# ═══════════════════════════════════════════════════════════

def test_10_decay_gradient_health():
    test_header(10, "Decay Rate Gradient Health")
    print("  Verifying: decay_logits gradients are non-zero and healthy")

    eraser = EraserOperator(state_dim=16, n_channels=8)
    eraser.train()

    B, L, C, D = 2, 4, 8, 16
    rho = torch.randn(B, L, C, D, requires_grad=True)
    rho_prior = torch.randn(B, L, C, D)

    # 前向传播
    rho_out = eraser(rho, rho_prior)

    # 构造loss
    loss = rho_out.sum()
    loss.backward()

    # 检查衰减率参数的梯度
    decay_grad = eraser.decay_logits.grad
    assert decay_grad is not None, "No gradient for decay_logits!"

    grad_norm = decay_grad.norm().item()
    grad_max = decay_grad.abs().max().item()
    grad_min = decay_grad.abs().min().item()
    grad_mean = decay_grad.abs().mean().item()

    print(f"  Decay logits grad norm: {grad_norm:.4f}")
    print(f"  Decay logits grad max:  {grad_max:.4f}")
    print(f"  Decay logits grad min:  {grad_min:.6f}")
    print(f"  Decay logits grad mean: {grad_mean:.4f}")

    # 梯度不应该消失（全零）或爆炸
    grad_healthy = grad_norm > 1e-6 and grad_norm < 1e6
    no_vanishing = grad_min > 1e-10

    # 也检查reset_logit的梯度
    reset_grad = eraser.reset_logit.grad
    reset_grad_val = reset_grad.item() if reset_grad is not None else 0
    print(f"  Reset logit grad: {reset_grad_val:.6f}")

    passed = grad_healthy and no_vanishing
    return report(passed, f"Grad norm: {grad_norm:.4f}, min: {grad_min:.2e}")


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 60)
    print("  BIIC Phase 1: Mathematical Verification Suite")
    print("  Cl(4,1) Clifford Algebra + Eraser Operator")
    print("=" * 60)

    tests = [
        test_1_multiplication_table,
        test_2_rotor_verification,
        test_3_grade0_invariance,
        test_4_grade5_invariance,
        test_5_grade1_equivariance,
        test_6_gradient_flow,
        test_7_multichannel_independence,
        test_8_eraser_convergence,
        test_9_immutable_core_preserved,
        test_10_decay_gradient_health,
    ]

    results = []
    for test_fn in tests:
        try:
            passed = test_fn()
        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            passed = False
        results.append(passed)

    # 汇总
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)

    for i, (passed, test_fn) in enumerate(zip(results, tests)):
        status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        name = test_fn.__name__.replace("test_", "").replace("_", " ").title()
        print(f"  Test {i+1:2d}: {status}  {name}")

    n_passed = sum(results)
    n_total = len(results)
    print(f"\n  Total: {n_passed}/{n_total} passed")

    if n_passed == n_total:
        print("\n  \033[92m*** ALL TESTS PASSED - Ready for Phase 2 ***\033[0m")
    else:
        print(f"\n  \033[91m*** {n_total - n_passed} TESTS FAILED - Fix before proceeding ***\033[0m")

    return n_passed == n_total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
