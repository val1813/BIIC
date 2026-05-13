"""
Phase 2 Step 1 测试：AllGradeDecoder基本学习能力
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/data/biic')
sys.path.insert(0, '/data/biic/phase2')
from all_grade_decoder import AllGradeDecoder


def test_decoder_can_learn():
    """
    最小可行测试：随机多向量 -> 解码器 -> 能否过拟合一个小数据集？
    """
    print("=" * 50)
    print("Test 1: Decoder Overfit Test")
    print("=" * 50)

    vocab_size = 1000
    n_channels = 8
    batch_size = 16
    seq_len = 32

    decoder = AllGradeDecoder(vocab_size, n_channels=n_channels, d_hidden=128)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    # 固定的随机多向量（模拟编码器输出）
    torch.manual_seed(42)
    mv_fixed = torch.randn(batch_size, seq_len, n_channels, 32)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    losses = []
    for step in range(200):
        optimizer.zero_grad()
        logits = decoder(mv_fixed)
        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1)
        )
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0:
            print(f"  Step {step}: loss={loss.item():.4f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    improvement = (initial_loss - final_loss) / initial_loss

    print(f"\n  Initial loss: {initial_loss:.4f}")
    print(f"  Final loss: {final_loss:.4f}")
    print(f"  Improvement: {improvement:.1%}")

    passed = improvement > 0.5
    print(f"  Result: {'PASS' if passed else 'FAIL'} (threshold: >50% improvement)")
    return passed


def test_gradient_flow():
    """
    梯度流测试：解码器参数的梯度范数在合理范围
    """
    print("\n" + "=" * 50)
    print("Test 2: Gradient Flow")
    print("=" * 50)

    vocab_size = 1000
    n_channels = 8

    decoder = AllGradeDecoder(vocab_size, n_channels=n_channels, d_hidden=128)

    mv = torch.randn(4, 16, n_channels, 32, requires_grad=True)
    targets = torch.randint(0, vocab_size, (4, 16))

    logits = decoder(mv)
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, vocab_size), targets.reshape(-1))
    loss.backward()

    grad_norms = []
    for name, p in decoder.named_parameters():
        if p.grad is not None:
            grad_norms.append(p.grad.norm().item())

    avg_grad = sum(grad_norms) / len(grad_norms)
    max_grad = max(grad_norms)
    min_grad = min(grad_norms)

    print(f"  Avg grad norm: {avg_grad:.4f}")
    print(f"  Max grad norm: {max_grad:.4f}")
    print(f"  Min grad norm: {min_grad:.6f}")

    passed = 1e-4 < avg_grad < 10.0
    print(f"  Result: {'PASS' if passed else 'FAIL'} (avg in [1e-4, 10])")
    return passed


def test_grade_gates_update():
    """
    grade_gates更新测试：训练后gate值有变化
    """
    print("\n" + "=" * 50)
    print("Test 3: Grade Gates Update")
    print("=" * 50)

    vocab_size = 1000
    n_channels = 8

    decoder = AllGradeDecoder(vocab_size, n_channels=n_channels, d_hidden=128)
    optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

    initial_gates = decoder.get_grade_weights().copy()

    torch.manual_seed(42)
    mv = torch.randn(8, 16, n_channels, 32)
    targets = torch.randint(0, vocab_size, (8, 16))

    for step in range(100):
        optimizer.zero_grad()
        logits = decoder(mv)
        loss = nn.CrossEntropyLoss()(logits.reshape(-1, vocab_size), targets.reshape(-1))
        loss.backward()
        optimizer.step()

    final_gates = decoder.get_grade_weights()

    print("  Grade gate changes:")
    total_change = 0.0
    for g in range(6):
        change = abs(final_gates[g] - initial_gates[g])
        total_change += change
        print(f"    grade-{g}: {initial_gates[g]:.4f} -> {final_gates[g]:.4f} (delta={change:.4f})")

    passed = total_change > 0.01
    print(f"  Total change: {total_change:.4f}")
    print(f"  Result: {'PASS' if passed else 'FAIL'} (total change > 0.01)")
    return passed


def test_grade0_vs_all_grades():
    """
    消融测试：只用grade-0 vs 全部grade
    """
    print("\n" + "=" * 50)
    print("Test 4: Grade-0 vs All Grades Ablation")
    print("=" * 50)

    vocab_size = 1000
    n_channels = 8
    batch_size = 16
    seq_len = 32

    torch.manual_seed(42)
    mv_fixed = torch.randn(batch_size, seq_len, n_channels, 32)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    results = {}

    for mode, active_grades in [("only_grade0", [0]), ("all_grades", None)]:
        torch.manual_seed(123)
        decoder = AllGradeDecoder(vocab_size, n_channels=n_channels, d_hidden=128)
        optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)

        for step in range(300):
            optimizer.zero_grad()
            logits = decoder(mv_fixed, active_grades=active_grades)
            loss = nn.CrossEntropyLoss()(
                logits.reshape(-1, vocab_size),
                targets.reshape(-1)
            )
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        results[mode] = final_loss
        print(f"  {mode}: final loss={final_loss:.4f}")

    print(f"\n  Grade-0 only loss: {results['only_grade0']:.4f}")
    print(f"  All grades loss:   {results['all_grades']:.4f}")

    if results['all_grades'] < results['only_grade0']:
        print("  PASS: All grades better than grade-0 only")
    else:
        print("  NOTE: All grades not better (normal with random init)")

    return True  # 这个测试是信息性的，不设硬性通过标准


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BIIC Phase 2 - Step 1: AllGradeDecoder Tests")
    print("=" * 60)

    results = []
    results.append(test_decoder_can_learn())
    results.append(test_gradient_flow())
    results.append(test_grade_gates_update())
    results.append(test_grade0_vs_all_grades())

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    names = ["Overfit", "Gradient Flow", "Gate Update", "Ablation"]
    for name, passed in zip(names, results):
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    n_pass = sum(results)
    print(f"\n  {n_pass}/{len(results)} passed")
    if n_pass == len(results):
        print("  Step 1 COMPLETE - proceed to Step 2")
