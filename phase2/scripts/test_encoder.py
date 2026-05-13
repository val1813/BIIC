"""
Phase 2 Step 2 测试：TokenToImmutableCore编码器 + grade分离验证
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/data/biic')
sys.path.insert(0, '/data/biic/phase2')
from token_to_ic import TokenToImmutableCore
from all_grade_decoder import AllGradeDecoder
from biic_loss import BIICLoss


def test_grade_separation():
    """
    测试grade分工是否在训练中自然产生
    """
    print("=" * 50)
    print("Test 1: Grade Separation Training")
    print("=" * 50)

    vocab_size = 1000
    n_channels = 8
    batch_size = 8
    seq_len = 16
    total_steps = 500

    encoder = TokenToImmutableCore(vocab_size, n_channels)
    decoder = AllGradeDecoder(vocab_size, n_channels, d_hidden=128)
    loss_fn = BIICLoss(vocab_size, n_channels)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(loss_fn.parameters()),
        lr=1e-3
    )

    torch.manual_seed(42)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    print("  Training...")
    losses = []
    for step in range(total_steps):
        optimizer.zero_grad()

        mv = encoder(token_ids)
        logits = decoder(mv)

        loss, loss_dict = loss_fn(mv, logits, targets, step, total_steps)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()),
            max_norm=1.0
        )
        optimizer.step()
        losses.append(loss.item())

        if step % 100 == 0:
            grade_norms = encoder.get_grade_norms(mv.detach())
            print(f"  Step {step}: loss={loss.item():.4f}, alpha={loss_dict['alpha']:.3f}")
            print(f"    Grade norms: " +
                  ", ".join([f"g{g}={n:.3f}" for g, n in grade_norms.items()]))

    # 验证loss下降
    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_decreased = final_loss < initial_loss
    print(f"\n  Loss: {initial_loss:.4f} -> {final_loss:.4f}")
    print(f"  Loss decreased: {loss_decreased}")

    # 验证grade范数有差异
    with torch.no_grad():
        mv = encoder(token_ids)
        grade_norms = encoder.get_grade_norms(mv)
        norms_list = list(grade_norms.values())
        norm_std = torch.tensor(norms_list).std().item()
        print(f"  Grade norm std: {norm_std:.4f} (should be > 0)")

    passed = loss_decreased and norm_std > 0.001
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_token_discrimination():
    """
    测试不同token的grade-0是否有区分能力
    """
    print("\n" + "=" * 50)
    print("Test 2: Token Discrimination via Grade-0")
    print("=" * 50)

    vocab_size = 1000
    n_channels = 8
    batch_size = 8
    seq_len = 16
    total_steps = 300

    encoder = TokenToImmutableCore(vocab_size, n_channels)
    decoder = AllGradeDecoder(vocab_size, n_channels, d_hidden=128)
    loss_fn = BIICLoss(vocab_size, n_channels)

    optimizer = torch.optim.Adam(
        list(encoder.parameters()) +
        list(decoder.parameters()) +
        list(loss_fn.parameters()),
        lr=1e-3
    )

    torch.manual_seed(42)
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = token_ids.clone()  # 自编码任务：预测自己

    for step in range(total_steps):
        optimizer.zero_grad()
        mv = encoder(token_ids)
        logits = decoder(mv)
        loss, _ = loss_fn(mv, logits, targets, step, total_steps)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(decoder.parameters()), 1.0)
        optimizer.step()

    # 检查不同token的grade-0余弦相似度
    with torch.no_grad():
        mv = encoder(token_ids)
        from clifford_cl41 import GRADE_SLICES
        g0_start, g0_end = GRADE_SLICES[0]
        grade0 = mv[..., g0_start:g0_end].squeeze(-1)  # [B, L, C]

        # 计算同一batch内不同位置的余弦相似度
        g0_flat = grade0.reshape(-1, n_channels)  # [B*L, C]
        g0_norm = g0_flat / (g0_flat.norm(dim=-1, keepdim=True) + 1e-8)
        sim_matrix = g0_norm @ g0_norm.T  # [B*L, B*L]

        # 排除对角线
        mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool)
        avg_sim = sim_matrix[mask].mean().item()

    print(f"  Avg cosine similarity between different tokens' grade-0: {avg_sim:.4f}")
    print(f"  (closer to 0 = better discrimination, closer to 1 = no discrimination)")

    passed = avg_sim < 0.9
    print(f"  Result: {'PASS' if passed else 'FAIL'} (threshold: < 0.9)")
    return passed


def test_encoder_gradient_health():
    """
    验证编码器梯度健康
    """
    print("\n" + "=" * 50)
    print("Test 3: Encoder Gradient Health")
    print("=" * 50)

    vocab_size = 1000
    n_channels = 8

    encoder = TokenToImmutableCore(vocab_size, n_channels)
    decoder = AllGradeDecoder(vocab_size, n_channels, d_hidden=128)

    token_ids = torch.randint(0, vocab_size, (4, 16))
    targets = torch.randint(0, vocab_size, (4, 16))

    mv = encoder(token_ids)
    logits = decoder(mv)
    loss = nn.CrossEntropyLoss()(logits.reshape(-1, vocab_size), targets.reshape(-1))
    loss.backward()

    # 检查各grade投影层的梯度
    grade_layers = [
        ('grade0', encoder.to_grade0),
        ('grade1', encoder.to_grade1),
        ('grade2', encoder.to_grade2),
        ('grade3', encoder.to_grade3),
        ('grade4', encoder.to_grade4),
        ('grade5', encoder.to_grade5),
    ]

    all_healthy = True
    for name, layer in grade_layers:
        grad_norm = layer.weight.grad.norm().item() if layer.weight.grad is not None else 0
        healthy = 1e-6 < grad_norm < 100
        status = "OK" if healthy else "BAD"
        print(f"  {name} grad norm: {grad_norm:.6f} [{status}]")
        if not healthy:
            all_healthy = False

    print(f"  Result: {'PASS' if all_healthy else 'FAIL'}")
    return all_healthy


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BIIC Phase 2 - Step 2: Encoder Tests")
    print("=" * 60)

    results = []
    results.append(test_grade_separation())
    results.append(test_token_discrimination())
    results.append(test_encoder_gradient_health())

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    names = ["Grade Separation", "Token Discrimination", "Gradient Health"]
    for name, passed in zip(names, results):
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    n_pass = sum(results)
    print(f"\n  {n_pass}/{len(results)} passed")
    if n_pass == len(results):
        print("  Step 2 COMPLETE - proceed to Step 3")
