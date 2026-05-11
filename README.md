# BIIC: Bio-Inspired Information Cell

**基于几何代数的语言模型无损信息表示框架**

A geometric algebra framework for lossless information representation in language models.

---

## 问题 / The Problem

当前语言模型的token表示有一个根本缺陷：所有语义信息压在一个扁平向量里，推理过程中被逐层覆盖。残差连接只加不减，长推理中无关状态不断累积。

Current token representations compress all semantics into a single flat vector that gets overwritten during inference. Residual connections only add, never subtract — irrelevant states accumulate over long reasoning chains.

三个具体失败：
1. **信息损耗不可逆** — 深层网络无法还原原始token语义
2. **信息过载无清除** — 残差连接只做加法，中间状态无限累积
3. **长推理系统性退化** — 推理深度越大，性能越差

Three concrete failures:
1. **Irreversible information loss** — deep layers cannot recover original token semantics
2. **No cleanup mechanism** — residual connections accumulate noise indefinitely
3. **Systematic long-reasoning degradation** — performance drops with inference depth

---

## 思路：从DNA学习 / Approach: Learn from DNA

DNA解决了一个类似问题。细胞必须同时：
- 永久保存基因组（身份）
- 动态读写表观遗传标记（当前状态）
- 通过去甲基化酶主动擦除过时标记

DNA solves an analogous problem. A cell must simultaneously:
- Preserve its genome (identity) permanently
- Dynamically read/write epigenetic marks (current state)
- Actively erase outdated marks via demethylation enzymes

我们将这个三层架构映射到数学结构：

We map this three-layer architecture to mathematics:

| DNA层 / DNA Layer | BIIC组件 / Component | 数学实现 / Math |
|-----------|---------------|--------------------------|
| 基因组（不变） / Genome (immutable) | 不变核 / Immutable Core | Grade-0标量（代数不变量） |
| 表观基因组（读写擦） / Epigenome (R/W/E) | 可变态 / Mutable State | Grade-1~4等变分量 |
| 去甲基化酶 / Demethylase | 遗忘算子 / Eraser | 选择性衰减算子 |

关键洞察：**Clifford几何代数Cl(4,1)在同一代数结构中同时提供不变量和等变量**，不变性由定理保证，不是工程近似。

Key insight: **Clifford algebra Cl(4,1) provides both invariant and equivariant quantities in the same structure**, with invariance guaranteed by theorem, not engineering.

---

## 核心假设 / Core Hypothesis

> 基于Clifford代数的语言信息承载物，其中grade-0在所有推理变换下严格不变、grade-1~4携带独立等变语义，在信息保持和长上下文推理上优于扁平token embedding。

> A language information carrier based on Clifford algebra, where grade-0 is strictly invariant under all inference transformations and grade-1~4 carry independent equivariant semantics, outperforms flat token embeddings in information preservation and long-context reasoning.

---

## 最小实验计划 / Experiment Plan

| 阶段 / Phase | 目标 / Goal | 状态 / Status |
|-------|------|--------|
| Phase 1 | Cl(4,1)数学性质验证 / Math verification | ✅ 完成 (3 seeds, 10/10 pass) |
| Phase 2 | 编解码链路验证 / Pipeline validation | ✅ 完成 (3 seeds, all pass) |
| Phase 3 | 6组对照实验 (H1/H2/H3) / Comparative | 🔄 运行中 / Running |
| Phase 4 | MVP语言模型 / MVP model | 📋 计划中 (dry run passed) |

---

## 已达成的结果 / Results

### Phase 1：Grade-0不变性是真实的 / Grade-0 Invariance is Real

100次连续sandwich积变换后 / After 100 consecutive transformations:

| 指标 / Metric | 值 / Value (mean ± std, 3 seeds) |
|--------|----------------------------|
| Grade-0不变性误差 | 6.56×10⁻⁶ ± 4.95×10⁻⁶ |
| Grade-5不变性误差 | 5.12×10⁻⁶ ± 3.81×10⁻⁶ |
| Grade-1等变性误差 | 1.15×10⁻⁶ ± 8.2×10⁻⁷ |
| 多通道泄漏 / Cross-channel leakage | 0.0（精确） |
| Eraser后grade-0变化 | 0.0（精确） |

Grade-0保持到机器精度不变。这是数学保证，不是工程近似。

Grade-0 stays invariant to machine precision. Mathematical guarantee, not engineering approximation.

### Phase 2：等变分量携带独立语义 / Equivariant Grades Carry Independent Semantics

| 指标 / Metric | 值 / Value |
|--------|-------|
| 全grade解码 vs 仅grade-0 / All-grade vs grade-0 only | 0.006 vs 0.032 loss (5.3×更好) |
| Grade-0 token区分度 / Token discrimination | 余弦相似度 0.029 ± 0.013 |
| Grade范数分离度 / Norm separation (std) | 9.23 ± 0.08 |
| 6层推理后grade-0变化 / After 6 layers | 0.0（精确） |
| 解码器过拟合改善 / Decoder overfit | 99.83% ± 0.001% |

等变分量（grade-1~4）携带不变核无法提供的语义信息。不同token的grade-0接近正交（cos_sim ≈ 0.03）。

Equivariant components carry semantic information beyond what the invariant core provides. Different tokens achieve near-orthogonal grade-0 representations.

### Phase 4 Dry Run：架构验证通过 / Architecture Validated

| 检查项 / Check | 结果 / Result |
|-------|--------|
| 前向传播 / Forward pass | ✅ |
| 训练中grade-0保持 / Grade-0 preserved | ✅ (change = 0.0) |
| 梯度传播 / Gradient flows | ✅ |
| DualCodebook双路贡献 / A/B both contribute | ✅ |
| 训练收敛 / Convergence (100 steps) | ✅ (34.6%) |
| Checkpoint保存加载 / Save/load | ✅ |

正式规模：37M参数，峰值4GB显存，~0.9s/step（单卡RTX 4090）。

Full-scale: 37M params, 4GB peak VRAM, ~0.9s/step on single RTX 4090.

---

## 后续计划 / What Comes Next

**Phase 3**（运行中 / running）：6组对照实验验证三个假设：
- H1：几何结构本身有价值，还是只来自正交约束？
- H2：等变分量有独立贡献，还是只来自维度更高？
- H3：Eraser在长序列上是否真正控制信息熵？

**Phase 4**（Phase 3后 / after Phase 3）：训练最小语言模型：
- DualCodebookDecoder（不变核/等变分量双路解码）
- SlowFast架构（慢速网络每K步，快速网络每步）
- 无残差连接（稳定性由代数不变性保证）

---

## 如果成功 / If This Works

如果Phase 3+4验证假设成立：

1. **无损长上下文** — grade-0无论推理多深都保持原始语义
2. **不需要KV Cache** — 可变态替代键值存储
3. **内建可解释性** — grade分解揭示模型"记住了什么"vs"在想什么"
4. **O(L)复杂度** — 慢快分离消除二次方注意力开销
5. **天然多模态对齐** — 不同模态共享同一代数空间，grade-0直接可比

If Phase 3+4 confirm the hypotheses:

1. **Lossless long-context** — grade-0 preserves semantics regardless of depth
2. **No KV cache** — mutable state replaces key-value storage
3. **Built-in interpretability** — grade decomposition reveals memory vs reasoning
4. **O(L) complexity** — slow-fast separation eliminates quadratic attention
5. **Natural multimodal alignment** — modalities share algebraic space

---

## 仓库结构 / Repository Structure

```
BIIC/
├── README.md
├── src/
│   ├── clifford_cl41.py          # Cl(4,1)几何代数实现
│   ├── rotor_utils.py            # 旋转子生成、sandwich积、稳定化
│   ├── eraser_ops.py             # GradeAwareEraser（选择性遗忘）
│   ├── token_to_ic.py            # Token → 不变核编码器
│   ├── all_grade_decoder.py      # AllGradeDecoder（门控多grade解码）
│   ├── mutable_state.py          # BIICLayer（Writer + Eraser）
│   └── biic_loss.py              # 分阶段退火辅助损失
├── tests/
│   ├── test_phase1.py            # 10项数学验证测试
│   ├── test_decoder_basic.py     # 解码器过拟合+梯度+消融
│   ├── test_encoder.py           # Grade分离+token区分
│   └── test_full_pipeline.py     # 端到端链路+Eraser有效性
├── results/
│   ├── phase1/phase1_results.json   # Phase 1完整数据（3 seeds × 10 tests）
│   └── phase2/phase2_results.json   # Phase 2完整数据（3 seeds × 5 tests）
├── figures/
│   ├── fig1_grade0_invariance.png
│   ├── fig2_decoder_loss_curve.png
│   ├── fig3_grade_norm_distribution.png
│   └── fig4_token_discrimination.png
├── requirements.txt
└── LICENSE
```

---

## 快速开始 / Quick Start

```bash
pip install torch numpy scipy matplotlib

# Phase 1验证（CPU，约2分钟）
cd tests
python test_phase1.py

# Phase 2链路测试（CPU，约10分钟）
python test_decoder_basic.py
python test_encoder.py
python test_full_pipeline.py
```

---

## 环境要求 / Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, SciPy, Matplotlib

Phase 1+2验证不需要GPU。Phase 3+4使用单卡RTX 4090。

No GPU required for Phase 1+2. Phase 3+4 use a single RTX 4090.

---

## 参考文献 / References

- Hestenes & Sobczyk, 1984. *Clifford Algebra to Geometric Calculus*
- Brehmer et al., 2023. Geometric Algebra Transformer (GATr). [arXiv:2305.18415](https://arxiv.org/abs/2305.18415)
- Li et al., 2025. Versor: A Geometric Sequence Architecture. [arXiv:2602.10195](https://arxiv.org/abs/2602.10195)
- 2025\. Toward a Functional Geometric Algebra for Natural Language Semantics. [arXiv:2604.25902](https://arxiv.org/abs/2604.25902)
- 2025\. All You Need is Geometric Algebra (CliffordNet). [arXiv:2601.06793](https://arxiv.org/abs/2601.06793)
- Wu & Zhang, 2017. TET-mediated active DNA demethylation. *Nature Reviews Genetics*
- Zou et al., 2023. Representation Engineering. [arXiv:2310.01405](https://arxiv.org/abs/2310.01405)

---

## License

MIT

## Citation

```bibtex
@misc{huang2025biic,
  title={Bio-Inspired Information Cell: A Geometric Algebra Framework for Lossless Information Representation in Language Models},
  author={Huang, Zhongchang},
  year={2025},
  note={Phase 1-2 complete, Phase 3-4 ongoing.}
}
```
