# BIIC: Bio-Inspired Information Cell

**→ [View full project page:val1813.github.io/BIIC/README](https://val1813.github.io/BIIC/README.html)**点击这个网站详看

A geometric algebra framework for lossless information representation in language models.




**基于几何代数的语言模型无损信息表示框架**



[![License: BSL-1.1](https://img.shields.io/badge/License-BSL--1.1-blue.svg)](LICENSE)
[![Phase 1](https://img.shields.io/badge/Phase%201-Complete-brightgreen)]()
[![Phase 2](https://img.shields.io/badge/Phase%202-Complete-brightgreen)]()
[![Phase 3](https://img.shields.io/badge/Phase%203-Running-yellow)]()

---

## 问题 / The Problem

当前语言模型的token表示有一个根本缺陷：所有语义信息压在一个扁平向量里，推理过程中被逐层覆盖。

Current token representations compress all semantics into a single flat vector that gets overwritten layer by layer during inference.

<p align="center">
<img src="figures/fig1_grade0_invariance.png" width="600">
<br><em>Fig 1. Grade-0 invariance after 100 consecutive transformations — error stays at 10⁻⁶ level (3 seeds)</em>
</p>

| 失败模式 / Failure Mode | 原因 / Cause |
|:---:|:---:|
| 信息损耗不可逆 / Irreversible loss | 深层覆盖原始语义 / Deep layers overwrite original semantics |
| 信息过载 / Information overload | 残差只加不减 / Residual only adds, never subtracts |
| 长推理退化 / Long-reasoning degradation | 无关状态无限累积 / Irrelevant states accumulate |

---

## 思路：从DNA学习 / Approach: Learn from DNA

DNA同时做到了三件事：永久保存基因组、动态读写表观标记、主动擦除过时标记。

DNA achieves three things simultaneously: permanent genome preservation, dynamic epigenetic read/write, and active erasure of outdated marks.

```
┌─────────────────────────────────────────────────────────┐
│  DNA Architecture          →    BIIC Architecture       │
├─────────────────────────────────────────────────────────┤
│  Genome (immutable)        →    Grade-0 (invariant)     │
│  Epigenome (read/write)    →    Grade-1~4 (equivariant) │
│  TET demethylase (erase)   →    GradeAwareEraser        │
└─────────────────────────────────────────────────────────┘
```

关键洞察：**Clifford几何代数Cl(4,1)在同一结构中同时提供不变量和等变量**，不变性由定理保证。

Key insight: **Clifford algebra Cl(4,1) provides both invariant and equivariant quantities in one structure** — invariance guaranteed by theorem.

---

## 实验结果 / Results

### Phase 1: 数学验证 / Mathematical Verification ✅

<p align="center">
<img src="figures/fig1_grade0_invariance.png" width="550">
</p>

| 指标 / Metric | 值 / Value (3 seeds) |
|:---|:---|
| Grade-0 invariance error (100 transforms) | 6.56×10⁻⁶ ± 4.95×10⁻⁶ |
| Grade-5 invariance error | 5.12×10⁻⁶ ± 3.81×10⁻⁶ |
| Multi-channel leakage | **0.0 (exact)** |
| Eraser preserves grade-0 | **0.0 (exact)** |

### Phase 2: 编解码链路 / Encoding-Decoding Pipeline ✅

<p align="center">
<img src="figures/fig2_decoder_loss_curve.png" width="550">
</p>

<p align="center">
<img src="figures/fig3_grade_norm_distribution.png" width="550">
<br><em>Fig 3. Grade separation emerges naturally — different grades learn different roles</em>
</p>

| 指标 / Metric | 值 / Value |
|:---|:---|
| All-grade vs grade-0 only decoding | **5.3× better** (0.006 vs 0.032) |
| Token discrimination (cosine sim) | 0.029 ± 0.013 (near-orthogonal) |
| Grade-0 after 6 inference layers | **0.0 change (exact)** |

<p align="center">
<img src="figures/fig4_token_discrimination.png" width="550">
<br><em>Fig 4. Different tokens achieve near-orthogonal grade-0 representations</em>
</p>

### Phase 4 Dry Run: 架构验证 / Architecture Validated ✅

Full-scale config validated: **37M params, 4GB VRAM, 0.9s/step** on single RTX 4090.

All 6 checks passed: forward, grade-0 preserved, gradients flow, DualCodebook works, training converges, checkpoint OK.

---

## 实验计划 / Experiment Plan

| Phase | 目标 / Goal | 状态 / Status |
|:---:|:---|:---:|
| 1 | Cl(4,1) 数学性质验证 | ✅ Complete |
| 2 | 编解码链路验证 | ✅ Complete |
| 3 | 6组对照 (H1/H2/H3假设检验) | 🔄 Running |
| 4 | MVP语言模型 (SlowFast + DualCodebook) | 📋 Planned |

**Phase 3 正在验证三个假设 / Testing three hypotheses:**
- H1: 几何结构本身有价值？还是只来自正交约束？
- H2: 等变分量有独立贡献？还是只来自维度更高？
- H3: Eraser在长序列上是否真正控制信息熵？

---

## 如果成功 / If This Works

| 能力 / Capability | 机制 / Mechanism |
|:---|:---|
| 无损长上下文 / Lossless long-context | Grade-0无论推理多深都保持原始语义 |
| 不需要KV Cache / No KV cache | 可变态替代键值存储 |
| 内建可解释性 / Built-in interpretability | Grade分解揭示"记住了什么" vs "在想什么" |
| O(L)复杂度 / Linear complexity | 慢快分离消除二次方注意力 |
| 天然多模态对齐 / Natural multimodal | 不同模态共享代数空间 |

---

## 仓库结构 / Structure

```
BIIC/
├── src/                          # 核心实现
│   ├── clifford_cl41.py          # Cl(4,1) 几何代数
│   ├── rotor_utils.py            # 旋转子 & sandwich积
│   ├── eraser_ops.py             # GradeAwareEraser
│   ├── token_to_ic.py            # 编码器
│   ├── all_grade_decoder.py      # 全grade门控解码器
│   ├── mutable_state.py          # BIICLayer
│   └── biic_loss.py              # 分阶段辅助损失
├── tests/                        # 验证测试
├── results/                      # 实验数据 (JSON, 3 seeds)
├── figures/                      # 论文图表
├── requirements.txt
└── LICENSE
```

## 快速开始 / Quick Start

```bash
pip install torch numpy scipy matplotlib

# Phase 1 验证 (CPU, ~2min)
python tests/test_phase1.py

# Phase 2 验证 (CPU, ~10min)
python tests/test_decoder_basic.py
python tests/test_encoder.py
python tests/test_full_pipeline.py
```

---

## 参考文献 / References

- Brehmer et al., 2023. [Geometric Algebra Transformer (GATr)](https://arxiv.org/abs/2305.18415). NeurIPS 2023.
- Li et al., 2025. [Versor: A Geometric Sequence Architecture](https://arxiv.org/abs/2602.10195).
- 2025\. [Toward a Functional Geometric Algebra for NLP](https://arxiv.org/abs/2604.25902).
- 2025\. [All You Need is Geometric Algebra (CliffordNet)](https://arxiv.org/abs/2601.06793).
- Wu & Zhang, 2017. TET-mediated active DNA demethylation. *Nature Reviews Genetics*.
- Zou et al., 2023. [Representation Engineering](https://arxiv.org/abs/2310.01405).

---

## Contact

对这个方向感兴趣、愿意一起写论文或探索新范式的朋友，欢迎联系：

Interested in collaborating on the paper or exploring new paradigms together? Reach out:

**WeChat: llmbbs**

---

## Citation

```bibtex
@misc{huang2025biic,
  title={Bio-Inspired Information Cell: A Geometric Algebra Framework for
         Lossless Information Representation in Language Models},
  author={Huang, Zhongchang},
  year={2025},
  note={Phase 1-2 complete, Phase 3-4 ongoing.}
}
```

## License

[Business Source License 1.1](LICENSE) — Free for non-production use. See LICENSE for details.