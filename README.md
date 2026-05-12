# BIIC: Bio-Inspired Information Cell

**→ [View full project page](https://val1813.github.io/BIIC/README.html)**

**基于几何代数的语言模型无损信息表示框架**

A geometric algebra framework for lossless information representation in language models.

[![License: BSL-1.1](https://img.shields.io/badge/License-BSL--1.1-blue.svg)](LICENSE)
[![Phase 1](https://img.shields.io/badge/Phase%201-Complete-brightgreen)]()
[![Phase 2](https://img.shields.io/badge/Phase%202-Complete-brightgreen)]()
[![Phase 3](https://img.shields.io/badge/Phase%203-Complete-brightgreen)]()
[![Phase 4](https://img.shields.io/badge/Phase%204-Complete-brightgreen)]()
[![Phase 5](https://img.shields.io/badge/Phase%205-Complete-brightgreen)]()
[![Phase 6](https://img.shields.io/badge/Phase%206-Planning-blue)]()

---

## 问题 / The Problem

当前语言模型的token表示有一个根本缺陷：所有语义信息压在一个扁平向量里，推理过程中被逐层覆盖。

Current token representations compress all semantics into a single flat vector that gets overwritten layer by layer during inference.

| 失败模式 / Failure Mode | 原因 / Cause |
|:---:|:---:|
| 信息损耗不可逆 / Irreversible loss | 深层覆盖原始语义 / Deep layers overwrite original semantics |
| 信息过载 / Information overload | 残差只加不减 / Residual only adds, never subtracts |
| 长推理退化 / Long-reasoning degradation | 无关状态无限累积 / Irrelevant states accumulate |

---

## 思路：从DNA学习 / Approach: Learn from DNA

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

## 实验设计与结果 / Experiments & Results

### Phase 1: 数学验证 / Mathematical Verification ✅

**验证什么 / What we test:** Cl(4,1)的grade-0在sandwich积变换下是否真的严格不变？多通道之间是否有信息泄漏？Eraser是否能保证不变核安全？

**Why:** 如果数学性质在工程实现中不成立，后续所有设计都没有基础。

<p align="center">
<img src="figures/fig1_grade0_invariance.png" width="550">
</p>

| 指标 / Metric | 值 / Value (3 seeds) |
|:---|:---|
| Grade-0 invariance error (100 transforms) | 6.56×10⁻⁶ ± 4.95×10⁻⁶ |
| Grade-5 invariance error | 5.12×10⁻⁶ ± 3.81×10⁻⁶ |
| Multi-channel leakage | **0.0 (exact)** |
| Eraser preserves grade-0 | **0.0 (exact)** |

**结论 / Conclusion:** Grade-0不变性在float32精度下100次变换后误差仅10⁻⁶级别。多通道间零泄漏。Eraser操作后grade-0变化精确为零。数学保证在工程中成立。

---

### Phase 2: 编解码链路 / Encoding-Decoding Pipeline ✅

**验证什么 / What we test:** 等变分量（grade-1~4）是否携带独立于不变核的语义信息？编码器能否自然产生grade分工？

**Why:** 如果等变分量只是冗余，BIIC就退化为一个普通的不变embedding，没有新贡献。

<p align="center">
<img src="figures/fig2_decoder_loss_curve.png" width="550">
</p>

<p align="center">
<img src="figures/fig3_grade_norm_distribution.png" width="550">
<br><em>Grade separation emerges naturally — different grades learn different roles</em>
</p>

| 指标 / Metric | 值 / Value |
|:---|:---|
| All-grade vs grade-0 only decoding | **5.3× better** (0.006 vs 0.032) |
| Token discrimination (cosine sim) | 0.029 ± 0.013 (near-orthogonal) |
| Grade-0 after 6 inference layers | **0.0 change (exact)** |

<p align="center">
<img src="figures/fig4_token_discrimination.png" width="550">
<br><em>Different tokens achieve near-orthogonal grade-0 representations</em>
</p>

**结论 / Conclusion:** 等变分量携带不变核无法提供的独立语义信息（5.3×改善）。不同token的grade-0接近正交（cos_sim=0.03），区分能力强。Grade分工在训练中自然产生，无需手动设计。

---

### Phase 3: 假设检验对照实验 / Hypothesis Testing ✅

**验证什么 / What we test:** 三个核心假设——
- H1: BIIC的优势来自几何结构本身，还是只来自正交约束？
- H2: 等变分量有独立贡献，还是只来自维度更高？
- H3: Eraser在短序列上是否有效？

**Why:** 排除混淆变量，确认BIIC的优势来源。

| Group | Description | Final Loss (mean ± std, 3 seeds) |
|:---|:---|:---|
| A1 | BIIC Full (Eraser=0.5) | **10.8285 ± 0.0008** |
| A2 | BIIC + Weak Eraser (0.01) | 10.8289 ± 0.0010 |
| B | Orthogonal Token + tanh (H1 baseline) | 10.8319 ± 0.0020 |
| C | Linear + LayerNorm (lower bound) | 10.8292 ± 0.0007 |
| D | BIIC grade-0 only (H2 ablation) | 10.8271 ± 0.0037 |
| E | 2048-dim Embedding (H2 dim-matched) | 10.9984 ± 0.0116 |

**结论 / Conclusions:**
- **H1 confirmed:** A1 (10.8285) < B (10.8319) — 几何结构优于纯正交约束
- **H2 confirmed:** A1 (10.8285) << E (10.9984) — 等变结构远优于纯高维度
- **H2b (unexpected):** D (10.8271) ≈ A1 — grade-0 alone is surprisingly strong at seq_len=64
- **H3 not confirmed:** A1 ≈ A2 — Eraser效果在短序列(64 tokens)下不明显，需要长序列验证

---

### Phase 4: 语言模型训练 / Language Model Training ✅

**验证什么 / What we test:** BIIC多向量能否作为语言模型的信息承载物，在真实文本上学习next-token prediction？

**Why:** Phase 1-3验证了数学性质和组件，Phase 4验证整个系统能否端到端工作。

| Metric | BIIC v0.2 | Transformer Baseline |
|:---|:---|:---|
| Params | 73M | 52M |
| Data | WikiText-103 (117M tokens) | WikiText-103 (117M tokens) |
| Steps | 4,600 (stopped) | 50,000 (converged) |
| Final PPL | 390 | **53.9** |
| Peak VRAM | 1,584 MB | 2,558 MB |
| Speed | 0.10 step/s | 36 step/s |

**结论 / Conclusion:** BIIC能学语言（PPL从58895降到390），但远未收敛。Transformer在同等训练下PPL=53.9。BIIC的sandwich积导致训练速度慢360×，这是工程瓶颈而非架构缺陷。当前BIIC的计算效率不足以在合理时间内收敛到可比PPL。

**Grade-0 Only消融（进行中）：** 冻结等变分量，只用grade-0做LM。如果PPL≈完整BIIC，说明等变分量在LM任务中没有贡献。

---

### 显存对比 / Memory Scaling ✅

**验证什么 / What we test:** BIIC（无KV Cache）vs Transformer（有KV Cache）在不同序列长度下的显存增长率。

**Why:** 如果BIIC的显存增长更慢，说明"无KV Cache"的架构优势在长序列时成立。

**v1 (batch=4, 24GB GPU, n_channels=8):**

| seq_len | BIIC (MB) | Transformer (MB) | Winner |
|:---:|:---:|:---:|:---:|
| 256 | 747 | 431 | Transformer |
| 512 | 972 | 640 | Transformer |
| 1024 | 1425 | 1060 | Transformer |
| 2048 | 2327 | **2622** | **BIIC** |

**v2 (batch=1, 48GB GPU, n_channels=64, 正式规模):**

| seq_len | BIIC (MB) | Transformer (MB) | Ratio |
|:---:|:---:|:---:|:---:|
| 256 | 526 | 664 | **0.79** |
| 512 | 870 | 854 | 1.02 |
| 1024 | 1548 | 1388 | 1.12 |
| 2048 | 2907 | 2468 | 1.18 |
| 4096 | 5624 | 4658 | 1.21 |
| 8192 | 11049 | 9128 | 1.21 |

**结论 / Conclusion:**
- v1 (小规模, batch=4): BIIC在seq>1800后更省显存，增长率3.1× vs 6.1×
- v2 (正式规模, batch=1): BIIC在seq=256时更省（0.79×），但长序列下比Transformer多21%。原因：当前BIICLayer的sandwich积是逐通道串行计算（64通道×32维blade），常数开销大；Transformer受益于PyTorch高度优化的attention kernel
- **优化方向：** 批量化sandwich积（向量化通道维度）可显著降低BIIC的显存开销

---

### Phase 5: 等变分量激活 / Equivariant Activation ✅

**验证什么 / What we test:** 能否通过跨token机制（Cohesin、相对不变注意力、分段Eraser）让等变分量在语言模型中自发激活？

**Why:** Phase 2证明等变分量携带独立信息，Phase 3 Probing证明grade-2自发编码句法（POS=0.789, DEP=0.823）。但在LM训练中，等变分量是否被模型主动利用？

**实验5.4: Cohesin + WikiText-103**

| 架构 | Final PPL | Gate值 | 结论 |
|:---:|:---:|:---:|:---|
| Original | 762.5 | — | baseline |
| Cohesin (gate init=0) | 747.1 | 0.509 | PPL低2.8%，但gate从0.5开始 |
| Cohesin (gate init=-4) | 762.6 | 0.025 | **gate不升高，模型选择不用** |

当gate从接近0开始时（sigmoid(-4)≈0.018），模型选择不打开gate。之前的"2.8%改善"是gate=0.5初始值的假象。

**实验5.5: 相对不变注意力 (Relative Invariant Attention)**

用grade-2的几何积作为额外的注意力分数：`score = score_sem + alpha * score_rel`

| 指标 | 初始值 | 最终值 | 结论 |
|:---:|:---:|:---:|:---|
| alpha | 0.018 | **0.0155** | 下降了，模型主动关闭 |
| gate | 0.018 | 0.025 | 几乎不动 |
| PPL | — | 751.2 | 与baseline无差异 |

**实验5.6: 分段Eraser (Segmented Eraser)**

每4层才擦除一次，grade-2永不擦除。监控grade-2范数是否在block内增长。

| Layer | g2_norm (step 0) | g2_norm (step 1900) | 变化 |
|:---:|:---:|:---:|:---:|
| 1-6 | ~2.03 | ~2.01 | **恒定，无增长模式** |

**Phase 5 总结论 / Phase 5 Conclusion:**

在WikiText-103 next-token prediction任务上，**等变分量无法被任何机制自发激活**。三种不同的激活策略（Cohesin跨token注意力、相对不变注意力、分段Eraser）全部失败。模型在有选择时，一致选择不使用等变分量。

这不意味着等变分量没有价值——Probing实验已证明grade-2确实编码了句法信息。问题在于：**next-token prediction任务本身不需要显式的关系表示**（Transformer也不需要显式关系表示就能做好LM）。

---

### Phase 5 → Phase 6: 方向转变 / Pivot

Phase 5的实验结果迫使我们重新审视架构方向：

**原路线（BIIC-v1）：** 在Transformer框架里塞BIIC → 本质是给Transformer换了个更复杂的embedding → 等变分量没有接口发挥作用 → 计算成本高但收益不明显

**新路线（BIIC-v2）：** 不再试图让等变分量在LM里"自发激活"，而是利用grade-0的数学保证作为稀疏架构的锚点

核心转变：从"让每个token更重"到"让信息按需稀疏激活"

```
BIIC-v1 (abandoned):
  每个token做全量Cl(4,1)运算
  等变分量被动等待激活
  计算成本随层数线性增长

BIIC-v2 (next):
  grade-0作为全局可寻址的身份锚点
  等变分量按需稀疏激活（MoE风格）
  grade-0相似度决定路由（不是内容路由）
```

---

### Phase 6: BIIC-v2 稀疏几何架构 / Sparse Geometric Architecture 📋

**核心思想：** grade-0不变核作为MoE路由的语义基础。这是DeepSeek V3和Mamba做不到的——它们的专家之间没有共享的不变锚点。

**计划实验：**

| 编号 | 实验 | 说明 |
|:---|:---|:---|
| 6.1 | Grade-0锚定路由 | 用grade-0相似度决定哪些token对需要交互 |
| 6.2 | 稀疏等变激活 | 只有高相似度的token对才激活grade-2计算 |
| 6.3 | 皮质柱并行 | token分组，组内浅层处理，组间grade-0投票 |

**BIIC-v2的独特价值主张：**
- MoE的路由有了语义基础（按grade-0路由，不是按内容路由）
- 皮质柱的投票有了数学基础（grade-0之间的几何度量）
- 稀疏激活有了信息论基础（grade-0保证身份不丢失）

---

## 总结：什么被证明了 / What Has Been Proven

| 声称 / Claim | 证据 / Evidence | 状态 |
|:---|:---|:---:|
| Grade-0在推理中严格不变 | Phase 1: 误差10⁻⁶, 精确零泄漏 | ✅ |
| 等变分量携带独立语义 | Phase 2: 5.3×解码改善 | ✅ |
| 几何结构优于正交约束 | Phase 3: A1 < B | ✅ |
| 等变结构优于纯高维度 | Phase 3: A1 << E (14.6σ) | ✅ |
| BIIC能学语言 | Phase 4: PPL 58895→390 | ✅ |
| Grade-2自发编码句法 | Probing: POS=0.789, DEP=0.823 | ✅ |
| 等变分量在LM中自发激活 | Phase 5: 三种机制全部失败 | ❌ |
| Cohesin激活等变分量 | Phase 5: gate从-4不升高 | ❌ |
| 相对不变注意力激活等变分量 | Phase 5: alpha下降0.018→0.015 | ❌ |
| 分段Eraser激活grade-2 | Phase 5: g2_norms恒定 | ❌ |
| 正式规模显存优势 | Memory v2: 长序列BIIC多21% | ❌ (需CUDA优化) |
| Eraser控制信息熵 | Phase 3: A1≈A2 at seq=64 | ⚠️ 需长序列验证 |

**关键洞察 / Key Insight:**

等变分量有信息（Probing证明），有结构（H2检验证明），但在next-token prediction任务中不会被模型主动利用。这不是BIIC的失败，而是揭示了一个更深的问题：**LM任务本身不需要显式的关系表示**。Transformer也不需要显式关系表示就能做好LM——它通过注意力权重隐式编码关系。

这个发现推动了BIIC-v2的方向转变：不再试图在LM里激活等变分量，而是利用grade-0的数学保证作为稀疏架构的锚点。

---

## 仓库结构 / Structure

```
BIIC/
├── src/                              # 核心实现
│   ├── clifford_cl41.py              # Cl(4,1) 几何代数
│   ├── rotor_utils.py                # 旋转子 & sandwich积
│   ├── eraser_ops.py                 # GradeAwareEraser
│   ├── token_to_ic.py                # 编码器
│   ├── all_grade_decoder.py          # 全grade门控解码器
│   ├── mutable_state.py              # BIICLayer
│   ├── biic_loss.py                  # 分阶段辅助损失
│   ├── train_biic_lm.py             # BIIC LM v0.1 训练
│   ├── train_biic_lm_v02.py         # BIIC LM v0.2 训练 (WikiText)
│   ├── run_phase3.py                 # Phase 3 对照实验
│   └── memory_scaling_experiment.py  # 显存对比实验
├── tests/                            # 验证测试 (可复现)
├── results/                          # 实验数据 (JSON, 3 seeds)
│   ├── phase1/, phase2/              # Phase 1+2 完整数据
│   ├── phase3_A1~E.json              # Phase 3 六组对照
│   ├── memory_scaling.json           # 显存对比 v1 (seq 256-2048)
│   ├── memory_scaling_extended.json  # 显存对比 v2 (seq 256-8192)
│   ├── transformer_baseline_50k.json # Transformer 50k步 PPL=53.9
│   ├── cohesin_v2.json              # Cohesin v2 toy任务
│   ├── phase5_wikitext_*.json       # Phase 5.4 WikiText对比
│   ├── phase5_relative_attn.json    # Phase 5.5 相对不变注意力
│   └── phase5_segmented_eraser.json # Phase 5.6 分段Eraser
├── figures/                          # 论文图表
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

---

## Contact

对这个方向感兴趣、愿意一起写论文或探索新范式的朋友，欢迎联系：

**WeChat: llmbbs**

---

## Citation

```bibtex
@misc{huang2025biic,
  title={Bio-Inspired Information Cell: A Geometric Algebra Framework for
         Lossless Information Representation in Language Models},
  author={Huang, Zhongchang},
  year={2025},
  note={Phase 1-5 complete. Key finding: grade-0 invariance verified,
        equivariant components encode syntax but cannot be spontaneously
        activated in LM tasks. Pivoting to sparse geometric architecture.}
}
```

## License

[Business Source License 1.1](LICENSE)
