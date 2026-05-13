# BIIC: Bio-Inspired Information Cell

**基于几何代数 Cl(4,1) 的语言模型信息表示框架**

**A Geometric Algebra Framework for Lossless Information Representation in Language Models**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Phase 1-6](https://img.shields.io/badge/Phases%201--6-Complete-brightgreen)]()
[![Phase 0](https://img.shields.io/badge/Phase%200%20(SFE)-Closed-red)]()

[中文](#关于这个项目) | [English](#about-this-project)

---

## 关于这个项目

我没有编程背景，也没有学术研究经历。这个项目是我尝试用 AI 作为研究伙伴，去探索一个我认为有意思的问题：**能否找到比 token embedding 更好的语义传送单元？**

72 小时密集实验，7 个阶段，30+ 个对比实验。最终结论是：这条路走不通——但过程中积累了大量关于"什么不行、为什么不行"的实证数据。

我把完整的实验代码、数据和分析都开源在这里。如果你对以下方向感兴趣，欢迎一起探索：
- 几何代数在深度学习中的应用
- Token 表示的结构化设计
- 用实验否定假设的研究方法论

**这不是一个成功的项目，但它是一张诚实的地图——标注了哪些路走不通。**

## About This Project

I have no programming background and no academic research experience. This project is my attempt to use AI as a research partner to explore a question I find fascinating: **Can we find a better semantic carrier than token embeddings?**

72 hours of intensive experimentation, 7 phases, 30+ controlled experiments. The final conclusion: this path doesn't work — but the process yielded substantial empirical data about what fails and why.

All experiment code, data, and analysis are open-sourced here. If you're interested in any of the following, let's explore together:
- Geometric algebra in deep learning
- Structured token representation design
- Hypothesis falsification as research methodology

**This is not a successful project, but it is an honest map — marking which paths are dead ends.**

---

## 核心思想 / Core Idea

当前语言模型的 token 表示有一个根本问题：所有语义信息压在一个扁平向量里，推理过程中被逐层覆盖。

Current LM token representations have a fundamental flaw: all semantics are compressed into a single flat vector that gets overwritten layer by layer during inference.

BIIC 的设想：从 DNA 的基因组/表观基因组分离中获得灵感，用 Clifford 几何代数 Cl(4,1) 构建一种新的 token 表示：

BIIC's approach: inspired by the genome/epigenome separation in DNA, use Clifford algebra Cl(4,1) to build a new token representation:

```
DNA Architecture          →    BIIC Architecture
Genome (immutable)        →    Grade-0 (invariant)
Epigenome (read/write)    →    Grade-1~4 (equivariant)
TET demethylase (erase)   →    GradeAwareEraser
```

关键性质：**Cl(4,1) 在同一结构中同时提供不变量和等变量**——grade-0 在 sandwich 积变换下严格不变（数学定理保证），grade-2 随变换协变。

Key property: **Cl(4,1) provides both invariant and equivariant quantities in one structure** — grade-0 is strictly invariant under sandwich product transformations (guaranteed by theorem), while grade-2 co-varies with transformations.

---

## 实验总览 / Experiment Overview

| Phase | 目标 / Objective | 关键结果 / Key Result | 结论 |
|:---:|------|---------|:---:|
| 1 | 数学验证 / Math verification | Grade-0 invariance error < 10⁻⁶, zero leakage | ✅ |
| 2 | 编解码链路 / Encode-decode pipeline | Equivariant components: 5.3× decoding improvement | ✅ |
| 3 | 假设检验 / Hypothesis testing | Geometric structure > orthogonal > high-dim (14.6σ) | ✅ |
| 4 | 语言模型训练 / LM training | PPL=390 (vs Transformer 53.9); Grade-2 encodes syntax | ⚠️ |
| 5 | 等变分量激活 / Equivariant activation | 13 experiments all failed, alpha never exceeded 0.029 | ❌ |
| 6 | 依存句法 / Dependency parsing | UAS 0.279 vs Transformer 0.752 (47pp gap) | ❌ |
| 0 | SFE 微共生 / SFE micro-symbiosis | Transformer attention systematically suppresses embedding-layer modulation | ❌ |

---

## 关键发现 / Key Findings

### 已证明 / Proven

| 发现 / Finding | 证据 / Evidence |
|------|------|
| Grade-0 在推理中严格不变 / Grade-0 strictly invariant during inference | 100 transforms, error 10⁻⁶, exact zero leakage |
| 等变分量携带独立语义信息 / Equivariant components carry independent semantics | 5.3× decoding improvement |
| 几何结构优于正交约束 / Geometric structure > orthogonal constraint | Phase 3: A1 < B |
| 等变结构远优于纯高维度 / Equivariant structure >> pure high-dim | Phase 3: 14.6σ difference |
| Grade-2 自发编码句法 / Grade-2 spontaneously encodes syntax | POS=0.789, DEP=0.823 (unsupervised) |

### 已证伪 / Disproven

| 假设 / Hypothesis | 否定证据 / Counter-evidence |
|------|---------|
| 等变分量在 LM 中自发激活 / Equivariant activation in LM | 13 experiments, alpha never > 0.029 |
| 几何积能提取句法关系 / Geometric product extracts syntax | Cohen's d = -0.157 (reversed) |
| BIIC 比 Transformer 计算效率更高 / BIIC more efficient than Transformer | Sandwich product ~360× overhead |
| 动态 embedding 在 Transformer 中存活 / Dynamic embedding survives in Transformer | 4 rounds of SFE consistently suppressed |
| BIIC 在依存句法上有优势 / BIIC advantage in dep parsing | UAS gap 47pp, no data-efficiency crossover |
| 长序列/深网络让等变分量更有效 / Longer seq/deeper net helps | alpha actually lower, PPL worse |

### 核心洞察 / Core Insights

> **信息 ≠ 效用。** Grade-2 有句法信息（Probing 证明），但 LM 任务不需要通过几何操作来利用它。信息存在于线性子空间中，不在代数结构中。
>
> **Information ≠ Utility.** Grade-2 has syntactic information (proven by probing), but LM tasks don't need geometric operations to exploit it. Information lives in linear subspaces, not algebraic structure.

> **语言 ≠ 物理。** 等变性在分子设计（SE(3)）和 DNA（互补链对称）中有效，因为有强制性的物理对称性。语言没有这样的约束。
>
> **Language ≠ Physics.** Equivariance works for molecular design (SE(3)) and DNA (complementary strand symmetry) because of mandatory physical symmetries. Language has no such constraints.

---

## 假设验证方法论（5 道闸门）/ Hypothesis Validation Methodology (5 Gates)

这是在项目中通过失败总结出来的方法论。每次提出新假设，用这五道闸门自我攻击：

A methodology distilled from failures in this project. Attack every new hypothesis with these 5 gates:

**闸门 1：核心功能测试 / Gate 1: Core Feasibility**  
假设声称的能力能否在计算效率上持平或更优？（BIIC 的 sandwich 积比 attention 贵 360×，这在提假设时就应该估算）

Can the claimed capability match or beat baseline computational efficiency? (BIIC's sandwich product is 360× more expensive than attention — this should be estimated before proposing the hypothesis)

**闸门 2：成功条件迁移 / Gate 2: Success Condition Transfer**  
前人类似工作的成功依赖哪些前提条件？这些条件在当前场景下是否存在？（等变性在蛋白质上成功因为有 SE(3)，语言没有）

What preconditions enabled prior similar work to succeed? Do those conditions exist here? (Equivariance works for proteins because of SE(3); language has no such symmetry)

**闸门 3：消融预判 / Gate 3: Ablation Prediction**  
实验前能否写下"完整版应该比简化版好 X 点"？写不出来说明假设没有被精确定义。

Can you write down "the full version should beat the ablation by X" before running? If not, the hypothesis isn't precisely defined.

**闸门 4：任务适配性 / Gate 4: Task Fit**  
数学上的优美不等于任务需要。先问"这个任务真的需要这个机制吗"。

Mathematical elegance ≠ task necessity. First ask: "Does this task actually need this mechanism?"

**闸门 5：最小可证伪点 / Gate 5: Minimum Falsification Point**  
这个假设最可能在哪里第一个失败？失败标准必须在实验前写死，不允许实验中修改。

Where is this hypothesis most likely to fail first? Failure criteria must be locked before the experiment — no moving goalposts.

---

## 仓库结构 / Repository Structure

```
BIIC/
├── README.md                 # 本文件 / This file
├── LICENSE                   # MIT
├── KNOWLEDGE_BASE.md         # 完整知识库 / Full technical reference
├── requirements.txt
├── src/                      # 核心实现 / Core implementation
│   ├── clifford_cl41.py      # Cl(4,1) 几何代数 / Geometric algebra
│   ├── rotor_utils.py        # 旋转子 & sandwich 积 / Rotors & sandwich product
│   ├── eraser_ops.py         # GradeAwareEraser
│   ├── token_to_ic.py        # 编码器 / Encoder
│   ├── all_grade_decoder.py  # 全 grade 门控解码器 / All-grade gated decoder
│   ├── mutable_state.py      # BIICLayer
│   └── ...
├── tests/                    # 验证测试 / Verification tests
├── figures/                  # 论文图表 / Paper figures
├── phase0/                   # SFE 微共生验证（第二阶段）/ SFE validation (stage 2)
├── phase1/                   # 数学验证 / Mathematical verification
├── phase2/                   # 编解码链路 / Encode-decode pipeline
├── phase3/                   # 假设检验 / Hypothesis testing
├── phase4/                   # 语言模型训练 / Language model training
├── phase5/                   # 等变分量激活 / Equivariant activation
└── phase6/                   # 依存句法 MVP / Dependency parsing MVP
```

每个 phase 目录下包含 / Each phase directory contains:
- `README.md` — 实验目标、设计、操作步骤、结果、结论 / Objective, design, steps, results, conclusion
- `scripts/` — 可复现的实验脚本 / Reproducible experiment scripts
- `results/` — 完整实验数据（JSON）/ Full experiment data (JSON)

---

## 快速开始 / Quick Start

```bash
pip install torch numpy scipy matplotlib

# Phase 1 数学验证 / Mathematical verification (CPU, ~2min)
python phase1/scripts/test_phase1.py

# Phase 2 编解码链路 / Encode-decode pipeline (CPU, ~10min)
python phase2/scripts/test_decoder_basic.py
python phase2/scripts/test_encoder.py
python phase2/scripts/test_full_pipeline.py
```

GPU 实验需要 NVIDIA GPU (推荐 24GB+) 和 WikiText-103 数据集。详见各 phase 的 README。

GPU experiments require NVIDIA GPU (24GB+ recommended) and WikiText-103 dataset. See each phase's README for details.

---

## 参考文献 / References

- Brehmer et al., 2023. [Geometric Algebra Transformer (GATr)](https://arxiv.org/abs/2305.18415). NeurIPS 2023.
- Li et al., 2025. [Versor: A Geometric Sequence Architecture](https://arxiv.org/abs/2602.10195).
- 2025\. [Toward a Functional Geometric Algebra for NLP](https://arxiv.org/abs/2604.25902).
- 2025\. [All You Need is Geometric Algebra (CliffordNet)](https://arxiv.org/abs/2601.06793).
- Wu & Zhang, 2017. TET-mediated active DNA demethylation. *Nature Reviews Genetics*.

---

## 一起探索 / Join the Exploration

这个项目证明了一件事：即使没有专业背景，借助 AI 工具也可以进行系统性的假设验证研究。虽然最终结论是否定的，但否定本身就是有价值的贡献——它告诉后来者这条路走不通，以及为什么走不通。

This project proves one thing: even without a professional background, systematic hypothesis-driven research is possible with AI tools. Although the final conclusion is negative, negation itself is a valuable contribution — it tells future researchers which paths are dead ends, and why.

如果你对几何代数、表示学习、或者"用 AI 做研究"这个方向感兴趣，欢迎：
- Fork 这个仓库，在你感兴趣的方向上继续探索
- 提 Issue 讨论新的假设或实验设计
- 联系我一起写论文或设计新实验

If you're interested in geometric algebra, representation learning, or "doing research with AI", you're welcome to:
- Fork this repo and explore directions that interest you
- Open an Issue to discuss new hypotheses or experiment designs
- Contact me to collaborate on papers or new experiments

**WeChat: llmbbs**

---

## Citation

```bibtex
@misc{huang2025biic,
  title={Bio-Inspired Information Cell: A Geometric Algebra Framework for
         Lossless Information Representation in Language Models},
  author={Huang, Zhongchang},
  year={2025},
  note={Phases 1-6 complete. Key finding: grade-0 invariance verified,
        equivariant components encode syntax but cannot be activated in LM.
        Negative results documented as contribution.}
}
```

## License

[MIT](LICENSE)
