# BIIC: Bio-Inspired Information Cell

**基于几何代数 Cl(4,1) 的语言模型信息表示框架**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Phase 1-6](https://img.shields.io/badge/Phases%201--6-Complete-brightgreen)]()
[![Phase 0](https://img.shields.io/badge/Phase%200%20(SFE)-Closed-red)]()

---

## 关于这个项目

我没有编程背景，也没有学术研究经历。这个项目是我尝试用 AI 作为研究伙伴，去探索一个我认为有意思的问题：**能否找到比 token embedding 更好的语义传送单元？**

72 小时密集实验，7 个阶段，30+ 个对比实验。最终结论是：这条路走不通——但过程中积累了大量关于"什么不行、为什么不行"的实证数据。

我把完整的实验代码、数据和分析都开源在这里。如果你对以下方向感兴趣，欢迎一起探索：
- 几何代数在深度学习中的应用
- Token 表示的结构化设计
- 用实验否定假设的研究方法论

**这不是一个成功的项目，但它是一张诚实的地图——标注了哪些路走不通。**

---

## 核心思想

当前语言模型的 token 表示有一个根本问题：所有语义信息压在一个扁平向量里，推理过程中被逐层覆盖。

BIIC 的设想：从 DNA 的基因组/表观基因组分离中获得灵感，用 Clifford 几何代数 Cl(4,1) 构建一种新的 token 表示：

```
DNA Architecture          →    BIIC Architecture
Genome (immutable)        →    Grade-0 (invariant)
Epigenome (read/write)    →    Grade-1~4 (equivariant)
TET demethylase (erase)   →    GradeAwareEraser
```

关键性质：**Cl(4,1) 在同一结构中同时提供不变量和等变量**——grade-0 在 sandwich 积变换下严格不变（数学定理保证），grade-2 随变换协变。

---

## 实验总览

| Phase | 目标 | 关键结果 | 结论 |
|:---:|------|---------|:---:|
| 1 | 数学验证 | Grade-0 不变性误差 < 10⁻⁶，零泄漏 | ✅ |
| 2 | 编解码链路 | 等变分量提供 5.3× 解码改善 | ✅ |
| 3 | 假设检验 | 几何结构 > 正交约束 > 纯高维度 (14.6σ) | ✅ |
| 4 | 语言模型训练 | PPL=390 (vs Transformer 53.9)；Grade-2 自发编码句法 | ⚠️ |
| 5 | 等变分量激活 | 13 个实验全部失败，alpha 从未超过 0.029 | ❌ |
| 6 | 依存句法 | UAS 0.279 vs Transformer 0.752 (差 47pp) | ❌ |
| 0 | SFE 微共生 | Transformer attention 系统性压制 embedding 层调制 | ❌ |

---

## 关键发现

### 已证明

| 发现 | 证据 |
|------|------|
| Grade-0 在推理中严格不变 | 100次变换误差 10⁻⁶，精确零泄漏 |
| 等变分量携带独立语义信息 | 5.3× 解码改善 |
| 几何结构优于正交约束 | Phase 3: A1 < B |
| 等变结构远优于纯高维度 | Phase 3: 14.6σ 差异 |
| Grade-2 自发编码句法 | POS=0.789, DEP=0.823（无监督） |

### 已证伪

| 假设 | 否定证据 |
|------|---------|
| 等变分量在 LM 中自发激活 | 13 个实验，alpha 从未超过 0.029 |
| 几何积能提取句法关系 | Cohen's d = -0.157（效应反向） |
| BIIC 比 Transformer 计算效率更高 | Sandwich 积约 360× 开销 |
| 动态 embedding 在 Transformer 中存活 | 4 轮 SFE 实验一致被压制 |
| BIIC 在依存句法上有优势 | UAS 差 47pp，无数据效率交叉点 |
| 长序列/深网络让等变分量更有效 | alpha 反而更低，PPL 更差 |

### 核心洞察

> **信息 ≠ 效用。** Grade-2 有句法信息（Probing 证明），但 LM 任务不需要通过几何操作来利用它。信息存在于线性子空间中，不在代数结构中。

> **语言 ≠ 物理。** 等变性在分子设计（SE(3)）和 DNA（互补链对称）中有效，因为有强制性的物理对称性。语言没有这样的约束。

---

## 假设验证方法论（5 道闸门）

这是在项目中通过失败总结出来的方法论。每次提出新假设，用这五道闸门自我攻击：

**闸门 1：核心功能测试**  
假设声称的能力能否在计算效率上持平或更优？（BIIC 的 sandwich 积比 attention 贵 360×，这在提假设时就应该估算）

**闸门 2：成功条件迁移**  
前人类似工作的成功依赖哪些前提条件？这些条件在当前场景下是否存在？（等变性在蛋白质上成功因为有 SE(3)，语言没有）

**闸门 3：消融预判**  
实验前能否写下"完整版应该比简化版好 X 点"？写不出来说明假设没有被精确定义。

**闸门 4：任务适配性**  
数学上的优美不等于任务需要。先问"这个任务真的需要这个机制吗"。

**闸门 5：最小可证伪点**  
这个假设最可能在哪里第一个失败？失败标准必须在实验前写死，不允许实验中修改。

---

## 仓库结构

```
BIIC/
├── README.md                 # 本文件
├── LICENSE                   # MIT
├── KNOWLEDGE_BASE.md         # 完整知识库（详细技术记录）
├── requirements.txt
├── src/                      # 核心实现
│   ├── clifford_cl41.py      # Cl(4,1) 几何代数
│   ├── rotor_utils.py        # 旋转子 & sandwich 积
│   ├── eraser_ops.py         # GradeAwareEraser
│   ├── token_to_ic.py        # 编码器
│   ├── all_grade_decoder.py  # 全 grade 门控解码器
│   ├── mutable_state.py      # BIICLayer
│   └── ...
├── tests/                    # 验证测试
├── figures/                  # 论文图表
├── phase0/                   # SFE 微共生验证（第二阶段）
├── phase1/                   # 数学验证
├── phase2/                   # 编解码链路
├── phase3/                   # 假设检验
├── phase4/                   # 语言模型训练
├── phase5/                   # 等变分量激活
└── phase6/                   # 依存句法 MVP
```

每个 phase 目录下包含：
- `README.md` — 实验目标、设计、操作步骤、结果、结论
- `scripts/` — 可复现的实验脚本
- `results/` — 完整实验数据（JSON）

---

## 快速开始

```bash
pip install torch numpy scipy matplotlib

# Phase 1 数学验证 (CPU, ~2min)
python phase1/scripts/test_phase1.py

# Phase 2 编解码链路 (CPU, ~10min)
python phase2/scripts/test_decoder_basic.py
python phase2/scripts/test_encoder.py
python phase2/scripts/test_full_pipeline.py
```

GPU 实验需要 NVIDIA GPU (推荐 24GB+) 和 WikiText-103 数据集。详见各 phase 的 README。

---

## 参考文献

- Brehmer et al., 2023. [Geometric Algebra Transformer (GATr)](https://arxiv.org/abs/2305.18415). NeurIPS 2023.
- Li et al., 2025. [Versor: A Geometric Sequence Architecture](https://arxiv.org/abs/2602.10195).
- 2025\. [Toward a Functional Geometric Algebra for NLP](https://arxiv.org/abs/2604.25902).
- 2025\. [All You Need is Geometric Algebra (CliffordNet)](https://arxiv.org/abs/2601.06793).
- Wu & Zhang, 2017. TET-mediated active DNA demethylation. *Nature Reviews Genetics*.

---

## 一起探索

这个项目证明了一件事：即使没有专业背景，借助 AI 工具也可以进行系统性的假设验证研究。虽然最终结论是否定的，但否定本身就是有价值的贡献——它告诉后来者这条路走不通，以及为什么走不通。

如果你对几何代数、表示学习、或者"用 AI 做研究"这个方向感兴趣，欢迎：
- Fork 这个仓库，在你感兴趣的方向上继续探索
- 提 Issue 讨论新的假设或实验设计
- 联系我一起写论文或设计新实验

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
