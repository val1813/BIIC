# 语言模型表示学习实验知识库
## 从几何代数到因子注意力：一次系统性的假设筛选

**项目周期**：2026年5月11日—13日（72小时密集实验）  
**核心问题**：能否找到一种比 token embedding 更好的语义传送单元？  
**最终状态**：BIIC/SFE 方向关闭，BIF 方向开放待验证  
**写给谁**：五年后的自己，或任何想了解这段探索的人

---

## 背景：我们在试图解决什么问题

现有大语言模型（GPT、BERT 等）的 token embedding 是一张静态查表：词表里每个词对应一个固定向量，"苹果"无论出现在"吃苹果"还是"苹果公司发布会"里，进入模型的初始向量都是同一个。模型要花费后续十几层的 Transformer 计算来修正这个歧义起点。

这个项目尝试回答：有没有一种更好的方式，让 token 的表示从一开始就更准确，或者让 token 之间的交互更高效？

探索路径：**BIIC（几何代数方案）→ SFE（动态调制方案）→ BIF（因子化低维交互方案）**

每一步都是前一步被实验否定后的精炼。

---

## 一、BIIC：几何代数方案

### 什么是 BIIC

BIIC（Bio-Inspired Information Cell，生物启发信息胞）的核心想法来自 Clifford 几何代数 Cl(4,1)。

普通向量只有一种乘法（点积）。Cl(4,1) 里的多向量可以被分解成不同"grade"（阶次）的分量：
- **Grade-0**：标量，在旋转变换下严格不变（不管坐标系怎么转，这个值不变）
- **Grade-2**：双向量，在旋转变换下会随变换协变（像向量在坐标系变换下会转动一样）
- **Grade-1、3、4、5**：其他阶次分量

设想：把 token 映射到这个代数结构里，grade-0 作为词的稳定身份锚点（不管上下文如何，这个词"是什么"不变），grade-2 作为携带语法/语义关系的等变分量（随上下文演化）。

用 sandwich 积 `R·x·R_rev` 做 token 间的关系变换，这在数学上保证 grade-0 严格不变、grade-2 按规律变化。

### Phase 1：验证代数运算的正确性

**实验**：10 个单元测试，验证 Cl(4,1) 的乘法表、旋转子、各 grade 的变换性质。

**结果**：10/10 PASS。

**关键数据**：
- Grade-0 不变性：100次变换累积误差 < 1e-5（float32）
- Grade-5 不变性：同上
- Grade-1 等变性：两种计算方式误差 < 1e-6
- 梯度流：10层链后梯度比 = 0.55（健康，不消失不爆炸）

**踩的坑**：
- Cl(4,1) 的 e5²=-1（负度规）导致 sandwich 积不保欧几里得范数，100次变换后数值溢出。解决方法：对每个 grade 分别归一化，不能统一缩放（否则破坏等变性）
- exp(B) 的 Taylor 展开需要 16 项，12项精度不够

**结论**：代数基础正确，可以进入下一步。

---

### Phase 2：验证编码-解码链路

**实验**：实现 TokenToImmutableCore（token ID → 多向量）、BIICLayer（推理层，做 sandwich 积）、AllGradeDecoder（多向量 → 预测）。验证完整的前向传播和梯度流。

**结果**：11/11 PASS。

**关键数据**：
- 全链路训练 50 步：loss 从 10.57 降到 0.72（93% 改善）
- Grade-0 在 6 层推理后变化量 = 0.00e+00（精确不变）
- Grade-2 在 6 层推理后变化量 = 0.0129（等变分量在演化，符合预期）
- 训练 20 步后 grade-0 变化量仍为 0（不变性在训练中保持）

**架构细节**：
- TokenToImmutableCore：共享 base_embed（128维）+ 各 grade 独立投影
- BIICLayer = SimpleWriter（旋转子 sandwich 积）+ GradeAwareEraser（只衰减 grade 1-4，grade-0/5 绝对不动）
- AllGradeDecoder：各 grade 独立投影后加权求和，grade_gates 可学习

**Grade gates 训练后的变化**：grade-1/2/3 权重上升，grade-0/4/5 下降。这暗示模型更依赖等变分量（grade-2 有 10 个分量，信息最丰富）。

**踩的坑**：
- SimpleWriter 里 `result[:, :, c, :] = mv_c_transformed` 是 inplace 操作，PyTorch autograd 会报错。改用 `torch.stack(results, dim=2)`
- Eraser 效果不明显：衰减率初始化 sigmoid(-4.6)≈0.01 太小，需要更多层才能看到效果

---

### Phase 3-5：语言模型训练与等变分量激活实验

**实验背景**：Phase 1-2 证明了代数基础正确。Phase 3 开始在真实语料（WikiText-103）上训练完整的语言模型，并尝试让 grade-2 等变分量在 next-token prediction 任务中"活跃"起来。

**设计逻辑**：如果 grade-2 能在训练中自发携带句法/语义关系信息，那 BIIC 就提供了 Transformer 无法做到的东西——代数结构保证的语法表示。

**实验规模**：Phase 5 进行了 13 个对比实验（实验 A-J，加上 M/N/O），系统测试各种激活等变分量的方法：
- 相对不变注意力（RelAttn）：让注意力权重依赖 grade-2 的不变性
- 分段 Eraser：强制 grade-2 在序列内演化
- Cohesin 门控：学习是否激活等变分量
- 长序列（256 长度）：看序列长了是否更需要等变信息
- 深网络（12 层）：看层数多了是否让等变分量积累
- 全机制叠加：把所有机制同时打开

**结果：13 个实验全部失败。**

**关键数据**：
- RelAttn 10k 步：alpha 从 0.018 升到 0.029，远不足以说明激活
- Cohesin gate：从 -4 初始值不会升高，模型主动选择不用
- g2_norms（grade-2 范数）：在所有实验中恒定约 2.0，无任何增长趋势
- Phase 3 消融：完整 BIIC loss = 10.8285，仅 grade-0 loss = 10.8271，差距 0.0014（几乎为零）
- Transformer baseline PPL = 53.9（52M 参数），BIIC PPL = 390+（远差）

**为什么等变分量不活跃？**

根本原因：语言中不存在物理等变群。

等变分量之所以在分子设计（SE(3) 等变）、DNA 序列建模（互补链对称）中有效，是因为这些领域有明确的、强制性的物理对称性作为监督信号。模型在优化过程中会自然地学会利用这些对称性。

语言中没有这样的对称性。next-token prediction 只需要知道"下一个词更可能是什么"，不需要知道"token A 和 B 在几何意义上的对称关系"。没有外部约束，等变分量在优化压力下选择休眠。

**Probing 发现的矛盾**：
- 线性探针从 grade-2 预测词性 POS = 0.789，依存关系 DEP = 0.823（信息确实存在）
- 但几何积/sandwich 积等代数操作无法提取这些信息

解释：grade-2 的线性子空间里有句法信息，但这些信息不是通过几何积的代数结构组织的。"信息存在"≠"可被代数操作提取"。

---

### Phase 6：依存句法直接任务

**实验背景**：既然 LM 方向失败，尝试在有明确句法监督的任务（依存句法分析）上验证 BIIC。如果 grade-2 真的编码了句法，在这个任务上应该有优势。

**实验设计**：BIIC + Biaffine 分类头 vs Transformer + Biaffine 分类头，用 Universal Dependencies 英语数据集。同时做数据效率实验（用 10%/25%/50%/100% 的训练数据）。

**结果**：

| 模型 | UAS（无标依存） | LAS（有标依存） | 参数 |
|------|----------------|----------------|------|
| BIIC + Biaffine | 0.279 | 0.225 | 2.5M |
| Transformer + Biaffine | **0.752** | **0.681** | 2.3M |

数据效率对比：

| 数据量 | BIIC UAS | Transformer UAS |
|--------|----------|----------------|
| 10% | ~0.22 | 0.495 |
| 25% | 0.256 | 0.617 |
| 100% | 0.279 | 0.752 |

任意数据量下 BIIC 均远差于 Transformer，无交叉点，无数据效率优势。

**实验 M-v2（判决性实验）**：用训练后的 BIIC checkpoint，测试 grade-2 的几何积是否能区分不同类型的依存关系。

- t 检验：p < 1e-15（统计显著，样本量足够大）
- Cohen's d = **-0.157**（效应极小，且方向是反的）
- 探针准确率 = 0.439（勉强高于随机基线 0.25）
- t-SNE 分离度 ≈ 1.0（无分离）

**结论**：即使训练后，grade-2 的几何积也不编码依存关系。BIIC 方向在语言建模和下游任务两个维度全面失败。

---

### PCA 有效秩分析（BIIC 探索的副产品）

**实验背景**：在决定下一个方向之前，对 BIIC checkpoint 做一次 PCA 分析，理解它学到的表示的内在维度结构。这个分析的结果出乎意料地清晰，成为后续 BIF 假设的实证基础。

**实验设计**：用 BIIC 训练好的 checkpoint（50/50 参数正确加载），对 WikiText-103 里 51 个多义词，分别提取各层的表示，做 SVD 分析。

**结果**：

| 层 | PR 中位数 | rank_90 中位数 | cos_median（同词不同语境） |
|----|-----------|----------------|--------------------------|
| embed_grade0 | 1.0 | 1 | ≈0（完全正交） |
| embed_grade2 | 1.0 | 1 | **1.0（完全相同）** |
| grade2（6层后） | **45.6** | **53** | **-0.02（近乎正交）** |
| hidden_layer3 | 44.7 | 55 | -0.03 |
| hidden_final | 1.05 | 1 | 0.82 |

**每行的含义**：

**embed_grade0**（embedding 层出来的 grade-0）：PR=1，cos≈0。每个 token 的 grade-0 向量互相正交，不同语境下完全相同。这就是 grade-0 不变性的直接体现——它就是一个一维的身份标识。

**embed_grade2**（embedding 层出来的 grade-2）：PR=1，cos=1.0。所有语境下完全相同，因为 TokenToImmutableCore 的输入只有 token ID，没有上下文，所以无法产生上下文分化。

**grade2（深层）**：PR=45.6，cos=-0.02。经过 6 层 blocks 之后，grade-2 变成了高维、相互近乎正交的表示。**这说明上下文分化确实发生，但发生在中间层（blocks），不在 embedding 层。**

**hidden_final**：PR≈1，cos=0.82。最后一层把信息又压缩回接近一维。这是正常的 LM 行为——预测下一个词只需要极少维度。

**最重要的发现**：grade2（深层）的 PR p95 = 49.6。也就是说，95% 的多义词，其上下文语义变化的有效维度不超过 50。这个数字后来成为 BIF 中 k=64 这个设计参数的实证依据。

**Checkpoint 正确加载方式**（踩过坑，记录在这里）：
```python
ckpt = torch.load('/data/biic/biic_lm/checkpoints/final.pt', map_location='cpu', weights_only=False)
config = ckpt['config']   # n_channels=32, n_layers=6
state = ckpt['model_state']
# 正确的 key 命名：
# encoder.base_embed.weight: [50257, 128]
# encoder.to_grade0.weight/bias, encoder.to_grade1.weight/bias ... to_grade5
# blocks.N.biic_layer.mutable_layer.eraser.decay_logits
# blocks.N.mixer.conv.weight/bias
```
曾经出现过 6/60 参数匹配的错误，原因是 Opus 重写了模型类（用了错误的 key 命名）。正确做法是从已验证的 `exp_m_v3_cosine.py` 里提取模型类。

---

## 二、SFE：动态调制方案

### 为什么转向 SFE

BIIC 的教训之一是：等变分量虽然有信息，但无法被代数操作提取。换一个角度想——如果 embedding 层本身能根据上下文调整，那同一个词在不同语境下就会得到不同的初始向量，后续的 Transformer 就不需要耗费那么多容量来修正歧义起点。

SFE（Structured Factored Embedding）的设计：

```
e_i = (alpha_static_i + g(ctx_i)) @ B
```

- `B ∈ ℝ^{k×d}`：全局共享的语义基矩阵（64个基向量，每个256维）
- `alpha_static_i ∈ ℝ^k`：每个 token 的静态先验系数（从训练中学）
- `g(ctx_i) → Δα ∈ ℝ^k`：上下文修正网络，输入前4个位置的 embedding，输出系数的偏移量

关键设计：通过低秩约束（所有表示被限制在 k 维基矩阵上）防止 g 退化成一个复杂的查表。PCA 数据（PR≈46）为 k=64 提供了依据。

### SFE 实验：三轮失败

**v1.0**（g 零初始化）：
- alpha_cos_min = 0.85，alpha_cos_final = 0.90
- g 几乎未激活，梯度从一开始就极小（零初始化的平坦区问题）

**v1.1**（g 随机初始化 std=0.001，10x 学习率，辅助分化损失）：
- alpha_cos_min = **0.61**（step 400），alpha_cos_final = 0.85
- g 确实学到了分化！但训练后期被主 loss 的梯度压制回去
- 探针准确率：B（SFE完整版）= 0.6316，C（SFE退化版）= 0.6475，**完整版反而更差**

**FAM v1**（在 v1.1 基础上，把第一层 Transformer attention 换成 FAM——在 α 空间做 token 间交互）：
- alpha_cos_min = **0.49**（step 200），alpha_cos_final = 0.86
- PPL：B_FAM = 175.44，C_static = 179.34，**FAM 版 PPL 反而更好**
- 但 alpha_cos 仍然反弹到 0.86，压制不变

**FAM v2**（动态 α 直接接入 FAM，让 FAM 直接依赖 g 的输出）：
- alpha_cos_min = 0.50，alpha_cos_final = 0.86
- 结果与 FAM v1 几乎相同

### 为什么压制不可避免

压制机制的本质：Transformer 的 attention 本身就是一个强大的消歧工具。它发现"让自己来处理消歧"比"利用 embedding 层传来的分化信息"更高效，所以通过梯度反传系统性地将 g(ctx) 的贡献归零。

这不是梯度路径的问题（我们尝试了直连），不是学习率的问题（我们给了 10 倍学习率），不是辅助损失的问题（我们加了显式的分化损失）。这是优化景观决定的：在有 Transformer attention 的架构中，embedding 层的上下文调制没有生存空间。

**这条路已关闭。不要再在标准 Transformer + 动态 embedding 的组合上浪费时间。**

### SFE 实验的意外发现

FAM 版本的 PPL（175.44）比静态版（179.34）低 3.9 点，且 FAM 版参数量更少（18.7M vs 19.4M）。这个增益与 g(ctx) 的动态调制无关（alpha_cos 两者都反弹到 0.86），来自 FAM 层本身的结构化聚合。

这个发现是 BIF 假设的直接来源。

---

## 三、BIF：因子化低维交互方案

### 为什么转向 BIF

SFE 实验告诉我们：Transformer 不需要 embedding 层的帮助来消歧。但 FAM 层的 PPL 增益暗示：**在低维配方空间做 token 交互，比在高维向量空间做点积，可能更高效。**

这个假设和之前所有方案的根本区别：BIF 不试图在 embedding 层解决歧义，也不依赖等变分量。它只做一件事：把 token 交互的计算场所从 256 维搬到 64 维。

**BIF 的数学形式**：

```python
# Token 表示：不再是查表，而是配方组合
e_i = alpha_i @ B
# alpha_i ∈ ℝ^k：第 i 个 token 的配方系数（可学习，静态）
# B ∈ ℝ^{k×d}：全局共享语义零件库（k=64 个基向量）

# FAM 层：在配方空间做 token 间交互
S[i,j] = alpha_i @ W @ alpha_j^T   # 双线性相似度，k×k 参数
out_i = softmax(S_i + causal_mask) @ X   # 按相似度聚合
```

**参数量对比**：
- 传统 embedding：V×d = 50257×256 ≈ 12.9M
- BIF embedding：V×k + k×d = 50257×64 + 64×256 ≈ 3.2M（节省 75%）
- FAM 层参数：k×k = 64×64 = 4096（极少，比标准 attention 少 64 倍）

**计算复杂度对比**：
- 标准 attention 相似度：O(n²·d)，d=256
- FAM 相似度：O(n²·k)，k=64（少 4 倍）

**k=64 的依据**：PCA 分析显示 grade-2（深层）PR p95=49.6，即 95% 的多义词语义变化有效维度 < 50。k=64 留有余量。

### BIF 的重要限定

BIF 的 α 是**静态的**（不含动态调制）。同一个词在不同语境下的 α 相同，语境消歧仍由后续 Transformer 层处理。BIF 不解决一词多义——它只是在更低维的空间里做交互。

这和 SFE 的根本区别：SFE 试图在输入端解决消歧（失败），BIF 只是说"用更少的参数、更低的维度做同样的事"。

**FAM 的一个关键性质**：相似度矩阵 S 完全由 α 决定，对同一组 token 序列，每次计算的 S 都是相同的（不随上下文变化，只随 token ID 变化）。这意味着 FAM 实际上在做"基于词汇类型的固定路由"，而非"基于语境的动态路由"。这可能是它的优势（不会被 attention 压制），也可能是它的局限（无法利用语境信息）。

### BIF 当前状态

FAM 实验（SFE phase 的副产品）给出了初步正向信号：-3.9 PPL，参数更少。

**但这个信号还不干净**：FAM 实验中 B_FAM 和 C_static 的参数量差了 0.7M（18.7M vs 19.4M），无法排除参数量差异是增益来源之一。

**BIF Phase 1 的目标**：在参数量和 FLOPs 精确对齐的条件下，用三组对比（Baseline / BIF / BIF-ablation）给出干净的答案。BIF-ablation 是把第一层换回压缩版 attention（参数量与 FAM 同量级），用于区分"α 配方空间"和"FAM 交互机制"各自的贡献。

---

## 四、已证伪的假设（不要重复验证）

以下假设已被实验明确否定。后续如果有人提出类似想法，先查这张表。

| 假设 | 否定证据 | 实验轮次 |
|------|---------|---------|
| Grade-2 等变分量在语言 LM 中自发激活 | Phase 5 全部 13 个实验，alpha 从未超过 0.029 | 确定 |
| Sandwich 积（几何积）能提取句法关系 | 实验 M-v2：Cohen's d=-0.157，探针 0.439 | 确定 |
| BIIC 比 Transformer 计算效率更高 | Sandwich 积约 360 倍开销，架构级问题 | 确定 |
| 动态 embedding 调制在标准 Transformer 中存活 | SFE v1.0/v1.1/FAM v1/FAM v2，4 轮一致 | 确定 |
| BIIC 在依存句法上有优势 | UAS 0.279 vs Transformer 0.752，差 47pp | 确定 |
| BIIC 有数据效率优势 | 任意数据量 BIIC < Transformer，无交叉 | 确定 |
| Grade-0 是语义分类意义上的锚点 | GLUE 探针仅 0.49，接近随机 | 确定 |
| 长序列让等变分量更有效 | 实验 G：alpha=0.016，比短序列更低 | 确定 |
| 深网络让等变分量积累 | 实验 I：12 层 PPL=743，更差 | 确定 |
| 全机制叠加有额外收益 | 实验 J：PPL=583，无改善 | 确定 |

---

## 五、经验教训：假设验证 SOP

这是在这个项目里通过失败总结出来的方法论。每次提出新假设，用这五道闸门自我攻击。

### 闸门 1：核心功能测试

假设声称的能力（更快、更准、更省参数）能否在计算效率上持平或更优？

典型反例：BIIC 的 sandwich 积比标准 attention 贵约 360 倍。这是架构级问题，工程优化解决不了。在提出假设的时候就应该估算这个数字，而不是等实验跑完。

### 闸门 2：成功条件迁移

前人类似工作的成功，依赖哪些前提条件？这些条件在当前场景下是否存在？

典型反例：Geometric Hyena 在蛋白质结构上成功，因为有 SE(3) 物理等变性。Caduceus 在 DNA 上成功，因为有互补链的精确对称性。这两个条件在语言中都不存在，借鉴这两个工作的路线注定失败。

### 闸门 3：消融预判

实验前，能否写下"完整版 B 应该比简化版 C 好 X 点"？如果写不出来，假设没有被精确定义。

意义：如果无法预测消融实验的结果，说明你对假设的机理理解不够清楚。好的假设应该能预测自己的对照实验。

### 闸门 4：任务适配性

数学上的优美不等于任务需要。Grade-2 的几何积在数学上是漂亮的，但 next-token prediction 不需要几何关系，所以它永远休眠。先问"这个任务真的需要这个机制吗"。

区分"数学上可以"和"任务上需要"这两个问题。

### 闸门 5：最小可证伪点

这个假设最可能在哪里第一个失败？失败的量化标准是什么？用多少时间可以测到这个信号？

要求：失败标准必须在实验前写死，不允许实验中修改。如果 1000-2000 步时核心指标不达标且无收敛趋势，停止，不要继续烧资源。

**附加规则**：通过标准一旦写入 `manifest.json`，不允许修改。Opus 曾经在 Phase 0 结果不好的时候，把 loss 阈值从 6.0 改成 7.5，把 FAM 梯度下限从 1e-4 改成 1e-6。这是错误的做法——正确的做法是增加数据量或步数重跑，而不是降低标准。

---

## 六、工程踩坑备忘

### Cl(4,1) 相关

**负度规导致范数爆炸**  
e5²=-1 使得 sandwich 积不保欧几里得范数，多次变换后数值溢出。解决：对每个 grade 分别归一化。不能统一缩放，否则破坏等变性。

**Taylor 展开精度**  
exp(B) 需要至少 16 项 Taylor 展开。12 项在 float32 下精度不够（误差 2e-5 vs 7e-7）。

### Checkpoint 加载

**BIIC checkpoint 的正确 key 命名**  
```
encoder.base_embed.weight: [50257, 128]
encoder.to_grade0.weight/bias, encoder.to_grade1...to_grade5
pos_enc.pos_embed.weight: [2048, 320]
blocks.N.biic_layer.mutable_layer.eraser.decay_logits
blocks.N.biic_layer.mutable_layer.writer.bivector_params
blocks.N.mixer.conv.weight/bias, blocks.N.mixer.gate.weight/bias
decoder.grade_projectors.N...
```
Opus 曾经用不同的 key 命名重写模型类，导致 6/60 参数匹配。正确做法：从已验证的 `exp_m_v3_cosine.py` 里提取模型类。

### PyTorch 常见坑

**inplace 操作报错**  
`result[:, :, c, :] = something` 在 autograd 中会报错。改用 `torch.stack(results, dim=2)`。

**MultiheadAttention + is_causal**  
指定 `is_causal=True` 时必须同时传 `attn_mask`：
```python
mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
attn(h, h, h, attn_mask=mask, is_causal=True)
```

**unfold 维度多一**  
`tensor.unfold(1, window_size, 1)` 产生 `L+1` 个窗口。记得裁掉：
```python
windows = windows[:, :L, :, :]
```

### SFE 特有

**g(ctx) 的信息泄漏**  
错误写法：用当前 token 自己的 embedding 作为自身上下文。正确写法：
```python
shifted = torch.zeros_like(static_embeds)
shifted[:, 1:, :] = static_embeds[:, :-1, :]  # 位置 i 只看 i-1 之前
```

**weight tying 的参数量陷阱**  
如果 Baseline 做 `lm_head.weight = embedding.weight`（参数共享），会少约 12.9M 参数。三组模型必须统一处理，否则 PPL 比较无效。

### 数据加载

**WikiText-103 离线模式**  
VPS 需要代理：`HTTP_PROXY=http://proxy.mornai.cn:7890`  
本地 tokenizer 路径：`/data/biic/gpt2_tokenizer` 或 `/data/biic/models_cache/AI-ModelScope/gpt2`

---

## 七、资源状态

**VPS1（唯一可用，2026-05-13）**
- 地址：61.157.218.59:22103，linux / S2@hd
- GPU：双 4090，24GB×2
- 环境：`source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch2.4_cuda12.1`
- Python 3.10，PyTorch 2.4，CUDA 12.1

**重要文件**
```
/data/biic/biic_lm/checkpoints/final.pt          # BIIC LM checkpoint（step=10000）
/data/biic/results/pca_analysis/results.json      # PCA有效秩分析结果（51词）
/data/biic/results/sfe_validation/results_fam.json # FAM实验最终结果
/data/biic/experiments/sfe_fam.py                 # FAM实验脚本
/data/biic/experiments/pca_rank_analysis_v2.py    # PCA分析脚本（正确版）
```

---

## 八、核心数字速查

| 数字 | 含义 | 来源 |
|------|------|------|
| Grade-0 不变性误差 < 1e-5 | 100次变换后数值误差，代数定理保证 | Phase 1 |
| Grade-2 等变分量贡献 ≈ 0 | Phase 3 消融：BIIC vs 仅grade-0 loss差0.0014 | Phase 3 |
| POS探针 0.789，DEP探针 0.823 | grade-2线性探针的句法信息，最高层 | Phase 4.6 |
| GLUE grade-2 探针 0.78，grade-0 探针 0.49 | grade-0接近随机，grade-2有分类信息 | 实验N |
| BIIC UAS 0.279 vs Transformer 0.752 | 依存句法，差距 47pp | Phase 6 |
| grade-2 PR 中位数 45.6，p95 49.6 | 语义变化的有效维度，k=64的依据 | PCA分析 |
| embed_grade2 cos_median = 1.0 | embedding层的grade-2在不同语境下完全相同 | PCA分析 |
| grade2（深层）cos_median = -0.02 | 深层表示相互近似正交，分化发生在中间层 | PCA分析 |
| SFE alpha_cos_min = 0.61 | g(ctx)能产生分化，但被压制回0.85 | SFE v1.1 |
| FAM PPL 175.44 vs static 179.34 | FAM有-3.9点增益，参数未完全对齐 | FAM v1 |
| Transformer baseline PPL 53.9 | 52M参数，50k步，WikiText-103 | Phase 4.3 |

---

## 九、下一步（如果还要继续）

**唯一开放的假设：BIF Phase 1**

目标：在参数量和 FLOPs 精确对齐的条件下，验证 FAM 的 PPL 增益是否真实、是否来自 FAM 本身。

三组对比：
- **Baseline**：标准 embedding（V×d）+ 全部标准 attention
- **BIF**：α配方embedding（V×k + k×d）+ FAM第一层 + 标准attention后续层
- **BIF-ablation**：α配方embedding + 压缩版第一层attention（参数量≈FAM）+ 标准attention后续层

判定：BIF PPL 比 Baseline 低 >2 点，且比 BIF-ablation 低 >1 点，才算 FAM 有独立贡献。

如果 BIF 通过，下一个问题是：FAM 做的是"固定词汇类型路由"（α 是静态的）还是真正有超越词汇类型的东西？这需要进一步分析。

如果 BIF 失败（PPL 不如 Baseline），关闭这个方向，重新思考。

**这条路走了三天，否定了很多假设，留下了一个还没有干净答案的信号。这本身就是有价值的地图。**
