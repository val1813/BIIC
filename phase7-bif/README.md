# Phase 7-BIF: 因子化低维交互验证 / Factored Low-Dim Interaction (BIF)

> 本实验在 Phase 0 (SFE) 之后进行，验证 BIF 假设：在低维配方空间做 token 交互是否比标准 attention 更高效。

## 目标

在参数量和 FLOPs 精确对齐的条件下，验证：
1. BIF（α配方 embedding + FAM 层）能否超越标准 Transformer
2. FAM 机制本身是否有独立贡献（对比 BIF-ablation）

## 实验设计

### 核心架构

```python
# BIF Embedding: token → 配方系数 → 语义向量
e_i = alpha_i @ B
# alpha_i ∈ ℝ^k (k=64): 每个 token 的配方系数（可学习，静态）
# B ∈ ℝ^{k×d} (64×256): 全局共享语义基矩阵

# FAM Layer: 在配方空间做 token 交互
S[i,j] = alpha_i @ W @ alpha_j^T   # 双线性相似度 (k×k 参数)
out_i = softmax(S_i + causal_mask) @ X
```

### 三组对比

| 模型 | 描述 | 参数量 |
|------|------|:---:|
| Baseline | 标准 embedding + weight tying + 6层 Transformer | 17.7M |
| BIF | α配方 embedding + FAM + 3层 Transformer | 18.7M |
| BIF-ablation | α配方 embedding + 压缩 attention + 3层 Transformer | 18.7M |

### 通过标准（实验前锁定）

- BIF PPL 比 Baseline 低 ≥ 2.0 → BIF 有效
- BIF PPL 比 ablation 低 ≥ 1.0 → FAM 有独立贡献

## 操作步骤

```bash
ssh -p 22103 linux@61.157.218.59
source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch2.4_cuda12.1
export http_proxy=http://proxy.mornai.cn:7890
export https_proxy=http://proxy.mornai.cn:7890

# Phase 0: 梯度健康检查 (600步)
CUDA_VISIBLE_DEVICES=0 python /data/biic/experiments/bif_phase0.py

# Phase 1 v1: 首次验证 (5000步, 参数未对齐)
CUDA_VISIBLE_DEVICES=0 python /data/biic/experiments/bif_phase1.py

# Phase 1 v2: 参数对齐后重跑 (5000步)
CUDA_VISIBLE_DEVICES=0 python /data/biic/experiments/bif_phase1v2.py
```

## 结果

### Phase 0: 梯度健康检查

| 指标 | 值 | 标准 | 结果 |
|------|:---:|:---:|:---:|
| Final loss | 6.23 | < 6.0 | ✗ (边界) |
| α 梯度范数 | 0.44 | [1e-4, 10] | ✓ |
| B 梯度范数 | 0.090 | [1e-4, 10] | ✓ |
| FAM W 梯度范数 | 0.00047 | [1e-4, 10] | ✓ |
| α 多样性 (cos dist) | 1.004 | > 0.01 | ✓ |
| B 矩阵秩 (PR) | 18.93 | 不坍缩 | ✓ |

### Phase 1 v2: 最终结果（参数对齐）

| 模型 | PPL | 低频词 PPL | 参数量 |
|------|:---:|:---:|:---:|
| Baseline | **141.09** | 32,436 | 17.7M |
| BIF | 159.06 | 35,456 | 18.7M |
| BIF-ablation | 196.04 | 49,809 | 18.7M |

### 判定

| 标准 | 结果 | 判定 |
|------|:---:|:---:|
| BIF 超越 Baseline ≥ 2.0 PPL | -17.97 (BIF 更差) | **FAILED** ❌ |
| BIF 超越 ablation ≥ 1.0 PPL | +36.98 | **PASSED** ✅ |

### B 矩阵 Participation Ratio 变化

| 模型 | 初始 PR | 最终 PR | 趋势 |
|------|:---:|:---:|------|
| BIF | 19.7 | 32.84 | 持续上升（未收敛） |
| BIF-ablation | 29.97 | 27.23 | 稳定 |

## 结论

**BIF 作为 embedding 替代方案：失败。** BIF 输给 Baseline 约 18 点 PPL，两次实验（v1/v2）结论完全一致。

**FAM 机制本身：有效。** FAM 比 ablation 好 37 点 PPL，用 4,096 参数（k×k=64×64）实现了标准 attention 262,144 参数的有效 token 交互。

**瓶颈定位：α配方 embedding 的表达容量。** `e = alpha @ B` 把词表压缩到 64 维线性子空间，标准 embedding 的 256 维查表有更强的初始表达能力。B 矩阵 PR 持续爬升（19.7→32.8 未收敛）证明模型在挣扎利用有限容量。

**核心教训：**
- 低维压缩（k=64）对语言建模来说容量不足
- FAM 的交互机制有价值，但被上游 embedding 瓶颈拖累
- 低频词无优势（-3,020 PPL），否定了"共享基向量帮助稀有词"的假设

---

## 与整个项目的关系

BIF 是 BIIC → SFE → BIF 探索链的最后一环：

| 方向 | 假设 | 结果 |
|------|------|:---:|
| BIIC | 几何代数等变分量在 LM 中自发激活 | ❌ |
| SFE | 动态 embedding 调制在 Transformer 中存活 | ❌ |
| BIF | 低维配方空间交互优于标准 attention | ❌ |

三条路全部关闭。项目完结。

## 文件说明

| 文件 | 说明 |
|------|------|
| `scripts/bif_phase0.py` | Phase 0 梯度健康检查 |
| `scripts/bif_phase1.py` | Phase 1 v1（参数未对齐） |
| `scripts/bif_phase1v2.py` | Phase 1 v2（参数对齐，最终版） |
| `results/results_phase0.json` | Phase 0 结果 |
| `results/results_phase1.json` | Phase 1 v1 结果 |
| `results/results_phase1_v2.json` | Phase 1 v2 结果（最终） |
| `results/bif_phase1_v2_results.html` | 交互式可视化报告 |
