# Phase 0: SFE微共生验证实验（第二阶段）

> 本实验在 Phase 1-6 完成后进行，验证 Structured Factored Embedding (SFE) 的动态调制假说。

## 目标

验证 `g(ctx)` 能否在 embedding 层产生有意义的上下文分化，并在标准 Transformer 架构中存活。

## 实验设计

核心架构：
```
SFE Embedding:
  embedding = (α_static + g(ctx)) @ B
  其中 g(ctx) = Linear(前4个位置的embedding拼接)

FAM Layer:
  S[i,j] = α_i @ W @ α_j^T  (双线性相似度)
  out = softmax(S + causal_mask) @ x  (加权聚合)
```

四轮递进实验：
1. **v1.0** — g 零初始化基线
2. **v1.1** — g 随机初始化 + 10x 学习率 + 辅助分化损失
3. **FAM v1** — 在 α 空间做 token 间交互（静态 α）
4. **FAM v2** — 动态 α 直接接入 FAM（梯度直连 g）

## 操作步骤

```bash
ssh -p 22103 linux@61.157.218.59
source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch2.4_cuda12.1
export http_proxy=http://proxy.mornai.cn:7890
export https_proxy=http://proxy.mornai.cn:7890

# v1.0 基线
CUDA_VISIBLE_DEVICES=1 python /data/biic/experiments/sfe_validation.py

# v1.1 修复版
CUDA_VISIBLE_DEVICES=1 python /data/biic/experiments/sfe_v11.py

# FAM 实验
CUDA_VISIBLE_DEVICES=1 python /data/biic/experiments/sfe_fam.py
```

## 结果

| 实验 | alpha_cos_min | alpha_cos_final | PPL | 结论 |
|------|:---:|:---:|:---:|------|
| v1.0 | 0.85 | 0.90 | — | g 几乎未激活（零初始化平坦区） |
| v1.1 | **0.61** | 0.85 | — | g 能学但被 attention 压制 |
| FAM v1 | **0.49** | 0.86 | 175.44 | 压制不变 |
| FAM v2 | **0.50** | 0.86 | 174.99 | 压制不变 |

附加发现：PCA 有效秩分析显示 grade-2 深层 PR 中位数 = 45.6，p95 = 49.6，为后续 BIF 方案的 k=64 提供了实证依据。

## 结论

**否定。** 在有 Transformer attention 的架构中，embedding 层的上下文调制没有生存空间。

压制机制的本质：Transformer 的 attention 本身就是强大的消歧工具。它发现"让自己来处理消歧"比"利用 embedding 层传来的分化信息"更高效，通过梯度反传系统性地将 g(ctx) 的贡献归零。

意外收获：FAM 版本 PPL（175.44）比静态版（179.34）低 3.9 点，且参数量更少。这个增益与 g(ctx) 无关，来自 FAM 层本身的结构化聚合，成为后续 BIF 假设的直接来源。

## 文件说明

| 文件 | 说明 |
|------|------|
| `scripts/sfe_validation.py` | v1.0 实验脚本 |
| `scripts/sfe_v11.py` | v1.1 实验脚本（g修复+辅助损失） |
| `scripts/sfe_fam.py` | FAM 实验脚本（v1/v2） |
| `scripts/pca_rank_analysis_v2.py` | PCA 有效秩分析 |
| `results/results.json` | v1.0 结果 |
| `results/results_v11.json` | v1.1 结果 |
| `results/results_fam.json` | FAM v1 结果 |
| `results/results_fam_v2.json` | FAM v2 结果 |
