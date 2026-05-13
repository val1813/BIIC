# Phase 6: 依存句法分析 MVP

## 目标

在有明确句法监督的任务（依存句法分析）上验证 BIIC。如果 grade-2 真的编码了句法，在这个任务上应该有优势。

## 实验设计

1. **主实验**：BIIC (grade-2 only) + Biaffine 分类头 vs Transformer + Biaffine 分类头
2. **数据效率**：用 10%/25%/50%/100% 训练数据对比学习曲线
3. **判决性实验 M-v2/v3**：测试 grade-2 的几何积/余弦/楔积能否区分依存关系类型

数据：Universal Dependencies English EWT (CoNLL-U 格式)

## 操作步骤

```bash
ssh -p 22103 linux@61.157.218.59
source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch2.4_cuda12.1
export http_proxy=http://proxy.mornai.cn:7890
export https_proxy=http://proxy.mornai.cn:7890

# 主实验：BIIC vs Transformer
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase6/dep_parsing_mvp.py

# 数据效率实验
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase6/dep_parsing_efficiency.py

# 判决性实验
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase6/exp_m_v2_dep_geometric.py
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase6/exp_m_v3_cosine.py
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase6/exp_m_v3_wedge.py
```

## 结果

### 主实验

| 模型 | UAS | LAS | 参数量 | 训练时间 |
|------|:---:|:---:|:---:|:---:|
| BIIC (grade-2 only) + Biaffine | 0.279 | 0.225 | 2.5M | 25.3 min |
| Transformer + Biaffine | **0.752** | **0.681** | 2.3M | 1.2 min |

### 数据效率

| 数据量 | BIIC UAS | Transformer UAS | 差距 |
|:---:|:---:|:---:|:---:|
| 10% | ~0.22 | 0.495 | -27pp |
| 25% | 0.256 | 0.617 | -36pp |
| 100% | 0.279 | 0.752 | -47pp |

任意数据量下 BIIC 均远差于 Transformer，无交叉点。

### 判决性实验

| 操作 | Cohen's d | 探针准确率 | 结论 |
|------|:---:|:---:|------|
| 几何积 (M-v2) | -0.157 | 0.44 | 效应反向 |
| 余弦相似度 (M-v3a) | 0.034 | 0.42 | 无差异 |
| 楔积 (M-v3b) | -0.173 | 0.42 | 效应反向 |

## 结论

BIIC 在依存句法分析上全面失败（UAS 差 47pp）。

原因分析：
1. BIICLayer 无跨 token 交互（逐 token 独立更新），无法建模 token 间依赖
2. Grade-2 有句法信息（Probing: DEP=0.823），但几何操作无法提取
3. 信息存在于线性子空间中，不在代数结构中

这是 BIIC 方向的最终否定实验：即使在最需要关系建模的任务上，代数操作也无法利用 grade-2 编码的信息。

## 文件说明

| 文件 | 说明 |
|------|------|
| `scripts/dep_parsing_mvp.py` | BIIC vs Transformer 主实验 |
| `scripts/dep_parsing_efficiency.py` | 数据效率对比 |
| `scripts/exp_m_v2_dep_geometric.py` | 几何积判决实验 |
| `scripts/exp_m_v3_cosine.py` | 余弦相似度实验 |
| `scripts/exp_m_v3_wedge.py` | 楔积实验 |
| `results/phase6_biic_dep.json` | BIIC 依存结果 |
| `results/phase6_transformer_dep.json` | Transformer 依存结果 |
| `results/exp_m_v2_dep_geometric.json` | 几何积结果 |
| `results/exp_m_v3_cosine.json` | 余弦结果 |
| `results/exp_m_v3_wedge.json` | 楔积结果 |
