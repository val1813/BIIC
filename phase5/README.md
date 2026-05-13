# Phase 5: 等变分量激活实验

## 目标

验证能否通过跨 token 机制（Cohesin 门控、相对不变注意力、分段 Eraser）让等变分量在语言模型中自发激活。

背景：Phase 4 Probing 证明 grade-2 自发编码句法（POS=0.789, DEP=0.823），但模型是否主动利用这些信息？

## 实验设计

### 主实验（5.1-5.6）

| 编号 | 机制 | 成功标准 |
|------|------|---------|
| 5.1 | Probing 诊断 | 确认 grade-2 有信息 |
| 5.2 | Cohesin v2 (toy) | gate > 0.1 |
| 5.4 | Cohesin + WikiText | PPL 改善 + gate 自发升高 |
| 5.5 | 相对不变注意力 | alpha > 0.1 |
| 5.6 | 分段 Eraser | grade-2 范数增长 |

### 扩展实验（A-J + M/N/O）

13 个系统性对比：不同通道数、序列长度、网络深度、机制组合。

## 操作步骤

```bash
ssh -p 22103 linux@61.157.218.59
source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch2.4_cuda12.1
export http_proxy=http://proxy.mornai.cn:7890
export https_proxy=http://proxy.mornai.cn:7890

# Cohesin vs Original on WikiText-103
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase5/biic_wikitext_compare.py --use_cohesin False
CUDA_VISIBLE_DEVICES=1 python /data/biic/phase5/biic_wikitext_compare.py --use_cohesin True

# 相对不变注意力 + 分段 Eraser
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase5/biic_v2_experiments.py

# 扩展实验 A-J
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase5/biic_with_cohesin.py --exp A
# ... (A through J)

# 补充实验 M/N/O
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase5/dep_geometric_analysis.py  # M
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase5/glue_probe.py              # N
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase5/routing_stability.py       # O
```

## 结果

### 主实验

| 实验 | PPL | 关键指标 | 结论 |
|------|:---:|---------|------|
| 5.4 Cohesin (gate=-4) | 762.6 | gate=0.025 | gate 不升高，模型选择不用 |
| 5.5 RelAttn | 751.2 | alpha: 0.018→0.015 | alpha 下降，模型主动关闭 |
| 5.6 SegEraser | 764.0 | g2_norms ≈ 2.0 恒定 | 无增长模式 |

### 扩展实验

| 实验 | 配置 | PPL | alpha | 结论 |
|------|------|:---:|:---:|------|
| A | RelAttn 10k | 430.0 | 0.029 | alpha 微升但远不足 |
| B | Combined 5k | 575.1 | 0.022 | 叠加无额外收益 |
| C | Baseline 10k | 472.0 | — | 对照 |
| E | 大通道 16ch | 709.0 | 0.018 | 更差 |
| G | 长序列 seq256 | 574.5 | 0.016 | alpha 反而更低 |
| I | 深网络 12层 | 743.2 | 0.019 | 更差 |
| J | 全机制组合 | 583.2 | 0.021 | 无协同效应 |

### 补充实验

| 实验 | 结论 |
|------|------|
| M: 依存几何积 | Cohen's d = -0.157（效应反向），几何积不编码依存 |
| N: GLUE 探针 | grade-2 = 0.78, grade-0 = 0.49（信息存在但不被利用） |
| O: 路由稳定性 | BIIC=0.133 vs SRA=0.129（无差异） |

## 结论

**13 个实验全部失败。** 等变分量无法被任何机制自发激活。

根本原因：语言中不存在物理等变群。等变分量在分子设计（SE(3)）和 DNA（互补链对称）中有效，是因为有强制性的物理对称性。语言没有这样的约束，next-token prediction 不需要显式的关系表示。

关键洞察：**信息 ≠ 效用**。Grade-2 有句法信息（Probing 证明），但 LM 任务不需要通过几何操作来利用它。信息存在于线性子空间中，不在代数结构中。

## 文件说明

| 文件 | 说明 |
|------|------|
| `scripts/biic_wikitext_compare.py` | Phase 5.4 对比脚本 |
| `scripts/biic_v2_experiments.py` | Phase 5.5/5.6 实验 |
| `scripts/biic_with_cohesin.py` | 扩展实验 A-J |
| `scripts/dep_geometric_analysis.py` | 实验 M |
| `scripts/glue_probe.py` | 实验 N |
| `scripts/routing_stability.py` | 实验 O |
| `results/phase5_*.json` | 主实验结果 |
| `results/extended/*.json` | 13 个扩展实验结果 |
