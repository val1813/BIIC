# Phase 4: 语言模型训练

## 目标

验证 BIIC 多向量能否作为语言模型的信息承载物，在真实文本（WikiText-103）上学习 next-token prediction。同时建立 Transformer baseline 作为对照。

## 实验设计

1. **BIIC v0.2 LM**：完整 BIIC 架构（73M 参数），WikiText-103 训练
2. **Transformer baseline**：标准 Transformer（52M 参数），同数据 50k 步训练至收敛
3. **显存对比**：不同序列长度下 BIIC vs Transformer 的 VRAM 增长
4. **Probing 分析**：用线性探针检测 grade-2 是否自发编码句法信息

## 操作步骤

```bash
ssh -p 22103 linux@61.157.218.59
source ~/anaconda3/etc/profile.d/conda.sh && conda activate torch2.4_cuda12.1
export http_proxy=http://proxy.mornai.cn:7890
export https_proxy=http://proxy.mornai.cn:7890

# BIIC v0.2 训练
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase4/train_biic_lm_v02.py

# Transformer baseline (50k步)
CUDA_VISIBLE_DEVICES=1 python /data/biic/phase4/train_transformer_50k.py

# 显存对比实验
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase4/memory_scaling_experiment.py

# Probing 分析
CUDA_VISIBLE_DEVICES=0 python /data/biic/phase4/probing.py
```

## 结果

### 语言模型对比

| 指标 | BIIC v0.2 | Transformer |
|------|:---:|:---:|
| 参数量 | 73M | 52M |
| 训练步数 | 4,600 (stopped) | 50,000 (converged) |
| Final PPL | 390 | **53.9** |
| Peak VRAM | 1,584 MB | 2,558 MB |
| 速度 | 0.10 step/s | 36 step/s |

### 显存对比 (batch=1, 48GB GPU)

| seq_len | BIIC (MB) | Transformer (MB) | 比值 |
|:---:|:---:|:---:|:---:|
| 256 | 526 | 664 | **0.79** |
| 512 | 870 | 854 | 1.02 |
| 1024 | 1548 | 1388 | 1.12 |
| 2048 | 2907 | 2468 | 1.18 |
| 8192 | 11049 | 9128 | 1.21 |

### Probing 结果

| 任务 | Grade-2 | Grade-0 |
|------|:---:|:---:|
| POS (词性) | **0.789** | — |
| DEP (依存) | **0.823** | — |
| GLUE 情感 | **0.95** | 0.49 |
| GLUE 词类 | **0.84** | 0.49 |

## 结论

BIIC 能学语言（PPL 从 58895 降到 390），但远未收敛。Sandwich 积导致训练速度慢 360×，这是工程瓶颈而非架构缺陷。

关键发现：**Grade-2 自发编码句法**（POS=0.789, DEP=0.823），无需任何显式监督。这证明几何代数结构确实能自然地组织语法信息。但 grade-0 的 GLUE 探针仅 0.49（接近随机），说明 grade-0 是纯粹的身份标识，不携带语义分类信息。

显存方面：BIIC 仅在 seq=256 时更省（0.79×），长序列下比 Transformer 多 21%，需要 CUDA 优化（批量化 sandwich 积）。

## 文件说明

| 文件 | 说明 |
|------|------|
| `scripts/train_biic_lm_v02.py` | BIIC v0.2 训练脚本 |
| `scripts/train_transformer_50k.py` | Transformer baseline 训练 |
| `scripts/memory_scaling_experiment.py` | 显存对比实验 |
| `scripts/probing.py` | POS/DEP/GLUE 线性探针 |
| `results/phase4_transformer_baseline_full.json` | Transformer 50k 步结果 |
| `results/phase4_memory_scaling_extended.json` | 显存对比数据 |
| `results/probing_results.json` | Probing 结果 |
| `results/v02_train_log.json` | BIIC v0.2 训练日志 |
