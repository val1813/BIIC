# Phase 2: 编解码链路验证

## 目标

验证完整的编码-解码链路：TokenToImmutableCore → BIICLayer → AllGradeDecoder，确认等变分量（grade-1~4）携带独立于不变核的语义信息，且 grade 分工在训练中自然产生。

## 实验设计

三步验证：
1. **解码器基础测试**（4项）：AllGradeDecoder 能否从多向量恢复 token 预测
2. **编码器测试**（3项）：TokenToImmutableCore 的 grade 分离和区分能力
3. **全链路测试**（4项）：端到端训练中 grade-0 不变性是否保持

## 操作步骤

```bash
# 本地 CPU 即可运行（~10分钟）
pip install torch numpy

python phase2/scripts/test_decoder_basic.py   # Step 1: 4/4 PASS
python phase2/scripts/test_encoder.py         # Step 2: 3/3 PASS
python phase2/scripts/test_full_pipeline.py   # Step 3: 4/4 PASS
```

## 结果

**11/11 PASS**

| 指标 | 值 | 意义 |
|------|-----|------|
| All-grade vs grade-0 only 解码 | **5.3× 改善** (0.006 vs 0.032) | 等变分量携带独立信息 |
| Token 区分度 (cosine sim) | 0.029 ± 0.013 | 不同 token 近乎正交 |
| Grade-0 经 6 层推理后变化 | **0.0 (精确)** | 不变性在推理中保持 |
| 全链路训练 50 步 | loss 10.57 → 0.72 (93%) | 系统可端到端训练 |
| Grade-0 训练 20 步后变化 | 0.00e+00 | 不变性在训练中保持 |

Grade gates 训练后变化：grade-1/2/3 权重上升，grade-0/4/5 下降。模型自然地更依赖等变分量（grade-2 有 10 个分量，信息最丰富）。

## 结论

等变分量携带不变核无法提供的独立语义信息（5.3× 改善）。不同 token 的 grade-0 接近正交（cos_sim=0.03），区分能力强。Grade 分工在训练中自然产生，无需手动设计。

关键踩坑：
- `result[:, :, c, :] = mv_c_transformed` 是 inplace 操作，PyTorch autograd 报错 → 改用 `torch.stack(results, dim=2)`

## 文件说明

| 文件 | 说明 |
|------|------|
| `scripts/test_decoder_basic.py` | Step 1 解码器测试 |
| `scripts/test_encoder.py` | Step 2 编码器测试 |
| `scripts/test_full_pipeline.py` | Step 3 全链路测试 |
| `results/phase2_results.json` | 完整测试结果数据 |
