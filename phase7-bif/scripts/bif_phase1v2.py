"""
BIF Phase 1 — 核心假设正式验证
目标：在参数量精确对齐的条件下，验证BIF（FAM层）是否比等参数量的标准Transformer更好。

三组模型（参数量尽量对齐）：
  Baseline    : 标准embedding + 5层标准Transformer（多一层补齐BIF省掉的embedding参数）
  BIF         : α配方embedding + FAM第一层 + 3层标准Transformer
  BIF-ablation: α配方embedding + 压缩attention第一层 + 3层标准Transformer
                消融FAM，区分"配方空间的价值"和"FAM机制的价值"

判定标准（实验前锁死，不允许修改）：
  主要：BIF val_ppl < Baseline val_ppl - 2.0
  消融：BIF val_ppl < BIF-ablation val_ppl - 1.0（FAM有独立贡献）
  额外：低频词PPL（词频后20%），BIF是否有更大优势

B矩阵秩监控（Phase 0发现的预警）：
  B的PR在step 150附近从50急剧坍缩到18，之后稳定
  Phase 1自动监控，PR < 10时激活正交正则化

运行方式：
  cd /data/biic
  CUDA_VISIBLE_DEVICES=1 nohup python -u experiments/bif_phase1.py \
    > /data/biic/logs/bif_phase1.log 2>&1 &

预计时间：6-8小时（4090单卡，三组串行）
"""

import os
import sys
import json
import math
import random
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# ============================================================
# 配置
# ============================================================
CONFIG = {
    "k": 64,
    "d": 256,
    "n_heads": 4,

    # 层数设置（v2参数对齐方案）
    # BIF:          1 FAM + 3 TF，params≈18.7M
    # BIF-ablation: 1 compressed-attn + 3 TF，params≈18.7M
    # Baseline:     weight_tying + 6 TF层，params≈19.1M（误差<3%）
    #
    # 关键修复：Baseline改为weight tying（embed和lm_head共享），
    # 省掉Phase 1 v1中重复的12.9M参数，然后加一层TF补回，
    # 使三组参数量对齐到约18-19M
    "bif_n_layers":      4,
    "ablation_n_layers": 4,
    "baseline_n_layers": 6,

    # 训练
    "seq_len":      128,
    "batch_size":   32,
    "lr":           3e-4,
    "fam_lr_mult":  5.0,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "total_steps":  5000,
    "grad_clip":    1.0,
    "seed":         42,

    # 监控
    "log_every":     100,
    "measure_every": 500,
    "probe_every":   1000,

    # 正交正则（B矩阵坍缩时自动激活）
    "ortho_reg_threshold": 10,

    # 路径
    "device":      "cuda:0",
    "results_dir": "/data/biic/results/bif_phase1_v2",
    "log_file":    "/data/biic/logs/bif_phase1_v2.log",

    # 判定标准（锁死）
    "pass_criteria": {
        "bif_vs_baseline_ppl_gap":  2.0,
        "bif_vs_ablation_ppl_gap":  1.0,
        "early_stop_steps":         2000,
        "early_stop_gap":           3.0,
    }
}

POLYSEMOUS_WORDS = [
    "book", "run", "bank", "light", "right", "fire", "deal",
    "lead", "match", "check", "clear", "fall", "field", "form",
    "ground", "head", "interest", "key", "draw", "figure"
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 数据集
# ============================================================
class WikiTextDataset(Dataset):
    def __init__(self, tokenized_ids, seq_len):
        self.seq_len = seq_len
        n = (len(tokenized_ids) // seq_len) * seq_len
        self.data = tokenized_ids[:n].reshape(-1, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


def load_data(config):
    from transformers import AutoTokenizer
    from datasets import load_dataset

    os.environ["HTTP_PROXY"]  = "http://proxy.mornai.cn:7890"
    os.environ["HTTPS_PROXY"] = "http://proxy.mornai.cn:7890"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-103-v1", trust_remote_code=True)

    train_lines = [l for l in dataset["train"]["text"]      if l.strip()][:50000]
    val_lines   = [l for l in dataset["validation"]["text"] if l.strip()][:2000]

    train_ids = np.array(tokenizer.encode("\n".join(train_lines)))
    val_ids   = np.array(tokenizer.encode("\n".join(val_lines)))

    return train_ids, val_ids, tokenizer.vocab_size, tokenizer


# ============================================================
# 模型模块（与Phase 0完全一致，无改动）
# ============================================================
class BIFEmbedding(nn.Module):
    def __init__(self, vocab_size, k=64, d=256):
        super().__init__()
        self.k = k
        self.d = d
        self.B     = nn.Parameter(torch.randn(k, d) * 0.02)
        self.alpha = nn.Embedding(vocab_size, k)
        nn.init.normal_(self.alpha.weight, std=0.02)

    def forward(self, token_ids):
        return torch.matmul(self.alpha(token_ids), self.B)

    def get_alpha(self, token_ids):
        return self.alpha(token_ids)


class FAMLayer(nn.Module):
    """
    Factor-Aware Mixing：在α配方空间（k=64维）做token间交互。
    相似度矩阵S完全由α决定：S[i,j] = alpha_i @ W @ alpha_j^T
    参数量：k×k = 4096，比标准attention少64倍。
    计算量：O(n²k)，k=64 << d=256。
    """
    def __init__(self, k=64, d=256, dropout=0.1):
        super().__init__()
        self.k       = k
        self.sim_W   = nn.Parameter(torch.randn(k, k) * 0.01)
        self.out_proj = nn.Linear(d, d)
        self.ln       = nn.LayerNorm(d)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x, alpha):
        B, L, _ = x.shape
        alpha_W = torch.matmul(alpha, self.sim_W)
        S = torch.matmul(alpha_W, alpha.transpose(-1, -2))
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=x.device), diagonal=1
        )
        S   = F.softmax(S + causal_mask, dim=-1)
        out = self.dropout(self.out_proj(torch.matmul(S, x)))
        return self.ln(x + out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        L    = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h    = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x    = x + self.dropout(h)
        return x + self.ffn(self.ln2(x))


# ============================================================
# 三个模型
# ============================================================
class BIFModel(nn.Module):
    """α配方embedding + FAM第一层 + (n_layers-1)层标准Transformer"""
    def __init__(self, vocab_size, k=64, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.bif_embed = BIFEmbedding(vocab_size, k=k, d=d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        self.fam    = FAMLayer(k=k, d=d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers - 1)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x    = self.bif_embed(input_ids)
        alpha = self.bif_embed.get_alpha(input_ids)
        x    = x + self.pos_embed(pos)
        x    = self.fam(x, alpha)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_f(x))


class BIFAblationModel(nn.Module):
    """
    消融对照：α配方embedding + 压缩attention + (n_layers-1)层标准Transformer
    与BIF唯一区别：第一层用压缩attention（QK投影到16维，参数量与FAM可比）
    目的：区分"低维配方空间的价值"和"FAM交互机制的价值"
    """
    def __init__(self, vocab_size, k=64, d_model=256, n_layers=4,
                 n_heads=4, compressed_dim=16):
        super().__init__()
        self.bif_embed      = BIFEmbedding(vocab_size, k=k, d=d_model)
        self.pos_embed      = nn.Embedding(512, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        self.compressed_dim = compressed_dim
        self.Wq  = nn.Linear(d_model, compressed_dim, bias=False)
        self.Wk  = nn.Linear(d_model, compressed_dim, bias=False)
        self.Wv  = nn.Linear(d_model, d_model, bias=False)
        self.Wo  = nn.Linear(d_model, d_model, bias=False)
        self.ln1     = nn.LayerNorm(d_model)
        self.ln_attn = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)
        self.layers  = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers - 1)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x    = self.bif_embed(input_ids) + self.pos_embed(pos)
        h    = self.ln1(x)
        Q, K, V = self.Wq(h), self.Wk(h), self.Wv(h)
        scale = math.sqrt(self.compressed_dim)
        S     = torch.matmul(Q, K.transpose(-1, -2)) / scale
        mask  = torch.triu(
            torch.full((L, L), float('-inf'), device=x.device), diagonal=1
        )
        S   = F.softmax(S + mask, dim=-1)
        out = self.dropout1(self.Wo(torch.matmul(S, V)))
        x   = self.ln_attn(x + out)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_f(x))


class BaselineModel(nn.Module):
    """
    标准Transformer：标准embedding + weight tying + 6层attention。

    v2修复：加入weight tying（lm_head.weight = embed.weight），
    省掉Phase 1 v1中重复的12.9M lm_head参数。
    改为6层TF来部分补回，使总参数量≈19.1M，与BIF/ablation的18.7M对齐（误差<3%）。

    参数量计算：
      embed:    50257 × 256 = 12.9M（与lm_head共享，不重复计算）
      pos_embed: 512 × 256  = 0.13M
      6层TF:    每层约1.05M × 6 = 6.3M
      ln_f:     可忽略
      lm_head:  与embed共享，不新增参数
      总计:     ≈19.3M
    """
    def __init__(self, vocab_size, d_model=256, n_layers=6, n_heads=4):
        super().__init__()
        self.embed     = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.pos_embed = nn.Embedding(512, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        self.layers    = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying：lm_head和embed共享参数，标准LM做法
        # 效果：省掉12.9M重复参数，与BIF/ablation的参数量对齐
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos  = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x    = self.embed(input_ids) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_f(x))


# ============================================================
# 监控工具
# ============================================================
def cosine_lr(step, total_steps, warmup_steps, base_lr):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_ppl(model, loader, device, max_batches=80):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids    = batch.to(device)
            logits = model(ids)
            V      = logits.shape[-1]
            loss   = F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                ids[:, 1:].reshape(-1),
                reduction="sum"
            )
            total_loss   += loss.item()
            total_tokens += ids[:, 1:].numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


def compute_lowfreq_ppl(model, loader, device, tokenizer, top_pct=0.2,
                        max_batches=80):
    """
    计算低频词（词频后20%）的PPL。
    BIF的共享零件库理论上对低频词更友好（可以从高频词借用基向量）。
    如果BIF的低频词PPL优势大于全局PPL优势，支持这个理论。
    """
    # 先统计词频
    freq_counter = Counter()
    cnt = 0
    for batch in loader:
        for seq in batch:
            freq_counter.update(seq.tolist())
        cnt += 1
        if cnt >= 100:
            break

    sorted_tokens = sorted(freq_counter.items(), key=lambda x: x[1])
    cutoff        = int(len(sorted_tokens) * top_pct)
    low_freq_ids  = set(t for t, _ in sorted_tokens[:cutoff])

    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids     = batch.to(device)
            logits  = model(ids)
            V       = logits.shape[-1]
            targets = ids[:, 1:]
            preds   = logits[:, :-1]
            mask    = torch.zeros_like(targets, dtype=torch.bool)
            for tid in low_freq_ids:
                mask |= (targets == tid)
            if mask.sum() == 0:
                continue
            loss = F.cross_entropy(
                preds.reshape(-1, V)[mask.reshape(-1)],
                targets.reshape(-1)[mask.reshape(-1)],
                reduction="sum"
            )
            total_loss   += loss.item()
            total_tokens += mask.sum().item()
    model.train()
    if total_tokens == 0:
        return float('inf')
    return math.exp(total_loss / total_tokens)


def check_B_rank(model):
    """检查BIF的B矩阵有效秩（Participation Ratio和rank_90）"""
    if not hasattr(model, 'bif_embed'):
        return None
    with torch.no_grad():
        B_matrix = model.bif_embed.B.detach().cpu()
        _, S, _  = torch.linalg.svd(B_matrix, full_matrices=False)
        s2    = S ** 2
        total = s2.sum().item()
        if total < 1e-12:
            return None
        PR      = float((s2.sum() ** 2 / (s2 ** 2).sum()).item())
        cumvar  = torch.cumsum(s2, dim=0) / total
        rank_90 = int((cumvar < 0.90).sum().item() + 1)
        top1    = float((s2[0] / total).item())
        top5    = float((s2[:5].sum() / total).item())
    return {
        "PR": round(PR, 2),
        "rank_90": rank_90,
        "top1_ratio": round(top1, 4),
        "top5_ratio": round(top5, 4),
    }


def ortho_regularization(B):
    """
    正交正则：惩罚B各行之间的余弦相似度（非对角元素）。
    loss = mean((B_norm @ B_norm^T - I)^2)  (off-diagonal only)
    """
    B_n  = F.normalize(B, dim=1)
    gram = torch.matmul(B_n, B_n.t())
    k    = gram.shape[0]
    off  = gram - torch.eye(k, device=gram.device)
    return (off ** 2).mean()


def compute_probe_accuracy(model, loader, tokenizer, device,
                           max_batches=30):
    """
    多义词线性探针：用token的embedding向量预测"下一个词"属于哪类。
    BIF的embedding是e=alpha@B，如果低维配方空间有效，探针准确率应更高。
    """
    word_token_ids = {}
    for word in POLYSEMOUS_WORDS:
        ids = tokenizer.encode(" " + word)
        if len(ids) == 1:
            word_token_ids[word] = ids[0]

    word_data = {w: {"embeds": [], "next_tokens": []} for w in word_token_ids}

    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            input_ids = batch.to(device)
            if hasattr(model, 'bif_embed'):
                embeds = model.bif_embed(input_ids)
            else:
                embeds = model.embed(input_ids)
            for word, tid in word_token_ids.items():
                mask = (input_ids[:, :-1] == tid)
                if mask.any():
                    word_data[word]["embeds"].append(
                        embeds[:, :-1][mask].cpu()
                    )
                    word_data[word]["next_tokens"].append(
                        input_ids[:, 1:][mask].cpu()
                    )

    valid_scores = []
    for word in word_token_ids:
        if not word_data[word]["embeds"]:
            continue
        all_embeds = torch.cat(word_data[word]["embeds"],      dim=0).numpy()
        all_next   = torch.cat(word_data[word]["next_tokens"], dim=0).numpy()
        if len(all_embeds) < 10:
            continue
        counter = Counter(all_next.tolist())
        top2    = counter.most_common(2)
        if len(top2) < 2:
            continue
        top2_tokens = [top2[0][0], top2[1][0]]
        if (top2[0][1] + top2[1][1]) / len(all_next) < 0.25:
            continue
        mask_np = np.isin(all_next, top2_tokens)
        X = all_embeds[mask_np]
        y = (all_next[mask_np] == top2_tokens[1]).astype(int)
        if len(X) < 6 or len(np.unique(y)) < 2:
            continue
        try:
            clf    = LogisticRegression(max_iter=1000, solver="lbfgs",
                                        random_state=42)
            scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
            valid_scores.append(float(scores.mean()))
        except Exception:
            pass

    model.train()
    return round(float(np.mean(valid_scores)), 4) if valid_scores else None


# ============================================================
# 单模型训练函数
# ============================================================
def train_one_model(model, model_name, train_loader, val_loader,
                    tokenizer, device, config, logger):
    is_bif      = hasattr(model, 'bif_embed') and hasattr(model, 'fam')
    is_bif_type = hasattr(model, 'bif_embed')

    # FAM的sim_W用更高学习率
    if is_bif:
        fam_params  = [model.fam.sim_W]
        other_params = [p for p in model.parameters()
                        if p is not model.fam.sim_W]
        param_groups = [
            {"params": other_params, "lr": config["lr"]},
            {"params": fam_params,   "lr": config["lr"] * config["fam_lr_mult"]},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": config["lr"]}]

    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=config["weight_decay"]
    )

    steps_list, train_loss_list, val_ppl_list = [], [], []
    lowfreq_ppl_list, b_rank_list, probe_list = [], [], []

    model.train()
    model.to(device)

    train_iter           = iter(train_loader)
    step                 = 0
    current_ortho_lambda = 0.0

    logger.info(
        f"[{model_name}] 开始训练  "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )

    while step < config["total_steps"]:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch      = next(train_iter)

        input_ids  = batch.to(device)
        current_lr = cosine_lr(
            step, config["total_steps"], config["warmup_steps"], config["lr"]
        )
        for i, pg in enumerate(optimizer.param_groups):
            pg["lr"] = current_lr if i == 0 else current_lr * config["fam_lr_mult"]

        logits  = model(input_ids)
        V       = logits.shape[-1]
        lm_loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            input_ids[:, 1:].reshape(-1)
        )

        if is_bif_type and current_ortho_lambda > 0:
            loss = lm_loss + current_ortho_lambda * ortho_regularization(
                model.bif_embed.B
            )
        else:
            loss = lm_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        step += 1

        if step % config["log_every"] == 0:
            logger.info(
                f"[{model_name}] step={step}  "
                f"loss={lm_loss.item():.4f}  lr={current_lr:.2e}"
            )

        if step % config["measure_every"] == 0:
            val_ppl      = compute_ppl(model, val_loader, device)
            lowfreq_ppl  = compute_lowfreq_ppl(
                model, val_loader, device, tokenizer
            )
            b_rank = check_B_rank(model) if is_bif_type else None

            steps_list.append(step)
            train_loss_list.append(round(lm_loss.item(), 4))
            val_ppl_list.append(round(val_ppl, 2))
            lowfreq_ppl_list.append(round(lowfreq_ppl, 2))
            if b_rank:
                b_rank_list.append({"step": step, **b_rank})

            logger.info(
                f"[{model_name}] step={step}  val_ppl={val_ppl:.2f}  "
                f"lowfreq_ppl={lowfreq_ppl:.2f}"
                + (f"  B_PR={b_rank['PR']:.1f}" if b_rank else "")
            )

            # B矩阵坍缩检测
            if (is_bif_type and b_rank
                    and b_rank["PR"] < config["ortho_reg_threshold"]
                    and current_ortho_lambda == 0.0):
                current_ortho_lambda = 0.01
                logger.info(
                    f"[{model_name}] B_PR={b_rank['PR']:.1f} < "
                    f"{config['ortho_reg_threshold']}，"
                    f"激活正交正则化 lambda={current_ortho_lambda}"
                )

        if step % config["probe_every"] == 0:
            probe_acc = compute_probe_accuracy(
                model, val_loader, tokenizer, device
            )
            probe_list.append({"step": step, "probe_accuracy": probe_acc})
            logger.info(
                f"[{model_name}] step={step}  probe_accuracy={probe_acc}"
            )

    # 最终指标
    final_ppl       = compute_ppl(model, val_loader, device)
    final_lf_ppl    = compute_lowfreq_ppl(
        model, val_loader, device, tokenizer
    )
    final_b_rank    = check_B_rank(model) if is_bif_type else None
    final_probe     = compute_probe_accuracy(
        model, val_loader, tokenizer, device
    )

    logger.info(
        f"[{model_name}] 完成  final_ppl={final_ppl:.2f}  "
        f"lowfreq_ppl={final_lf_ppl:.2f}"
        + (f"  B_PR={final_b_rank['PR']:.1f}" if final_b_rank else "")
    )

    curve = {
        "steps":           steps_list,
        "train_loss":      train_loss_list,
        "val_ppl":         val_ppl_list,
        "lowfreq_ppl":     lowfreq_ppl_list,
        "b_rank_history":  b_rank_list,
        "probe_history":   probe_list,
    }
    final = {
        "val_ppl":             round(final_ppl, 2),
        "lowfreq_ppl":         round(final_lf_ppl, 2),
        "params":              sum(p.numel() for p in model.parameters()),
        "ortho_lambda_used":   current_ortho_lambda,
        "final_B_rank":        final_b_rank,
        "final_probe_accuracy": final_probe,
    }
    return curve, final


# ============================================================
# 主函数
# ============================================================
def main():
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["log_file"]), exist_ok=True)

    logger = logging.getLogger("bif_phase1")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(CONFIG["log_file"], encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("=" * 60)
    logger.info("BIF Phase 1 — 核心假设正式验证")
    logger.info("=" * 60)

    # 锁定判定标准
    manifest = {
        "experiment":          "BIF Phase 1",
        "locked_pass_criteria": CONFIG["pass_criteria"],
        "note": "pass_criteria在实验开始后不允许修改"
    }
    with open(os.path.join(CONFIG["results_dir"], "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("判定标准已锁定")

    device = CONFIG["device"]

    logger.info("加载数据...")
    train_ids, val_ids, vocab_size, tokenizer = load_data(CONFIG)
    logger.info(
        f"vocab={vocab_size}  "
        f"train_tokens={len(train_ids)}  val_tokens={len(val_ids)}"
    )

    batch_sizes = [CONFIG["batch_size"], 16, 8]
    for bs in batch_sizes:
        try:
            train_ds = WikiTextDataset(train_ids, CONFIG["seq_len"])
            val_ds   = WikiTextDataset(val_ids,   CONFIG["seq_len"])
            train_loader = DataLoader(
                train_ds, batch_size=bs, shuffle=True,
                num_workers=2, pin_memory=True
            )
            val_loader = DataLoader(
                val_ds, batch_size=bs, shuffle=False,
                num_workers=2, pin_memory=True
            )

            k, d, nh = CONFIG["k"], CONFIG["d"], CONFIG["n_heads"]

            baseline = BaselineModel(
                vocab_size, d_model=d,
                n_layers=CONFIG["baseline_n_layers"], n_heads=nh
            )
            bif = BIFModel(
                vocab_size, k=k, d_model=d,
                n_layers=CONFIG["bif_n_layers"], n_heads=nh
            )
            ablation = BIFAblationModel(
                vocab_size, k=k, d_model=d,
                n_layers=CONFIG["ablation_n_layers"],
                n_heads=nh, compressed_dim=16
            )

            p_base = sum(p.numel() for p in baseline.parameters())
            p_bif  = sum(p.numel() for p in bif.parameters())
            p_abl  = sum(p.numel() for p in ablation.parameters())
            logger.info(
                f"参数量 — Baseline:{p_base:,}  BIF:{p_bif:,}  Ablation:{p_abl:,}"
            )
            logger.info(
                f"BIF/Baseline={p_bif/p_base:.3f}  "
                f"Ablation/Baseline={p_abl/p_base:.3f}"
            )

            training_curves = {}
            final_results   = {}

            # ── 串行训练：Baseline → Ablation → BIF ─────────────────
            logger.info("=" * 50)
            logger.info("训练 Baseline")
            curve_b, final_b = train_one_model(
                baseline, "Baseline",
                train_loader, val_loader, tokenizer, device, CONFIG, logger
            )
            training_curves["Baseline"] = curve_b
            final_results["Baseline"]   = final_b
            del baseline
            torch.cuda.empty_cache()

            logger.info("=" * 50)
            logger.info("训练 BIF-ablation（消融FAM）")
            curve_a, final_a = train_one_model(
                ablation, "BIF-ablation",
                train_loader, val_loader, tokenizer, device, CONFIG, logger
            )
            training_curves["BIF-ablation"] = curve_a
            final_results["BIF-ablation"]   = final_a
            del ablation
            torch.cuda.empty_cache()

            logger.info("=" * 50)
            logger.info("训练 BIF")
            curve_f, final_f = train_one_model(
                bif, "BIF",
                train_loader, val_loader, tokenizer, device, CONFIG, logger
            )
            training_curves["BIF"] = curve_f
            final_results["BIF"]   = final_f
            del bif
            torch.cuda.empty_cache()

            # ── Verdict ──────────────────────────────────────────────
            ppl_bif  = final_results["BIF"]["val_ppl"]
            ppl_base = final_results["Baseline"]["val_ppl"]
            ppl_abl  = final_results["BIF-ablation"]["val_ppl"]
            lf_bif   = final_results["BIF"]["lowfreq_ppl"]
            lf_base  = final_results["Baseline"]["lowfreq_ppl"]

            crit            = CONFIG["pass_criteria"]
            bif_beats_base  = (ppl_base - ppl_bif) >= crit["bif_vs_baseline_ppl_gap"]
            fam_has_value   = (ppl_abl  - ppl_bif) >= crit["bif_vs_ablation_ppl_gap"]
            lf_advantage    = round(lf_base - lf_bif, 2)

            if bif_beats_base and fam_has_value:
                rec = (
                    f"PASS: BIF beats Baseline by "
                    f"{ppl_base-ppl_bif:.1f}pt, "
                    f"FAM contributes {ppl_abl-ppl_bif:.1f}pt. Proceed to scale."
                )
            elif bif_beats_base and not fam_has_value:
                rec = (
                    f"PARTIAL: BIF beats Baseline by {ppl_base-ppl_bif:.1f}pt, "
                    f"but FAM gap={ppl_abl-ppl_bif:.1f}pt < "
                    f"{crit['bif_vs_ablation_ppl_gap']}. "
                    f"Gain from alpha-space, not FAM."
                )
            elif not bif_beats_base and fam_has_value:
                rec = (
                    f"PARTIAL: FAM beats ablation by {ppl_abl-ppl_bif:.1f}pt, "
                    f"but BIF={ppl_bif:.1f} does not beat Baseline={ppl_base:.1f}."
                )
            else:
                rec = (
                    f"FAIL: BIF={ppl_bif:.1f} does not beat "
                    f"Baseline={ppl_base:.1f}. BIF hypothesis rejected."
                )

            verdict = {
                "ppl_BIF":            ppl_bif,
                "ppl_Baseline":       ppl_base,
                "ppl_BIF_ablation":   ppl_abl,
                "bif_vs_baseline_gap": round(ppl_base - ppl_bif, 2),
                "fam_contribution_gap": round(ppl_abl - ppl_bif, 2),
                "lowfreq_ppl_BIF":    lf_bif,
                "lowfreq_ppl_Base":   lf_base,
                "lowfreq_advantage":  lf_advantage,
                "bif_beats_baseline": bif_beats_base,
                "fam_has_value":      fam_has_value,
                "params": {
                    "Baseline":     p_base,
                    "BIF":          p_bif,
                    "BIF-ablation": p_abl,
                },
                "recommendation": rec,
            }

            logger.info("=" * 60)
            logger.info("Phase 1 Verdict")
            logger.info(json.dumps(verdict, indent=2))

            print("\n" + "=" * 60)
            print("BIF Phase 1 结果汇总")
            print("=" * 60)
            print(f"  Baseline    : PPL={ppl_base:.2f}  低频词PPL={lf_base:.2f}  params={p_base:,}")
            print(f"  BIF-ablation: PPL={ppl_abl:.2f}  params={p_abl:,}")
            print(f"  BIF         : PPL={ppl_bif:.2f}  低频词PPL={lf_bif:.2f}  params={p_bif:,}")
            print()
            print(f"  BIF vs Baseline  : {ppl_base-ppl_bif:+.2f}pt  "
                  f"(需>{crit['bif_vs_baseline_ppl_gap']})  "
                  f"{'✓' if bif_beats_base else '✗'}")
            print(f"  BIF vs Ablation  : {ppl_abl-ppl_bif:+.2f}pt  "
                  f"(需>{crit['bif_vs_ablation_ppl_gap']})  "
                  f"{'✓' if fam_has_value else '✗'}")
            print(f"  低频词优势       : {lf_advantage:+.2f}pt")
            print(f"\n  结论: {rec}")
            print("=" * 60)

            output = {
                "version":        "phase1_v2",
                "config":         CONFIG,
                "verdict":        verdict,
                "training_curves": training_curves,
                "final_results":  final_results,
            }
            results_path = os.path.join(
                CONFIG["results_dir"], "results_phase1_v2.json"
            )
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)
            logger.info(f"结果已保存: {results_path}")
            break

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM with batch_size={bs}")
            torch.cuda.empty_cache()
            if bs == batch_sizes[-1]:
                logger.error("所有batch_size均OOM")
                raise


if __name__ == "__main__":
    main()
