"""
BIF Phase 0 — 代码与梯度健全性检查
目标：500步内确认代码无结构性bug，梯度流正常，配方空间未坍缩。
通过后进入Phase 1。

运行方式：
  cd /data/biic
  CUDA_VISIBLE_DEVICES=1 python -u experiments/bif_phase0.py

预计时间：25分钟（4090单卡）
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
from torch.utils.data import Dataset, DataLoader

# ============ 配置 ============
CONFIG = {
    # 模型结构
    "k": 64,             # 配方维度（语义零件库的基数量）
    "d": 256,            # 向量维度
    "n_layers": 4,       # Transformer层数
    "n_heads": 4,        # 注意力头数
    "ctx_window": 4,     # SFE上下文窗口（Phase 0暂不使用动态α，保留接口）

    # 训练
    "seq_len": 128,
    "batch_size": 16,    # Phase 0用小batch，快速迭代
    "lr": 3e-4,
    "weight_decay": 0.01,
    "warmup_steps": 50,
    "total_steps": 600,  # Phase 0跑600步
    "grad_clip": 1.0,
    "seed": 42,
    "fam_lr_mult": 5.0,  # FAM sim_W独立学习率倍率

    # 监控
    "log_every": 10,
    "check_every": 50,   # 每50步做一次健康检查

    # 路径
    "device": "cuda:0",
    "results_dir": "/data/biic/results/bif_phase0",
    "log_file": "/data/biic/logs/bif_phase0.log",

    # Phase 0通过标准（写死，不允许实验中修改）
    "pass_criteria": {
        "loss_at_600_lt": 6.0,           # 600步时loss < 6.0
        "grad_norm_alpha_min": 1e-4,     # α梯度范数下限
        "grad_norm_alpha_max": 10.0,     # α梯度范数上限
        "grad_norm_B_min": 1e-4,         # B梯度范数下限
        "grad_norm_B_max": 10.0,         # B梯度范数上限
        "grad_norm_FAM_min": 1e-4,       # FAM梯度范数下限
        "grad_norm_FAM_max": 10.0,       # FAM梯度范数上限
        "alpha_cos_distance_min": 0.01,  # α token间余弦距离中位数下限（防坍缩）
        "bif_flops_ratio_max": 1.05,     # BIF FLOPs不超过Baseline的105%
    }
}

POLYSEMOUS_WORDS = [
    "book", "run", "bank", "light", "right", "fire", "deal",
    "lead", "match", "check", "clear", "fall", "field", "form"
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============ 数据集 ============
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

    os.environ["HTTP_PROXY"] = "http://proxy.mornai.cn:7890"
    os.environ["HTTPS_PROXY"] = "http://proxy.mornai.cn:7890"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-103-v1", trust_remote_code=True)

    # Phase 0用足够数据确保loss能降到健康水平
    train_lines = [l for l in dataset["train"]["text"] if l.strip()][:50000]
    val_lines = [l for l in dataset["validation"]["text"] if l.strip()][:2000]

    train_ids = np.array(tokenizer.encode("\n".join(train_lines)))
    val_ids = np.array(tokenizer.encode("\n".join(val_lines)))

    return train_ids, val_ids, tokenizer.vocab_size, tokenizer


# ============ BIF Embedding ============
class BIFEmbedding(nn.Module):
    """
    用α配方 + 共享零件库B替代标准embedding查表。
    e_i = alpha_i @ B，其中alpha_i是64维配方系数，B是64×256的共享基矩阵。
    """
    def __init__(self, vocab_size, k=64, d=256):
        super().__init__()
        self.k = k
        self.d = d
        # 共享语义零件库：k个基向量，每个d维
        self.B = nn.Parameter(torch.randn(k, d) * 0.02)
        # 每个token的配方系数：V个token，每个k维
        self.alpha = nn.Embedding(vocab_size, k)
        nn.init.normal_(self.alpha.weight, std=0.02)

    def forward(self, token_ids):
        # [B, L, k] @ [k, d] = [B, L, d]
        alpha = self.alpha(token_ids)
        return torch.matmul(alpha, self.B)

    def get_alpha(self, token_ids):
        return self.alpha(token_ids)


# ============ FAM层（Factor-Aware Mixing）============
class FAMLayer(nn.Module):
    """
    在α配方空间做token间交互，替代第一层自注意力。

    相似度矩阵S完全由α决定：S[i,j] = alpha_i @ W @ alpha_j^T
    这意味着token交互基于"配方相似性"而非高维向量点积。

    计算复杂度：O(n²k)，k=64 << d=256，比标准attention O(n²d)更低。
    参数量：k×k = 4096，比标准attention的4×d×d = 262144少得多。
    """
    def __init__(self, k=64, d=256, dropout=0.1):
        super().__init__()
        self.k = k
        self.d = d
        # 双线性相似核：k×k = 4096个参数
        self.sim_W = nn.Parameter(torch.randn(k, k) * 0.01)
        self.out_proj = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, alpha):
        """
        x:     [B, L, d] token向量表示
        alpha: [B, L, k] token的α配方系数
        """
        B, L, _ = x.shape

        # 在α空间计算token间相似度
        # alpha_W: [B, L, k]
        alpha_W = torch.matmul(alpha, self.sim_W)
        # S: [B, L, L]
        S = torch.matmul(alpha_W, alpha.transpose(-1, -2))

        # 因果mask（自回归）
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=x.device), diagonal=1
        )
        S = F.softmax(S + causal_mask, dim=-1)

        # 按相似度聚合token表示
        out = self.dropout(self.out_proj(torch.matmul(S, x)))
        return self.ln(x + out)


# ============ 标准Transformer块 ============
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        L = x.shape[1]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + self.dropout(h)
        return x + self.ffn(self.ln2(x))


# ============ 三个模型 ============
class BIFModel(nn.Module):
    """
    BIF = α配方embedding + FAM第一层 + (n_layers-1)层标准Transformer
    这是核心假设。
    """
    def __init__(self, vocab_size, k=64, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.bif_embed = BIFEmbedding(vocab_size, k=k, d=d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

        self.fam = FAMLayer(k=k, d=d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers - 1)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.bif_embed(input_ids)
        alpha = self.bif_embed.get_alpha(input_ids)  # [B, L, k]

        x = x + self.pos_embed(pos)
        x = self.fam(x, alpha)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_f(x))


class BIFAblationModel(nn.Module):
    """
    BIF-ablation = α配方embedding + 压缩attention第一层 + (n_layers-1)层标准Transformer
    消融FAM，保留α配方空间，参数量与BIF对齐。
    压缩attention：QK投影256→32，使参数量与FAM的k×k≈同量级。
    """
    def __init__(self, vocab_size, k=64, d_model=256, n_layers=4, n_heads=4,
                 compressed_dim=32):
        super().__init__()
        self.bif_embed = BIFEmbedding(vocab_size, k=k, d=d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

        # 压缩版attention：QK在低维空间做，V和输出保持d_model
        # 参数量：d×c + d×c + d×d + d×d ≈ 2×256×32 + 2×256×256 ≈ 147K
        # FAM参数量：k×k + d×d = 4096 + 65536 ≈ 70K
        # 可进一步压缩compressed_dim，但保持量级可比即可
        self.compressed_dim = compressed_dim
        self.Wq = nn.Linear(d_model, compressed_dim, bias=False)
        self.Wk = nn.Linear(d_model, compressed_dim, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln_attn = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(0.1)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers - 1)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)

        x = self.bif_embed(input_ids)
        x = x + self.pos_embed(pos)

        # 压缩attention第一层
        h = self.ln1(x)
        Q = self.Wq(h)  # [B, L, c]
        K = self.Wk(h)  # [B, L, c]
        V = self.Wv(h)  # [B, L, d]

        scale = math.sqrt(self.compressed_dim)
        S = torch.matmul(Q, K.transpose(-1, -2)) / scale  # [B, L, L]
        mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        S = F.softmax(S + mask, dim=-1)
        out = self.dropout1(self.Wo(torch.matmul(S, V)))
        x = self.ln_attn(x + out)

        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_f(x))


class BaselineModel(nn.Module):
    """
    标准Transformer baseline：标准embedding + 全部标准attention层。
    参数量通过增大d_model或层数来对齐BIF。
    """
    def __init__(self, vocab_size, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.pos_embed = nn.Embedding(512, d_model)
        nn.init.normal_(self.pos_embed.weight, std=0.02)

        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # weight tying：embedding和lm_head共享参数，这是Baseline的标准做法
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.embed(input_ids) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_f(x))


# ============ FLOPs估算 ============
def estimate_flops(model, seq_len, batch_size=1, device='cpu'):
    """
    用一个dummy batch估算前向传播的FLOPs。
    优先使用fvcore，不可用时返回None（FLOPs项改为INFO，不阻断）。
    """
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
        flops = FlopCountAnalysis(model, dummy)
        return flops.total()
    except ImportError:
        return None


# ============ 健康检查函数 ============
def check_gradient_health(model, model_type):
    """检查各关键模块的梯度范数。返回dict。"""
    result = {}

    if model_type == "bif":
        # α系数梯度
        if model.bif_embed.alpha.weight.grad is not None:
            result["grad_norm_alpha"] = model.bif_embed.alpha.weight.grad.norm().item()
        else:
            result["grad_norm_alpha"] = 0.0

        # B矩阵梯度
        if model.bif_embed.B.grad is not None:
            result["grad_norm_B"] = model.bif_embed.B.grad.norm().item()
        else:
            result["grad_norm_B"] = 0.0

        # FAM双线性核梯度
        if model.fam.sim_W.grad is not None:
            result["grad_norm_FAM_W"] = model.fam.sim_W.grad.norm().item()
        else:
            result["grad_norm_FAM_W"] = 0.0

    return result


def check_alpha_diversity(model, val_loader, tokenizer, device, n_batches=5):
    """
    检查α系数的token间余弦距离中位数。
    如果所有token的α都趋近于同一向量，说明配方空间坍缩。
    """
    model.eval()
    word_token_ids = {}
    for word in POLYSEMOUS_WORDS:
        ids = tokenizer.encode(" " + word)
        if len(ids) == 1:
            word_token_ids[word] = ids[0]

    all_alphas = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= n_batches:
                break
            input_ids = batch.to(device)
            if hasattr(model, 'bif_embed'):
                alpha = model.bif_embed.get_alpha(input_ids)  # [B, L, k]
                # 随机取一些token的alpha
                B_, L_, k_ = alpha.shape
                flat_alpha = alpha.reshape(-1, k_)
                # 随机采样最多200个
                idx = torch.randperm(flat_alpha.shape[0])[:200]
                all_alphas.append(flat_alpha[idx].cpu())

    if not all_alphas:
        return None

    all_alphas = torch.cat(all_alphas, dim=0)  # [N, k]
    normed = F.normalize(all_alphas, dim=-1)
    cos_sim = torch.mm(normed, normed.t())
    n = cos_sim.shape[0]
    triu_idx = torch.triu_indices(n, n, offset=1)
    pairwise = cos_sim[triu_idx[0], triu_idx[1]]
    # 余弦相似度的中位数，转化为余弦距离（1-cos）
    cos_dist_median = (1 - pairwise).median().item()

    model.train()
    return cos_dist_median


def check_B_rank(model):
    """
    检查共享零件库B的奇异值谱，评估有效秩。
    如果有效秩接近1，说明64个基向量实际上只有少数几个在工作。
    """
    if not hasattr(model, 'bif_embed'):
        return None

    with torch.no_grad():
        B_matrix = model.bif_embed.B.detach().cpu()  # [k, d]
        U, S, Vt = torch.linalg.svd(B_matrix, full_matrices=False)

        s2 = S ** 2
        total = s2.sum().item()
        cumvar = torch.cumsum(s2, dim=0) / total

        rank_90 = (cumvar < 0.90).sum().item() + 1
        top1_ratio = (s2[0] / total).item()
        top5_ratio = (s2[:5].sum() / total).item()

        # Participation Ratio（有效秩的稳健估计）
        PR = (s2.sum() ** 2 / (s2 ** 2).sum()).item()

    return {
        "rank_90": rank_90,
        "top1_ratio": round(top1_ratio, 4),
        "top5_ratio": round(top5_ratio, 4),
        "PR": round(PR, 2),
    }


# ============ 训练工具 ============
def cosine_scheduler(step, total_steps, warmup_steps, lr):
    """Phase 0用constant lr with warmup，不做cosine衰减。
    保持有效学习率让模型在有限步数内充分学习。"""
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    return lr  # constant after warmup


def compute_ppl(model, loader, device, max_batches=50):
    model.eval()
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            ids = batch.to(device)
            logits = model(ids)
            V = logits.shape[-1]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                ids[:, 1:].reshape(-1),
                reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += ids[:, 1:].numel()
    model.train()
    return math.exp(total_loss / max(total_tokens, 1))


# ============ Phase 0主函数 ============
def run_phase0(config, logger):
    device = config["device"]
    set_seed(config["seed"])

    # 数据
    logger.info("加载数据...")
    train_ids, val_ids, vocab_size, tokenizer = load_data(config)
    train_ds = WikiTextDataset(train_ids, config["seq_len"])
    val_ds = WikiTextDataset(val_ids, config["seq_len"])
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"],
                            shuffle=False, num_workers=2, pin_memory=True)
    logger.info(f"vocab={vocab_size}, train_tokens={len(train_ids)}, val_tokens={len(val_ids)}")

    # 初始化三个模型
    k = config["k"]
    d = config["d"]
    nl = config["n_layers"]
    nh = config["n_heads"]

    bif_model = BIFModel(vocab_size, k=k, d_model=d, n_layers=nl, n_heads=nh).to(device)
    ablation_model = BIFAblationModel(vocab_size, k=k, d_model=d, n_layers=nl, n_heads=nh).to(device)
    baseline_model = BaselineModel(vocab_size, d_model=d, n_layers=nl, n_heads=nh).to(device)

    # 参数量统计
    params = {
        "BIF": sum(p.numel() for p in bif_model.parameters()),
        "BIF-ablation": sum(p.numel() for p in ablation_model.parameters()),
        "Baseline": sum(p.numel() for p in baseline_model.parameters()),
    }
    for name, p in params.items():
        logger.info(f"{name} params: {p:,}")

    # 注意：Phase 0不要求参数量对齐，只验证代码健康性
    # Phase 1才做精确对齐

    # FLOPs估算（BIF vs Baseline）
    logger.info("估算FLOPs...")
    bif_flops = estimate_flops(bif_model, config["seq_len"], device=device)
    baseline_flops = estimate_flops(baseline_model, config["seq_len"], device=device)

    if bif_flops is not None and baseline_flops is not None:
        flops_ratio = bif_flops / max(baseline_flops, 1)
        flops_available = True
        logger.info(f"BIF FLOPs: {bif_flops:,}")
        logger.info(f"Baseline FLOPs: {baseline_flops:,}")
        logger.info(f"BIF/Baseline FLOPs ratio: {flops_ratio:.3f}")
    else:
        flops_ratio = None
        flops_available = False
        param_ratio = params["BIF"] / max(params["Baseline"], 1)
        logger.info(f"fvcore不可用，FLOPs无法精确测量。")
        logger.info(f"BIF/Baseline参数量比: {param_ratio:.4f}（仅作INFO记录，不阻断）")
        logger.info(f"Phase 1需安装fvcore后精确测量")

    # Phase 0只训练BIF，Baseline作为参照
    # ============ 训练BIF ============
    logger.info("=" * 50)
    logger.info(f"Phase 0: 训练BIF（{config['total_steps']}步）")

    # FAM sim_W用独立学习率（5x），其余参数用默认lr
    fam_params = [bif_model.fam.sim_W]
    other_params = [p for n, p in bif_model.named_parameters()
                    if p is not bif_model.fam.sim_W]
    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": config["lr"]},
        {"params": fam_params, "lr": config["lr"] * config["fam_lr_mult"]},
    ], weight_decay=config["weight_decay"])

    bif_model.train()
    train_iter = iter(train_loader)
    step = 0

    # 记录检查结果
    checks = []
    pass_flags = {
        "loss_ok": False,
        "grad_alpha_ok": False,
        "grad_B_ok": False,
        "grad_FAM_ok": False,
        "alpha_diversity_ok": False,
        "flops_ok": True,  # 默认True，仅在fvcore可用时才检查
    }
    if flops_available:
        pass_flags["flops_ok"] = flops_ratio <= config["pass_criteria"]["bif_flops_ratio_max"]

    while step < config["total_steps"]:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch.to(device)
        current_lr = cosine_scheduler(step, config["total_steps"],
                                      config["warmup_steps"], config["lr"])
        for i, pg in enumerate(optimizer.param_groups):
            if i == 0:
                pg["lr"] = current_lr
            else:
                # FAM sim_W独立学习率
                pg["lr"] = current_lr * config["fam_lr_mult"]

        logits = bif_model(input_ids)
        V = logits.shape[-1]
        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, V),
            input_ids[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bif_model.parameters(), config["grad_clip"])

        # 每步都记录梯度（Phase 0要求密集监控）
        grad_info = check_gradient_health(bif_model, "bif")

        optimizer.step()
        step += 1

        if step % config["log_every"] == 0:
            logger.info(
                f"step={step}, loss={loss.item():.4f}, lr={current_lr:.6f}, "
                f"grad_alpha={grad_info.get('grad_norm_alpha', 0):.2e}, "
                f"grad_B={grad_info.get('grad_norm_B', 0):.2e}, "
                f"grad_FAM={grad_info.get('grad_norm_FAM_W', 0):.2e}"
            )

        if step % config["check_every"] == 0:
            # 健康检查
            alpha_div = check_alpha_diversity(bif_model, val_loader, tokenizer, device)
            B_rank = check_B_rank(bif_model)
            ppl = compute_ppl(bif_model, val_loader, device)

            check_result = {
                "step": step,
                "loss": round(loss.item(), 4),
                "val_ppl": round(ppl, 2),
                "grad_norm_alpha": grad_info.get("grad_norm_alpha"),
                "grad_norm_B": grad_info.get("grad_norm_B"),
                "grad_norm_FAM_W": grad_info.get("grad_norm_FAM_W"),
                "alpha_cos_distance_median": round(alpha_div, 6) if alpha_div else None,
                "B_rank": B_rank,
            }
            checks.append(check_result)
            logger.info(f"CHECK@{step}: {json.dumps(check_result)}")

    # ============ 最终判定 ============
    logger.info("=" * 50)
    logger.info("Phase 0 最终判定")

    final_loss = loss.item()
    final_grad = check_gradient_health(bif_model, "bif")
    final_alpha_div = check_alpha_diversity(bif_model, val_loader, tokenizer, device)

    crit = config["pass_criteria"]

    pass_flags["loss_ok"] = final_loss < crit["loss_at_600_lt"]
    pass_flags["grad_alpha_ok"] = (
        crit["grad_norm_alpha_min"] <= final_grad.get("grad_norm_alpha", 0) <= crit["grad_norm_alpha_max"]
    )
    pass_flags["grad_B_ok"] = (
        crit["grad_norm_B_min"] <= final_grad.get("grad_norm_B", 0) <= crit["grad_norm_B_max"]
    )
    pass_flags["grad_FAM_ok"] = (
        crit["grad_norm_FAM_min"] <= final_grad.get("grad_norm_FAM_W", 0) <= crit["grad_norm_FAM_max"]
    )
    pass_flags["alpha_diversity_ok"] = (
        final_alpha_div is not None and
        final_alpha_div >= crit["alpha_cos_distance_min"]
    )

    all_pass = all(pass_flags.values())

    verdict = {
        "phase": "0",
        "all_pass": all_pass,
        "pass_flags": pass_flags,
        "final_loss": round(final_loss, 4),
        "final_alpha_cos_distance": round(final_alpha_div, 6) if final_alpha_div else None,
        "final_grad": {k: round(v, 6) for k, v in final_grad.items()},
        "flops_ratio": round(flops_ratio, 4) if flops_ratio is not None else "N/A (fvcore not installed)",
        "flops_info": f"BIF/Baseline参数量比={params['BIF']/max(params['Baseline'],1):.4f}" if not flops_available else None,
        "params": params,
        "recommendation": (
            "PASS: proceed to Phase 1" if all_pass
            else f"FAIL: check {[k for k, v in pass_flags.items() if not v]}"
        )
    }

    logger.info(f"Verdict: {json.dumps(verdict, indent=2)}")
    print("\n" + "="*60)
    print("BIF Phase 0 结果")
    print("="*60)
    for flag, val in pass_flags.items():
        status = "✓" if val else "✗"
        print(f"  {status} {flag}")
    print(f"\n  {'PASS → 进入Phase 1' if all_pass else 'FAIL → 检查日志，修复后重跑'}")
    print("="*60)

    return verdict, checks


# ============ 主入口 ============
def main():
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["log_file"]), exist_ok=True)

    logger = logging.getLogger("bif_phase0")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(CONFIG["log_file"], encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info("BIF Phase 0 — 代码与梯度健全性检查")
    logger.info(f"设备: {CONFIG['device']}")
    logger.info(f"配置: {json.dumps({k: v for k, v in CONFIG.items() if k != 'pass_criteria'}, indent=2)}")
    logger.info(f"通过标准: {json.dumps(CONFIG['pass_criteria'], indent=2)}")

    # Phase 0实验不允许中途修改通过标准
    manifest = {
        "experiment": "BIF Phase 0",
        "locked_pass_criteria": CONFIG["pass_criteria"],
        "locked_at": "experiment start",
        "note": "pass_criteria在实验开始后不允许修改"
    }
    manifest_path = os.path.join(CONFIG["results_dir"], "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(f"实验宪法已锁定: {manifest_path}")

    try:
        verdict, checks = run_phase0(CONFIG, logger)
    except torch.cuda.OutOfMemoryError:
        logger.error("OOM: 尝试将batch_size从16降到8")
        CONFIG["batch_size"] = 8
        verdict, checks = run_phase0(CONFIG, logger)

    # 保存结果
    output = {
        "verdict": verdict,
        "checks": checks,
        "config": CONFIG,
    }
    results_path = os.path.join(CONFIG["results_dir"], "results_phase0.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"结果已保存: {results_path}")

    # Phase 0通过后的下一步提示
    if verdict["all_pass"]:
        logger.info("")
        logger.info("Phase 0通过。Phase 1参数量对齐方案：")
        logger.info("  BIF embedding省下的参数≈9.7M，补到Transformer层")
        logger.info("  三组模型目标参数量：约10M（通过调整n_layers或hidden_dim）")
        logger.info("  建议：BIF用4层（1FAM+3TF），Baseline用5层标准TF，总参数对齐到10M±0.5M")
        logger.info("  Phase 1脚本：bif_phase1.py（待生成）")


if __name__ == "__main__":
    main()
