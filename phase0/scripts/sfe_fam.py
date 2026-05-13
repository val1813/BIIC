"""
最小FAM验证实验 — SFE + Factor-Aware Mixing
回答：如果下游有一个层明确依赖α系数结构，主loss会保留g的分化还是继续压制它？

对比：
  B_FAM：SFE(动态) → FAM层 → 3层标准Transformer → LM头
  C_static：SFE(静态) → 4层标准Transformer → LM头

判定标准：
  1. alpha_cos_min < 0.65（g产生显著分化）
  2. alpha_cos_final < 0.80（FAM让分化被保留，不反弹）
  3. B_FAM PPL ≤ C PPL + 5（α分化对语言建模有帮助或至少无害）
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
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from collections import Counter

# ============ 配置 ============
CONFIG = {
    "k": 64,
    "d": 256,
    "ctx_window": 4,
    "n_layers": 4,
    "n_heads": 4,
    "seq_len": 128,
    "batch_size": 32,
    "lr": 3e-4,
    "g_lr_multiplier": 10,
    "weight_decay": 0.01,
    "warmup_steps": 200,
    "total_steps": 5000,
    "grad_clip": 1.0,
    "seed": 42,
    "log_every": 100,
    "measure_every": 200,
    "probe_every": 1000,
    "device": "cuda:0",
    "results_dir": "/data/biic/results/sfe_validation",
    "log_file": "/data/biic/logs/sfe_fam.log",
    "lambda_div": 0.01,
}

POLYSEMOUS_WORDS = [
    "book", "run", "bank", "light", "right", "fire", "deal", "lead", "match", "check",
    "clear", "fall", "field", "form", "frame", "grant", "ground", "head", "interest", "key"
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ============ 数据集 ============
class WikiTextDataset(Dataset):
    def __init__(self, tokenized_ids, seq_len):
        self.seq_len = seq_len
        self.data = tokenized_ids[: (len(tokenized_ids) // seq_len) * seq_len]
        self.data = self.data.reshape(-1, seq_len)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.long)


def load_wikitext_data():
    from transformers import AutoTokenizer
    os.environ["HTTP_PROXY"] = "http://proxy.mornai.cn:7890"
    os.environ["HTTPS_PROXY"] = "http://proxy.mornai.cn:7890"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-103-v1", trust_remote_code=True)

    train_lines = [l for l in dataset["train"]["text"] if l.strip()][:50000]
    train_ids = tokenizer.encode("\n".join(train_lines))

    val_lines = [l for l in dataset["validation"]["text"] if l.strip()][:5000]
    val_ids = tokenizer.encode("\n".join(val_lines))

    return np.array(train_ids), np.array(val_ids), tokenizer.vocab_size, tokenizer


# ============ SFE Embedding（与v1.1完全相同）============
class SFEEmbedding(nn.Module):
    def __init__(self, vocab_size, k=64, d=256, ctx_window=4, use_dynamic=True):
        super().__init__()
        self.k = k
        self.d = d
        self.ctx_window = ctx_window
        self.use_dynamic = use_dynamic

        self.B = nn.Parameter(torch.randn(k, d) * 0.02)
        self.alpha = nn.Embedding(vocab_size, k)
        nn.init.normal_(self.alpha.weight, std=0.02)

        if use_dynamic:
            self.g = nn.Linear(ctx_window * d, k, bias=True)
            nn.init.normal_(self.g.weight, std=0.001)
            nn.init.zeros_(self.g.bias)

    def forward(self, token_ids, ctx_embeds=None):
        alpha_static = self.alpha(token_ids)

        if self.use_dynamic and ctx_embeds is not None:
            B_size, L, d = ctx_embeds.shape
            pad = torch.zeros(B_size, self.ctx_window, d, device=ctx_embeds.device)
            padded = torch.cat([pad, ctx_embeds], dim=1)
            windows = padded.unfold(1, self.ctx_window, 1)
            windows = windows[:, :L, :, :]
            windows = windows.permute(0, 1, 3, 2)
            windows = windows.reshape(B_size, L, self.ctx_window * d)
            delta_alpha = self.g(windows)
            alpha_dynamic = alpha_static + delta_alpha
        else:
            alpha_dynamic = alpha_static

        return torch.matmul(alpha_dynamic, self.B)


# ============ FAM层（核心新模块）============
class FAMLayer(nn.Module):
    """
    Factor-Aware Mixing：在α系数空间做token间交互。
    相似度矩阵S完全由α决定，不走Q/K投影。
    这使得g产生的Δα直接影响token间信息流动，主loss有理由保留g的分化。
    """
    def __init__(self, k=64, d=256):
        super().__init__()
        self.k = k
        self.d = d
        # 双线性相似函数：S[i,j] = alpha_i @ W @ alpha_j^T
        self.sim_W = nn.Parameter(torch.randn(k, k) * 0.01)
        self.out_proj = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, alpha):
        """
        x:     [B, L, d] 当前token的向量表示
        alpha: [B, L, k] 当前token的α系数（静态先验，来自SFE.alpha）
        """
        B, L, _ = x.shape

        # 在α空间计算token间相似度
        alpha_W = torch.matmul(alpha, self.sim_W)      # [B, L, k]
        S = torch.matmul(alpha_W, alpha.transpose(-1, -2))  # [B, L, L]

        # 因果mask
        causal_mask = torch.triu(
            torch.full((L, L), float('-inf'), device=x.device), diagonal=1
        )
        S = S + causal_mask
        S = F.softmax(S, dim=-1)

        # 用相似度对x做加权聚合
        out = torch.matmul(S, x)              # [B, L, d]
        out = self.dropout(self.out_proj(out))

        # 残差 + LayerNorm
        return self.ln(x + out)


# ============ 标准Transformer块（与v1.1相同）============
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
        mask = torch.nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, is_causal=True)
        x = x + self.dropout(h)
        h = self.ln2(x)
        x = x + self.ffn(h)
        return x


# ============ Model B_FAM：SFE(动态) → FAM → 3层Transformer → LM头 ============
class ModelB_FAM(nn.Module):
    def __init__(self, vocab_size, k=64, d_model=256, ctx_window=4, n_layers=4, n_heads=4):
        super().__init__()
        self.sfe = SFEEmbedding(vocab_size, k=k, d=d_model,
                                ctx_window=ctx_window, use_dynamic=True)
        self.pos_embedding = nn.Embedding(512, d_model)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)

        # 第一层用FAM
        self.fam = FAMLayer(k=k, d=d_model)
        # 后续n_layers-1层用标准Transformer
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads) for _ in range(n_layers - 1)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)

        # SFE with shifted context（避免信息泄漏）
        static_embeds = self.sfe(input_ids, ctx_embeds=None).detach()
        shifted = torch.zeros_like(static_embeds)
        shifted[:, 1:, :] = static_embeds[:, :-1, :]
        x = self.sfe(input_ids, ctx_embeds=shifted)

        # 动态α（含Δα）：让FAM直接依赖g的输出，梯度直接流向g
        alpha_static = self.sfe.alpha(input_ids)  # [B, L, k]
        # 重新计算delta_alpha用于FAM（与SFE forward逻辑一致）
        pad = torch.zeros(B, self.sfe.ctx_window, self.sfe.d, device=shifted.device)
        padded = torch.cat([pad, shifted], dim=1)
        windows = padded.unfold(1, self.sfe.ctx_window, 1)[:, :L, :, :]
        windows = windows.permute(0, 1, 3, 2).reshape(B, L, self.sfe.ctx_window * self.sfe.d)
        delta_alpha = self.sfe.g(windows)
        alpha_dynamic = alpha_static + delta_alpha  # [B, L, k] — 含g的贡献

        x = x + self.pos_embedding(pos)

        # FAM第一层：用动态α做token间交互，梯度直接保护g
        x = self.fam(x, alpha_dynamic)

        # 标准Transformer后续层
        for layer in self.layers:
            x = layer(x)

        x = self.ln_f(x)
        return self.lm_head(x)

    def get_sfe_embeddings(self, input_ids):
        with torch.no_grad():
            static_embeds = self.sfe(input_ids, ctx_embeds=None)
            shifted = torch.zeros_like(static_embeds)
            shifted[:, 1:, :] = static_embeds[:, :-1, :]
            return self.sfe(input_ids, ctx_embeds=shifted)


# ============ Model C_static（对照，与v1.1完全相同）============
class ModelC_Static(nn.Module):
    """SFE embedding（静态，无g）→ 4层标准Transformer → LM头"""
    def __init__(self, vocab_size, k=64, d_model=256, ctx_window=4, n_layers=4, n_heads=4):
        super().__init__()
        self.sfe = SFEEmbedding(vocab_size, k=k, d=d_model,
                                ctx_window=ctx_window, use_dynamic=False)
        self.pos_embedding = nn.Embedding(512, d_model)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        x = self.sfe(input_ids)
        x = x + self.pos_embedding(pos)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def get_sfe_embeddings(self, input_ids):
        with torch.no_grad():
            return self.sfe(input_ids)


# ============ 工具函数 ============
def cosine_scheduler(step, total_steps, warmup_steps, lr):
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def get_word_token_ids(tokenizer):
    word_token_ids = {}
    for word in POLYSEMOUS_WORDS:
        ids = tokenizer.encode(" " + word)
        if len(ids) == 1:
            word_token_ids[word] = ids[0]
        else:
            ids2 = tokenizer.encode(word)
            if len(ids2) == 1:
                word_token_ids[word] = ids2[0]
    return word_token_ids


def compute_alpha_divergence(model, val_loader, tokenizer, device):
    model.eval()
    word_token_ids = get_word_token_ids(tokenizer)
    word_embeddings = {w: [] for w in word_token_ids}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch.to(device)
            sfe_embeds = model.get_sfe_embeddings(input_ids)
            for word, tid in word_token_ids.items():
                mask = (input_ids == tid)
                if mask.any():
                    word_embeddings[word].append(sfe_embeds[mask].cpu())

    valid_scores = []
    for word in word_token_ids:
        if not word_embeddings[word]:
            continue
        all_embeds = torch.cat(word_embeddings[word], dim=0)
        if all_embeds.shape[0] < 15:
            continue
        normed = F.normalize(all_embeds, dim=-1)
        cos_sim = torch.mm(normed, normed.t())
        n = cos_sim.shape[0]
        triu_idx = torch.triu_indices(n, n, offset=1)
        pairwise = cos_sim[triu_idx[0], triu_idx[1]]
        valid_scores.append(pairwise.median().item())

    alpha_cos_mean = float(np.mean(valid_scores)) if valid_scores else 1.0
    model.train()
    return alpha_cos_mean


def compute_probe_accuracy(model, val_loader, tokenizer, device):
    model.eval()
    word_token_ids = get_word_token_ids(tokenizer)
    word_data = {w: {"embeds": [], "next_tokens": []} for w in word_token_ids}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch.to(device)
            sfe_embeds = model.get_sfe_embeddings(input_ids)
            for word, tid in word_token_ids.items():
                mask = (input_ids[:, :-1] == tid)
                if mask.any():
                    word_data[word]["embeds"].append(sfe_embeds[:, :-1][mask].cpu())
                    word_data[word]["next_tokens"].append(input_ids[:, 1:][mask].cpu())

    valid_scores = []
    for word in word_token_ids:
        if not word_data[word]["embeds"]:
            continue
        all_embeds = torch.cat(word_data[word]["embeds"], dim=0).numpy()
        all_next = torch.cat(word_data[word]["next_tokens"], dim=0).numpy()
        if len(all_embeds) < 10:
            continue
        counter = Counter(all_next.tolist())
        top2 = counter.most_common(2)
        if len(top2) < 2:
            continue
        top2_tokens = [top2[0][0], top2[1][0]]
        if (top2[0][1] + top2[1][1]) / len(all_next) < 0.30:
            continue
        mask_np = np.isin(all_next, top2_tokens)
        X = all_embeds[mask_np]
        y = (all_next[mask_np] == top2_tokens[1]).astype(int)
        if len(X) < 6 or len(np.unique(y)) < 2:
            continue
        try:
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
            valid_scores.append(float(scores.mean()))
        except Exception:
            pass

    model.train()
    return float(np.mean(valid_scores)) if valid_scores else 0.0


def alpha_diversity_loss(sfe_embeds, input_ids, word_token_ids, lambda_div=0.01):
    loss = torch.tensor(0.0, device=input_ids.device)
    count = 0
    for tid in word_token_ids.values():
        mask = (input_ids == tid)
        if mask.sum() < 2:
            continue
        embeds = sfe_embeds[mask]
        normed = F.normalize(embeds, dim=-1)
        cos_sim = torch.mm(normed, normed.t())
        loss = loss + cos_sim.triu(diagonal=1).mean()
        count += 1
    return lambda_div * (loss / max(count, 1))


def evaluate_ppl(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch.to(device)
            logits = model(input_ids)
            V = logits.shape[-1]
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, V),
                input_ids[:, 1:].reshape(-1),
                reduction="sum"
            )
            total_loss += loss.item()
            total_tokens += input_ids[:, 1:].numel()
    ppl = math.exp(total_loss / max(total_tokens, 1))
    model.train()
    return ppl


# ============ 训练函数 ============
def train_model(model, model_name, train_loader, val_loader, tokenizer, device, config):
    logger = logging.getLogger("sfe_fam")

    has_g = hasattr(model, 'sfe') and hasattr(model.sfe, 'g')

    # g参数单独分组，10x学习率
    g_params = list(model.sfe.g.parameters()) if has_g else []
    g_param_ids = set(id(p) for p in g_params)
    other_params = [p for p in model.parameters() if id(p) not in g_param_ids]

    param_groups = [{'params': other_params, 'lr': config['lr']}]
    if g_params:
        param_groups.append({'params': g_params, 'lr': config['lr'] * config['g_lr_multiplier']})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])

    word_token_ids = get_word_token_ids(tokenizer) if has_g else {}

    model.train()
    model.to(device)

    steps_list, train_loss_list, val_ppl_list = [], [], []
    alpha_cos_list, probe_acc_list = [], []
    has_sfe = hasattr(model, 'get_sfe_embeddings')

    logger.info(f"[{model_name}] 开始训练 params={sum(p.numel() for p in model.parameters()):,}")
    train_iter = iter(train_loader)
    step = 0

    while step < config["total_steps"]:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch.to(device)
        current_lr = cosine_scheduler(step, config["total_steps"], config["warmup_steps"], config["lr"])

        # 更新学习率（g用10x）
        for i, pg in enumerate(optimizer.param_groups):
            if i == 0:
                pg["lr"] = current_lr
            else:
                pg["lr"] = current_lr * config["g_lr_multiplier"]

        logits = model(input_ids)
        V = logits.shape[-1]
        main_loss = F.cross_entropy(logits[:, :-1].reshape(-1, V), input_ids[:, 1:].reshape(-1))

        # 辅助分化损失（只对有g的模型）
        if has_g and word_token_ids:
            sfe_embeds = model.get_sfe_embeddings(input_ids)
            div_loss = alpha_diversity_loss(sfe_embeds, input_ids, word_token_ids, config['lambda_div'])
            loss = main_loss + div_loss
        else:
            loss = main_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        optimizer.step()
        step += 1

        if step % config["log_every"] == 0:
            logger.info(f"[{model_name}] step={step}, loss={loss.item():.4f}, lr={current_lr:.6f}")

        if step % config["measure_every"] == 0:
            ppl = evaluate_ppl(model, val_loader, device)
            steps_list.append(step)
            train_loss_list.append(round(loss.item(), 4))
            val_ppl_list.append(round(ppl, 2))
            logger.info(f"[{model_name}] step={step}, val_ppl={ppl:.2f}")

            if has_sfe:
                alpha_mean = compute_alpha_divergence(model, val_loader, tokenizer, device)
                alpha_cos_list.append(round(alpha_mean, 4))
                logger.info(f"[{model_name}] step={step}, alpha_cos_mean={alpha_mean:.4f}")

        if step % config["probe_every"] == 0 and has_sfe:
            probe_mean = compute_probe_accuracy(model, val_loader, tokenizer, device)
            probe_acc_list.append(round(probe_mean, 4))
            logger.info(f"[{model_name}] step={step}, probe_accuracy={probe_mean:.4f}")

    final_ppl = evaluate_ppl(model, val_loader, device)
    logger.info(f"[{model_name}] 完成, final_ppl={final_ppl:.2f}")

    curve = {
        "steps": steps_list,
        "train_loss": train_loss_list,
        "val_ppl": val_ppl_list,
    }
    if has_sfe and alpha_cos_list:
        curve["alpha_cos_mean"] = alpha_cos_list
        curve["alpha_cos_min"] = min(alpha_cos_list)
        curve["alpha_cos_min_step"] = steps_list[alpha_cos_list.index(min(alpha_cos_list))]
    if probe_acc_list:
        curve["probe_accuracy"] = probe_acc_list

    final_info = {
        "val_ppl": round(final_ppl, 2),
        "params": sum(p.numel() for p in model.parameters()),
    }
    if has_sfe and alpha_cos_list:
        final_info["alpha_cos_mean_final"] = alpha_cos_list[-1]
        final_info["alpha_cos_min"] = min(alpha_cos_list)
    if probe_acc_list:
        final_info["probe_accuracy_final"] = probe_acc_list[-1]

    return curve, final_info


# ============ 主函数 ============
def main():
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["log_file"]), exist_ok=True)

    logger = logging.getLogger("sfe_fam")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(CONFIG["log_file"], encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(ch)

    device = CONFIG["device"]
    logger.info("SFE FAM验证实验")
    logger.info(f"设备: {device}")

    logger.info("加载WikiText-103...")
    train_ids, val_ids, vocab_size, tokenizer = load_wikitext_data()
    logger.info(f"vocab={vocab_size}, train_tokens={len(train_ids)}, val_tokens={len(val_ids)}")

    batch_sizes = [CONFIG["batch_size"], 16, 8]
    for bs in batch_sizes:
        try:
            logger.info(f"batch_size={bs}")
            train_ds = WikiTextDataset(train_ids, CONFIG["seq_len"])
            val_ds = WikiTextDataset(val_ids, CONFIG["seq_len"])
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

            d, k, nl, nh = CONFIG["d"], CONFIG["k"], CONFIG["n_layers"], CONFIG["n_heads"]
            training_curves = {}
            final_results = {}

            # Model C（对照，先跑）
            logger.info("=" * 50 + "\n训练 Model C (SFE Static, 4层Transformer)")
            model_c = ModelC_Static(vocab_size, k=k, d_model=d,
                                    ctx_window=CONFIG["ctx_window"], n_layers=nl, n_heads=nh)
            logger.info(f"Model C params: {sum(p.numel() for p in model_c.parameters()):,}")
            curve_c, final_c = train_model(
                model_c, "ModelC_Static", train_loader, val_loader, tokenizer, device, CONFIG
            )
            training_curves["C_static"] = curve_c
            final_results["C_static"] = final_c
            del model_c
            torch.cuda.empty_cache()

            # Model B_FAM（核心验证，后跑）
            logger.info("=" * 50 + "\n训练 Model B_FAM (SFE动态 + FAM第一层 + 3层Transformer)")
            model_b = ModelB_FAM(vocab_size, k=k, d_model=d,
                                 ctx_window=CONFIG["ctx_window"], n_layers=nl, n_heads=nh)
            logger.info(f"Model B_FAM params: {sum(p.numel() for p in model_b.parameters()):,}")
            curve_b, final_b = train_model(
                model_b, "ModelB_FAM", train_loader, val_loader, tokenizer, device, CONFIG
            )
            training_curves["B_fam"] = curve_b
            final_results["B_fam"] = final_b
            del model_b
            torch.cuda.empty_cache()

            # ============ Verdict ============
            ppl_b = final_results["B_fam"]["val_ppl"]
            ppl_c = final_results["C_static"]["val_ppl"]
            alpha_min = final_results["B_fam"].get("alpha_cos_min")
            alpha_final = final_results["B_fam"].get("alpha_cos_mean_final")
            probe_b = final_results["B_fam"].get("probe_accuracy_final")
            probe_c = final_results["C_static"].get("probe_accuracy_final")

            cond1 = alpha_min is not None and alpha_min < 0.65
            cond2 = alpha_final is not None and alpha_final < 0.80
            cond3 = ppl_b <= ppl_c + 5.0

            if cond1 and cond2 and cond3:
                rec = "symbiotic architecture validated: alpha divergence preserved and useful"
            elif cond1 and cond2 and not cond3:
                rec = "FAM preserves divergence but divergence not useful for LM"
            elif cond1 and not cond2:
                rec = "suppression persists even with symbiotic layer: SFE direction terminated"
            else:
                rec = "g(ctx) fails to produce meaningful divergence even with FAM"

            verdict = {
                "cond1_alpha_min_lt_065": cond1,
                "cond2_alpha_final_lt_080": cond2,
                "cond3_ppl_competitive": cond3,
                "alpha_cos_min": alpha_min,
                "alpha_cos_final": alpha_final,
                "ppl_B_FAM": ppl_b,
                "ppl_C_static": ppl_c,
                "ppl_diff": round(ppl_b - ppl_c, 2),
                "probe_B_FAM": probe_b,
                "probe_C_static": probe_c,
                "recommendation": rec,
            }

            output = {
                "version": "fam_v1",
                "description": "SFE(dynamic) + FAM first layer vs SFE(static) + standard Transformer",
                "training_curves": training_curves,
                "final": final_results,
                "verdict": verdict,
            }

            results_path = os.path.join(CONFIG["results_dir"], "results_fam.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            logger.info(f"结果已保存: {results_path}")
            logger.info(f"Verdict:\n{json.dumps(verdict, indent=2)}")
            print(f"\n{'='*60}\n实验完成\n{results_path}\nVerdict: {rec}\n{'='*60}")
            break

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM with batch_size={bs}")
            torch.cuda.empty_cache()
            if bs == batch_sizes[-1]:
                logger.error("所有batch_size都OOM")
                raise


if __name__ == "__main__":
    main()
