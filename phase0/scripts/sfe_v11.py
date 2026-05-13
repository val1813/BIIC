"""
微共生验证实验 v1.1 — g(ctx)修复版
只跑B vs C，给g合理训练条件：小随机初始化 + 10x学习率 + 5000步
"""

import os
import sys
import json
import math
import time
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
    "log_file": "/data/biic/logs/sfe_v11.log",
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
    """加载WikiText-103，返回训练集前50000句和验证集前5000句的token ids"""
    from transformers import AutoTokenizer

    # VPS需要代理访问HuggingFace
    os.environ["HTTP_PROXY"] = "http://proxy.mornai.cn:7890"
    os.environ["HTTPS_PROXY"] = "http://proxy.mornai.cn:7890"

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("wikitext", "wikitext-103-v1", trust_remote_code=True)

    train_lines = [l for l in dataset["train"]["text"] if l.strip()][:50000]
    train_text = "\n".join(train_lines)
    train_ids = tokenizer.encode(train_text)

    val_lines = [l for l in dataset["validation"]["text"] if l.strip()][:5000]
    val_text = "\n".join(val_lines)
    val_ids = tokenizer.encode(val_text)

    return np.array(train_ids), np.array(val_ids), tokenizer.vocab_size, tokenizer


# ============ SFE Embedding ============
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
            # v1.1改动1：小随机初始化，不再全零
            nn.init.normal_(self.g.weight, std=0.001)
            nn.init.zeros_(self.g.bias)

    def forward(self, token_ids, ctx_embeds=None):
        alpha_static = self.alpha(token_ids)

        if self.use_dynamic and ctx_embeds is not None:
            B_size, L, d = ctx_embeds.shape
            pad = torch.zeros(B_size, self.ctx_window, d, device=ctx_embeds.device)
            padded = torch.cat([pad, ctx_embeds], dim=1)  # [B, ctx_window+L, d]
            windows = padded.unfold(1, self.ctx_window, 1)  # [B, L+1, d, ctx_window]
            windows = windows[:, :L, :, :]  # 裁掉多余窗口 → [B, L, d, ctx_window]
            windows = windows.permute(0, 1, 3, 2)  # [B, L, ctx_window, d]
            windows = windows.reshape(B_size, L, self.ctx_window * d)
            delta_alpha = self.g(windows)
            alpha_dynamic = alpha_static + delta_alpha
        else:
            alpha_dynamic = alpha_static

        return torch.matmul(alpha_dynamic, self.B)


# ============ Transformer ============
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


class ModelB_SFE(nn.Module):
    """SFE embedding (动态) → Transformer → LM头"""
    def __init__(self, vocab_size, k=64, d_model=256, ctx_window=4, n_layers=4, n_heads=4):
        super().__init__()
        self.sfe = SFEEmbedding(vocab_size, k=k, d=d_model, ctx_window=ctx_window, use_dynamic=True)
        self.pos_embedding = nn.Embedding(512, d_model)
        nn.init.normal_(self.pos_embedding.weight, std=0.02)
        self.layers = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids):
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).unsqueeze(0)
        with torch.no_grad():
            static_embeds = self.sfe(input_ids, ctx_embeds=None)
        shifted = torch.zeros_like(static_embeds)
        shifted[:, 1:, :] = static_embeds[:, :-1, :].detach()
        x = self.sfe(input_ids, ctx_embeds=shifted)
        x = x + self.pos_embedding(pos)
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


class ModelC_SFEAblation(nn.Module):
    """SFE embedding (退化版，g关闭) → Transformer → LM头"""
    def __init__(self, vocab_size, k=64, d_model=256, ctx_window=4, n_layers=4, n_heads=4):
        super().__init__()
        self.sfe = SFEEmbedding(vocab_size, k=k, d=d_model, ctx_window=ctx_window, use_dynamic=False)
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


# ============ Scheduler ============
def cosine_scheduler(step, total_steps, warmup_steps, lr):
    if step < warmup_steps:
        return lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ============ 测量函数 ============
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

    results = {}
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
        med = pairwise.median().item()
        results[word] = med
        valid_scores.append(med)

    alpha_cos_mean = float(np.mean(valid_scores)) if valid_scores else 1.0
    model.train()
    return results, alpha_cos_mean


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

    results = {}
    valid_scores = []
    for word in word_token_ids:
        if not word_data[word]["embeds"]:
            results[word] = None
            continue
        all_embeds = torch.cat(word_data[word]["embeds"], dim=0).numpy()
        all_next = torch.cat(word_data[word]["next_tokens"], dim=0).numpy()
        if len(all_embeds) < 10:
            results[word] = None
            continue

        counter = Counter(all_next.tolist())
        top2 = counter.most_common(2)
        if len(top2) < 2:
            results[word] = None
            continue

        top2_tokens = [top2[0][0], top2[1][0]]
        top2_count = top2[0][1] + top2[1][1]
        if top2_count / len(all_next) < 0.30:
            results[word] = None
            continue

        mask = np.isin(all_next, top2_tokens)
        X = all_embeds[mask]
        y = (all_next[mask] == top2_tokens[1]).astype(int)
        if len(X) < 6 or len(np.unique(y)) < 2:
            results[word] = None
            continue

        try:
            clf = LogisticRegression(max_iter=1000, solver="lbfgs")
            scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
            acc = float(scores.mean())
            results[word] = acc
            valid_scores.append(acc)
        except Exception:
            results[word] = None

    probe_mean = float(np.mean(valid_scores)) if valid_scores else 0.0
    model.train()
    return results, probe_mean


def alpha_diversity_loss(sfe_embeds, input_ids, word_token_ids, lambda_div=0.01):
    """辅助损失：鼓励同一token在不同位置产生不同表示"""
    loss = torch.tensor(0.0, device=input_ids.device)
    count = 0
    for tid in word_token_ids.values():
        mask = (input_ids == tid)
        if mask.sum() < 2:
            continue
        embeds = sfe_embeds[mask]  # [N, d]
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
    """训练单个模型，v1.1: g参数10x lr + 辅助分化损失"""
    logger = logging.getLogger("sfe")

    # 参数分组，g用更高学习率
    g_params = []
    has_g = hasattr(model, 'sfe') and hasattr(model.sfe, 'g')
    if has_g:
        g_params = list(model.sfe.g.parameters())
    g_param_ids = set(id(p) for p in g_params)
    other_params = [p for p in model.parameters() if id(p) not in g_param_ids]

    param_groups = [{'params': other_params, 'lr': config['lr']}]
    if g_params:
        param_groups.append({'params': g_params, 'lr': config['lr'] * config['g_lr_multiplier']})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=config["weight_decay"])

    total_steps = config["total_steps"]
    warmup_steps = config["warmup_steps"]
    lr = config["lr"]

    model.train()
    model.to(device)

    steps_list, train_loss_list, val_ppl_list = [], [], []
    alpha_cos_list, probe_acc_list = [], []
    has_sfe = hasattr(model, 'get_sfe_embeddings')

    # 辅助损失需要word_token_ids
    word_token_ids = get_word_token_ids(tokenizer) if has_g else {}

    logger.info(f"[{model_name}] 开始训练, total_steps={total_steps}, g_lr_mult={config['g_lr_multiplier']}, lambda_div={config['lambda_div']}")
    train_iter = iter(train_loader)
    step = 0

    while step < total_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch.to(device)
        current_lr = cosine_scheduler(step, total_steps, warmup_steps, lr)
        for pg in optimizer.param_groups:
            if pg is param_groups[0]:
                pg["lr"] = current_lr
            elif len(param_groups) > 1 and pg is param_groups[1]:
                pg["lr"] = current_lr * config['g_lr_multiplier']

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
                _, alpha_mean = compute_alpha_divergence(model, val_loader, tokenizer, device)
                alpha_cos_list.append(round(alpha_mean, 4))
                logger.info(f"[{model_name}] step={step}, alpha_cos_mean={alpha_mean:.4f}")

        if step % config["probe_every"] == 0 and has_sfe:
            _, probe_mean = compute_probe_accuracy(model, val_loader, tokenizer, device)
            probe_acc_list.append(round(probe_mean, 4))
            logger.info(f"[{model_name}] step={step}, probe_accuracy={probe_mean:.4f}")

    # 最终评估
    final_ppl = evaluate_ppl(model, val_loader, device)
    logger.info(f"[{model_name}] 训练完成, final_ppl={final_ppl:.2f}")

    curve = {
        "steps": steps_list,
        "train_loss": train_loss_list,
        "val_ppl": val_ppl_list,
    }
    if has_sfe:
        curve["alpha_cos_mean"] = alpha_cos_list
        curve["alpha_cos_min"] = min(alpha_cos_list) if alpha_cos_list else None
        curve["alpha_cos_min_step"] = steps_list[alpha_cos_list.index(min(alpha_cos_list))] if alpha_cos_list else None
        if probe_acc_list:
            curve["probe_accuracy"] = probe_acc_list

    final_info = {
        "val_ppl": round(final_ppl, 2),
        "params": sum(p.numel() for p in model.parameters()),
    }
    if has_sfe and alpha_cos_list:
        final_info["alpha_cos_mean_final"] = alpha_cos_list[-1]
    if has_sfe and probe_acc_list:
        final_info["probe_accuracy_final"] = probe_acc_list[-1]

    return curve, final_info


# ============ 主函数 ============
def main():
    set_seed(CONFIG["seed"])

    os.makedirs(CONFIG["results_dir"], exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG["log_file"]), exist_ok=True)

    # 日志
    logger = logging.getLogger("sfe")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(CONFIG["log_file"], encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(ch)

    device = CONFIG["device"]
    logger.info(f"SFE v1.1 — g(ctx)修复版")
    logger.info(f"设备: {device}")
    logger.info(f"配置: {json.dumps(CONFIG, indent=2, ensure_ascii=False)}")

    # 加载数据
    logger.info("加载WikiText-103...")
    train_ids, val_ids, vocab_size, tokenizer = load_wikitext_data()
    logger.info(f"vocab={vocab_size}, train_tokens={len(train_ids)}, val_tokens={len(val_ids)}")

    # OOM重试
    batch_sizes = [CONFIG["batch_size"], 16, 8]
    for bs in batch_sizes:
        try:
            logger.info(f"batch_size={bs}")
            train_ds = WikiTextDataset(train_ids, CONFIG["seq_len"])
            val_ds = WikiTextDataset(val_ids, CONFIG["seq_len"])
            train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True)
            val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

            d = CONFIG["d"]
            k = CONFIG["k"]
            nl = CONFIG["n_layers"]
            nh = CONFIG["n_heads"]

            training_curves = {}
            final_results = {}

            # v1.1改动5：跳过Model A，只跑C和B
            # # Model A (跳过)
            # logger.info("=" * 50 + "\n训练 Model A (Static Baseline)")
            # model_a = ModelA_Static(vocab_size, d_model=d, n_layers=nl, n_heads=nh)
            # ...

            # Model C
            logger.info("=" * 50 + "\n训练 Model C (SFE Ablation)")
            model_c = ModelC_SFEAblation(vocab_size, k=k, d_model=d, ctx_window=CONFIG["ctx_window"], n_layers=nl, n_heads=nh)
            pc = sum(p.numel() for p in model_c.parameters())
            logger.info(f"Model C params: {pc:,}")
            curve_c, final_c = train_model(model_c, "ModelC_SFEAblation", train_loader, val_loader, tokenizer, device, CONFIG)
            training_curves["C_sfe_ablation"] = curve_c
            final_results["C_sfe_ablation"] = final_c
            del model_c
            torch.cuda.empty_cache()

            # Model B
            logger.info("=" * 50 + "\n训练 Model B (SFE Full, g_init=0.001, g_lr=10x)")
            model_b = ModelB_SFE(vocab_size, k=k, d_model=d, ctx_window=CONFIG["ctx_window"], n_layers=nl, n_heads=nh)
            pb = sum(p.numel() for p in model_b.parameters())
            logger.info(f"Model B params: {pb:,}")
            curve_b, final_b = train_model(model_b, "ModelB_SFE", train_loader, val_loader, tokenizer, device, CONFIG)
            training_curves["B_sfe_full"] = curve_b
            final_results["B_sfe_full"] = final_b
            del model_b
            torch.cuda.empty_cache()

            # Verdict（含alpha_cos_min判定）
            alpha_b = final_results["B_sfe_full"].get("alpha_cos_mean_final")
            alpha_b_min = training_curves["B_sfe_full"].get("alpha_cos_min")
            probe_b = final_results["B_sfe_full"].get("probe_accuracy_final")
            probe_c = final_results["C_sfe_ablation"].get("probe_accuracy_final")

            g_ctx_works = alpha_b is not None and alpha_b < 0.80
            dynamic_matters = (probe_b is not None and probe_c is not None and probe_b > probe_c)

            # 三级判定
            if alpha_b_min is not None and alpha_b_min < 0.75 and alpha_b is not None and alpha_b < 0.80:
                divergence_verdict = "stable_divergence"
            elif alpha_b_min is not None and alpha_b_min < 0.75 and alpha_b is not None and alpha_b >= 0.85:
                divergence_verdict = "suppressed_by_main_loss"
            else:
                divergence_verdict = "no_divergence"

            if g_ctx_works and dynamic_matters:
                rec = "SFE g(ctx) validated with proper training: proceed to scale"
            elif g_ctx_works and not dynamic_matters:
                rec = "g activates but no semantic benefit: architecture needs rethink"
            elif divergence_verdict == "suppressed_by_main_loss":
                rec = "g learns then gets suppressed: confirms need for symbiotic architecture"
            else:
                rec = "g(ctx) hypothesis rejected even with proper training"

            verdict = {
                "g_ctx_works": g_ctx_works,
                "dynamic_matters": dynamic_matters,
                "alpha_cos_min": alpha_b_min,
                "alpha_cos_min_step": training_curves["B_sfe_full"].get("alpha_cos_min_step"),
                "divergence_verdict": divergence_verdict,
                "recommendation": rec,
            }

            output = {
                "version": "v1.1",
                "changes_from_v10": ["g_init_std=0.001", "g_lr_multiplier=10", "total_steps=5000", "measure_every=200", "alpha_diversity_loss_lambda=0.01"],
                "training_curves": training_curves,
                "final": final_results,
                "verdict": verdict,
            }

            results_path = os.path.join(CONFIG["results_dir"], "results_v11.json")
            with open(results_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

            logger.info(f"结果已保存: {results_path}")
            logger.info(f"Verdict: {json.dumps(verdict, indent=2)}")
            print(f"\n{'='*50}\n实验完成! {results_path}\nVerdict: {rec}\n{'='*50}")
            break

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM with batch_size={bs}")
            torch.cuda.empty_cache()
            if bs == batch_sizes[-1]:
                logger.error("所有batch_size都OOM")
                raise


if __name__ == "__main__":
    main()
