"""
Phase 6: Dependency Parsing Data Efficiency Experiment
BIIC grade-2 + Biaffine vs Transformer + Biaffine
Key metric: data efficiency curve (10%/25%/50%/100% training data)

Each configuration: 3 seeds
GPU0: BIIC encoder
GPU1: Transformer encoder

Saves: /data/biic/results/phase6_dep_parsing_efficiency.json
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import json
import time
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
import argparse

sys.path.insert(0, "/data/biic")
sys.path.insert(0, "/data/biic/phase2")

from clifford_cl41 import GRADE_SLICES, N_BLADES
from token_to_ic import TokenToImmutableCore
from mutable_state import BIICLayer


# ============================================================
# Data Loading
# ============================================================

class UDDataset(Dataset):
    """Universal Dependencies dataset from CoNLL-U files"""
    def __init__(self, conllu_path, max_len=64, word2id=None, label2id=None):
        self.sentences = []
        self.heads = []
        self.labels = []
        self.max_len = max_len

        build_vocab = (word2id is None)
        if build_vocab:
            self.word2id = {"<pad>": 0, "<unk>": 1, "<root>": 2}
            self.label2id = {"<pad>": 0}
        else:
            self.word2id = word2id
            self.label2id = label2id

        cur_tokens = []
        cur_heads = []
        cur_deprels = []

        with open(conllu_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    if cur_tokens:
                        if len(cur_tokens) <= max_len - 1:
                            self.sentences.append(cur_tokens)
                            self.heads.append(cur_heads)
                            self.labels.append(cur_deprels)
                        cur_tokens = []
                        cur_heads = []
                        cur_deprels = []
                    continue
                if line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 8:
                    continue
                if "-" in parts[0] or "." in parts[0]:
                    continue

                token = parts[1]
                head = int(parts[6]) if parts[6] != "_" else 0
                deprel = parts[7].split(":")[0] if parts[7] != "_" else "dep"

                cur_tokens.append(token)
                cur_heads.append(head)
                cur_deprels.append(deprel)

                if build_vocab:
                    if token.lower() not in self.word2id:
                        self.word2id[token.lower()] = len(self.word2id)
                    if deprel not in self.label2id:
                        self.label2id[deprel] = len(self.label2id)

            if cur_tokens and len(cur_tokens) <= max_len - 1:
                self.sentences.append(cur_tokens)
                self.heads.append(cur_heads)
                self.labels.append(cur_deprels)

        self.vocab_size = len(self.word2id)
        self.n_labels = len(self.label2id)

    def subset(self, fraction, seed=42):
        """Return a new dataset with only a fraction of the data"""
        rng = np.random.RandomState(seed)
        n = len(self.sentences)
        n_use = max(1, int(n * fraction))
        indices = rng.permutation(n)[:n_use]
        new_ds = UDDataset.__new__(UDDataset)
        new_ds.word2id = self.word2id
        new_ds.label2id = self.label2id
        new_ds.vocab_size = self.vocab_size
        new_ds.n_labels = self.n_labels
        new_ds.max_len = self.max_len
        new_ds.sentences = [self.sentences[i] for i in indices]
        new_ds.heads = [self.heads[i] for i in indices]
        new_ds.labels = [self.labels[i] for i in indices]
        return new_ds

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        heads = self.heads[idx]
        labels = self.labels[idx]

        token_ids = [self.word2id["<root>"]]
        for t in tokens:
            token_ids.append(self.word2id.get(t.lower(), self.word2id["<unk>"]))

        head_ids = [0]
        for h in heads:
            head_ids.append(h)

        label_ids = [0]
        for r in labels:
            label_ids.append(self.label2id.get(r, 0))

        return {
            "token_ids": token_ids,
            "heads": head_ids,
            "labels": label_ids,
            "length": len(token_ids),
        }


def collate_fn(batch, device):
    max_len = max(b["length"] for b in batch)
    B = len(batch)

    token_ids = torch.zeros(B, max_len, dtype=torch.long)
    heads = torch.full((B, max_len), -1, dtype=torch.long)
    labels = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b["length"]
        token_ids[i, :L] = torch.tensor(b["token_ids"])
        heads[i, :L] = torch.tensor(b["heads"])
        labels[i, :L] = torch.tensor(b["labels"])
        mask[i, :L] = True

    return token_ids.to(device), heads.to(device), labels.to(device), mask.to(device)


# ============================================================
# Biaffine Head
# ============================================================

class BiaffineHead(nn.Module):
    def __init__(self, input_dim, n_labels, arc_dim=256, label_dim=64):
        super().__init__()
        self.arc_head = nn.Linear(input_dim, arc_dim)
        self.arc_dep = nn.Linear(input_dim, arc_dim)
        self.label_head = nn.Linear(input_dim, label_dim)
        self.label_dep = nn.Linear(input_dim, label_dim)

        self.W_arc = nn.Parameter(torch.zeros(arc_dim, arc_dim))
        self.b_arc = nn.Parameter(torch.zeros(1))
        self.W_label = nn.Parameter(torch.zeros(n_labels, label_dim, label_dim))
        self.b_label = nn.Parameter(torch.zeros(n_labels))

        nn.init.xavier_uniform_(self.W_arc)
        nn.init.xavier_uniform_(self.W_label.view(n_labels, -1))

    def forward(self, h):
        B, L, _ = h.shape
        arc_h = torch.tanh(self.arc_head(h))
        arc_d = torch.tanh(self.arc_dep(h))
        label_h = torch.tanh(self.label_head(h))
        label_d = torch.tanh(self.label_dep(h))

        arc_scores = torch.bmm(arc_h @ self.W_arc, arc_d.transpose(1, 2)) + self.b_arc
        label_scores = torch.einsum("bid,kdf,bjf->bijk", label_h, self.W_label, label_d)
        label_scores = label_scores + self.b_label

        return arc_scores, label_scores


# ============================================================
# BIIC Encoder (grade-2 features)
# ============================================================

class BIICDepEncoder(nn.Module):
    def __init__(self, vocab_size, n_channels=8, n_layers=6):
        super().__init__()
        self.encoder = TokenToImmutableCore(vocab_size, n_channels)
        self.layers = nn.ModuleList([
            BIICLayer(n_channels, use_residual=True, use_eraser=True)
            for _ in range(n_layers)
        ])
        self.n_channels = n_channels
        self.output_dim = n_channels * 10  # grade-2

    def forward(self, token_ids):
        mv = self.encoder(token_ids)  # [B, L, C, 32]
        mv_prior = mv.clone().detach()
        for layer in self.layers:
            mv = layer(mv, mv_prior)
        g2_start, g2_end = GRADE_SLICES[2]
        grade2 = mv[..., g2_start:g2_end]  # [B, L, C, 10]
        B, L, C, d = grade2.shape
        return grade2.reshape(B, L, C * d)


# ============================================================
# Transformer Encoder (baseline)
# ============================================================

class TransformerDepEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=80, n_layers=6, n_heads=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(512, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model*4,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_dim = d_model

    def forward(self, token_ids):
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(token_ids) + self.pos_embed(pos)
        pad_mask = (token_ids == 0)
        h = self.transformer(x, src_key_padding_mask=pad_mask)
        return h


# ============================================================
# Full Model
# ============================================================

class DepParsingModel(nn.Module):
    def __init__(self, encoder, n_labels):
        super().__init__()
        self.encoder = encoder
        self.head = BiaffineHead(encoder.output_dim, n_labels)

    def forward(self, token_ids):
        h = self.encoder(token_ids)
        arc_scores, label_scores = self.head(h)
        return arc_scores, label_scores


# ============================================================
# Training & Evaluation
# ============================================================

def compute_loss(arc_scores, label_scores, heads, labels, mask):
    B, L = heads.shape
    arc_loss = F.cross_entropy(
        arc_scores[mask].view(-1, arc_scores.shape[-1]),
        heads[mask].clamp(0, arc_scores.shape[-1]-1),
        reduction="mean"
    )
    head_indices = heads.clamp(0, L-1)
    head_idx_expanded = head_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, label_scores.shape[-1])
    label_at_gold = torch.gather(
        label_scores.transpose(1, 2), 2, head_idx_expanded
    ).squeeze(2)
    label_loss = F.cross_entropy(label_at_gold[mask], labels[mask], reduction="mean")
    return arc_loss + label_loss


def evaluate(model, dataloader, device):
    model.eval()
    correct_arcs = 0
    correct_labels = 0
    total = 0

    with torch.no_grad():
        for token_ids, heads, labels, mask in dataloader:
            arc_scores, label_scores = model(token_ids)
            pred_heads = arc_scores.argmax(dim=2)
            pred_head_idx = pred_heads.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, label_scores.shape[-1])
            label_at_pred = torch.gather(
                label_scores.transpose(1, 2), 2, pred_head_idx
            ).squeeze(2)
            pred_labels = label_at_pred.argmax(dim=-1)

            eval_mask = mask.clone()
            eval_mask[:, 0] = False

            correct_arcs += ((pred_heads == heads) & eval_mask).sum().item()
            correct_labels += ((pred_heads == heads) & (pred_labels == labels) & eval_mask).sum().item()
            total += eval_mask.sum().item()

    uas = correct_arcs / total if total > 0 else 0
    las = correct_labels / total if total > 0 else 0
    return uas, las


def train_single_config(model, train_ds, val_ds, device, n_steps=5000, lr=2e-3,
                        eval_every=100, batch_size=32, model_name="model"):
    """Train one configuration, return log with UAS/LAS curves"""
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                             collate_fn=lambda b: collate_fn(b, device), drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                           collate_fn=lambda b: collate_fn(b, device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())

    log = {"steps": [], "train_losses": [], "val_uas": [], "val_las": [], "params": n_params}
    step = 0
    t_start = time.time()

    model.train()
    while step < n_steps:
        for token_ids, heads, labels, mask in train_loader:
            if step >= n_steps:
                break

            # LR schedule
            if step < 200:
                cur_lr = lr * (step + 1) / 200
            else:
                progress = (step - 200) / max(1, n_steps - 200)
                cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg["lr"] = cur_lr

            optimizer.zero_grad(set_to_none=True)
            arc_scores, label_scores = model(token_ids)
            loss = compute_loss(arc_scores, label_scores, heads, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % eval_every == 0:
                uas, las = evaluate(model, val_loader, device)
                elapsed = time.time() - t_start
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                print("  [%s] Step %5d | loss=%.4f | UAS=%.4f | LAS=%.4f | %.1f step/s" % (
                    model_name, step, loss.item(), uas, las, speed), flush=True)
                log["steps"].append(step)
                log["train_losses"].append(loss.item())
                log["val_uas"].append(uas)
                log["val_las"].append(las)
                model.train()

            step += 1

    # Final eval
    final_uas, final_las = evaluate(model, val_loader, device)
    elapsed = time.time() - t_start
    log["final_uas"] = final_uas
    log["final_las"] = final_las
    log["training_minutes"] = elapsed / 60
    print("  [%s] Final: UAS=%.4f LAS=%.4f (%.1f min)" % (model_name, final_uas, final_las, elapsed/60), flush=True)
    return log


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["biic", "transformer"])
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--n_steps", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--save_path", type=str, default="/data/biic/results/phase6_dep_parsing_efficiency.json")
    args = parser.parse_args()

    device = torch.device("cuda:%d" % args.gpu if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("Phase 6: Dep Parsing Data Efficiency - %s (GPU %d)" % (args.model.upper(), args.gpu))
    print("Device: %s" % device)
    print("=" * 60)

    # Load full dataset
    conllu_train = "/data/biic/ud_english_ewt/en_ewt-ud-train.conllu"
    conllu_dev = "/data/biic/ud_english_ewt/en_ewt-ud-dev.conllu"

    print("Loading data...")
    full_train = UDDataset(conllu_train, max_len=64)
    val_ds = UDDataset(conllu_dev, max_len=64, word2id=full_train.word2id, label2id=full_train.label2id)
    print("  Train: %d sentences, Vocab: %d, Labels: %d" % (len(full_train), full_train.vocab_size, full_train.n_labels))
    print("  Val: %d sentences" % len(val_ds))

    # Data fractions and seeds
    fractions = [0.10, 0.25, 0.50, 1.00]
    seeds = [42, 123, 456]

    all_results = {
        "model": args.model,
        "fractions": fractions,
        "seeds": seeds,
        "vocab_size": full_train.vocab_size,
        "n_labels": full_train.n_labels,
        "n_train_full": len(full_train),
        "n_val": len(val_ds),
        "runs": []
    }

    for frac in fractions:
        for seed in seeds:
            print("\n" + "-" * 40)
            print("Fraction=%.0f%%, Seed=%d" % (frac * 100, seed))
            print("-" * 40)

            torch.manual_seed(seed)
            np.random.seed(seed)

            # Get data subset
            if frac < 1.0:
                train_subset = full_train.subset(frac, seed=seed)
            else:
                train_subset = full_train
            print("  Training on %d sentences" % len(train_subset))

            # Build model
            if args.model == "biic":
                encoder = BIICDepEncoder(full_train.vocab_size, n_channels=8, n_layers=6)
            else:
                encoder = TransformerDepEncoder(full_train.vocab_size, d_model=80, n_layers=6, n_heads=4)

            model = DepParsingModel(encoder, full_train.n_labels).to(device)
            name = "%s_f%.0f_s%d" % (args.model, frac*100, seed)

            log = train_single_config(
                model, train_subset, val_ds, device,
                n_steps=args.n_steps, lr=args.lr,
                eval_every=100, batch_size=args.batch_size,
                model_name=name
            )
            log["fraction"] = frac
            log["seed"] = seed
            log["n_train"] = len(train_subset)
            all_results["runs"].append(log)

            # Free memory
            del model, encoder
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: %s" % args.model.upper())
    print("=" * 60)
    print("Fraction | Avg UAS | Avg LAS | Convergence")
    print("-" * 50)
    for frac in fractions:
        runs = [r for r in all_results["runs"] if r["fraction"] == frac]
        avg_uas = np.mean([r["final_uas"] for r in runs])
        avg_las = np.mean([r["final_las"] for r in runs])
        print("  %4.0f%%  |  %.4f |  %.4f |  %.1f min avg" % (
            frac*100, avg_uas, avg_las, np.mean([r["training_minutes"] for r in runs])))

    # Save
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print("\nSaved: %s" % args.save_path)


if __name__ == "__main__":
    main()
