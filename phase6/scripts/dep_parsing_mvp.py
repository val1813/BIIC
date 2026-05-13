"""
Phase 6: 依存句法MVP
BIIC grade-2 + Biaffine vs Transformer + Biaffine
数据集: Universal Dependencies en_ewt
评估: UAS / LAS
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

sys.path.insert(0, '/data/biic')
sys.path.insert(0, '/data/biic/phase2')

from clifford_cl41 import GRADE_SLICES, N_BLADES
from token_to_ic import TokenToImmutableCore
from mutable_state import BIICLayer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================
# Data Loading
# ============================================================

class UDDataset(Dataset):
    """Universal Dependencies dataset from CoNLL-U files"""
    def __init__(self, conllu_path, max_len=64, word2id=None, label2id=None):
        print(f"Loading CoNLL-U from {conllu_path}...")

        self.sentences = []
        self.heads = []
        self.labels = []
        self.max_len = max_len

        # Build or reuse vocab
        build_vocab = (word2id is None)
        if build_vocab:
            self.word2id = {'<pad>': 0, '<unk>': 1, '<root>': 2}
            self.label2id = {'<pad>': 0}
        else:
            self.word2id = word2id
            self.label2id = label2id

        # Parse CoNLL-U
        cur_tokens = []
        cur_heads = []
        cur_deprels = []

        with open(conllu_path, 'r', encoding='utf-8') as f:
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
                if line.startswith('#'):
                    continue
                parts = line.split('\t')
                if len(parts) < 8:
                    continue
                # Skip multi-word tokens (e.g., "1-2")
                if '-' in parts[0] or '.' in parts[0]:
                    continue

                token = parts[1]
                head = int(parts[6]) if parts[6] != '_' else 0
                deprel = parts[7].split(':')[0] if parts[7] != '_' else 'dep'

                cur_tokens.append(token)
                cur_heads.append(head)
                cur_deprels.append(deprel)

                if build_vocab:
                    if token.lower() not in self.word2id:
                        self.word2id[token.lower()] = len(self.word2id)
                    if deprel not in self.label2id:
                        self.label2id[deprel] = len(self.label2id)

            # Last sentence
            if cur_tokens and len(cur_tokens) <= max_len - 1:
                self.sentences.append(cur_tokens)
                self.heads.append(cur_heads)
                self.labels.append(cur_deprels)

        self.vocab_size = len(self.word2id)
        self.n_labels = len(self.label2id)
        print(f"  {len(self.sentences)} sentences, vocab={self.vocab_size}, labels={self.n_labels}")

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        tokens = self.sentences[idx]
        heads = self.heads[idx]
        labels = self.labels[idx]

        # Convert to ids (prepend root)
        token_ids = [self.word2id['<root>']]
        for t in tokens:
            token_ids.append(self.word2id.get(t.lower(), self.word2id['<unk>']))

        head_ids = [0]  # root's head is itself
        for h in heads:
            head_ids.append(h)

        label_ids = [0]  # root label
        for r in labels:
            label_ids.append(self.label2id.get(r, 0))

        return {
            'token_ids': token_ids,
            'heads': head_ids,
            'labels': label_ids,
            'length': len(token_ids),
        }


def collate_fn(batch):
    max_len = max(b['length'] for b in batch)
    B = len(batch)

    token_ids = torch.zeros(B, max_len, dtype=torch.long)
    heads = torch.full((B, max_len), -1, dtype=torch.long)
    labels = torch.zeros(B, max_len, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)

    for i, b in enumerate(batch):
        L = b['length']
        token_ids[i, :L] = torch.tensor(b['token_ids'])
        heads[i, :L] = torch.tensor(b['heads'])
        labels[i, :L] = torch.tensor(b['labels'])
        mask[i, :L] = True

    return token_ids.to(device), heads.to(device), labels.to(device), mask.to(device)


# ============================================================
# Biaffine Head
# ============================================================

class BiaffineHead(nn.Module):
    def __init__(self, input_dim, n_labels, arc_dim=500, label_dim=100):
        super().__init__()
        self.arc_head = nn.Linear(input_dim, arc_dim)
        self.arc_dep = nn.Linear(input_dim, arc_dim)
        self.label_head = nn.Linear(input_dim, label_dim)
        self.label_dep = nn.Linear(input_dim, label_dim)

        # Biaffine weights
        self.W_arc = nn.Parameter(torch.zeros(arc_dim, arc_dim))
        self.b_arc = nn.Parameter(torch.zeros(1))
        self.W_label = nn.Parameter(torch.zeros(n_labels, label_dim, label_dim))
        self.b_label = nn.Parameter(torch.zeros(n_labels))

        nn.init.xavier_uniform_(self.W_arc)
        nn.init.xavier_uniform_(self.W_label.view(n_labels, -1))

    def forward(self, h):
        """
        h: [B, L, d]
        Returns: arc_scores [B, L, L], label_scores [B, L, L, n_labels]
        """
        B, L, _ = h.shape

        arc_h = torch.tanh(self.arc_head(h))  # [B, L, arc_dim]
        arc_d = torch.tanh(self.arc_dep(h))   # [B, L, arc_dim]
        label_h = torch.tanh(self.label_head(h))  # [B, L, label_dim]
        label_d = torch.tanh(self.label_dep(h))   # [B, L, label_dim]

        # Arc scores: biaffine(head, dep) -> [B, L_head, L_dep]
        # score[i,j] = arc_h[i] @ W @ arc_d[j] + b
        arc_scores = torch.bmm(arc_h @ self.W_arc, arc_d.transpose(1, 2)) + self.b_arc

        # Label scores: for each (head, dep) pair
        # [B, L, label_dim] -> [B, L, 1, label_dim] x [n_labels, label_dim, label_dim] x [B, 1, L, label_dim]
        # Simplified: compute for all pairs
        # label_h: [B, L, ld], label_d: [B, L, ld]
        # For each label k: score_k[i,j] = label_h[i] @ W_k @ label_d[j]
        n_labels = self.W_label.shape[0]
        ld = label_h.shape[-1]

        # Efficient: [B, L, ld] @ [n_labels, ld, ld] -> use einsum
        # label_scores[b, i, j, k] = label_h[b,i] @ W_label[k] @ label_d[b,j]
        # = einsum('bid, kdf, bjf -> bijk')
        label_scores = torch.einsum('bid,kdf,bjf->bijk', label_h, self.W_label, label_d)
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
        # grade-2 output dim = n_channels * 10
        self.output_dim = n_channels * 10

    def forward(self, token_ids):
        """
        token_ids: [B, L]
        Returns: [B, L, n_channels * 10] (grade-2 features)
        """
        mv = self.encoder(token_ids)  # [B, L, C, 32]
        mv_prior = mv.clone().detach()
        for layer in self.layers:
            mv = layer(mv, mv_prior)

        # Extract grade-2
        g2_start, g2_end = GRADE_SLICES[2]
        grade2 = mv[..., g2_start:g2_end]  # [B, L, C, 10]
        B, L, C, d = grade2.shape
        return grade2.reshape(B, L, C * d)  # [B, L, C*10]


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
        """
        token_ids: [B, L]
        Returns: [B, L, d_model]
        """
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        x = self.embed(token_ids) + self.pos_embed(pos)
        # Create padding mask
        pad_mask = (token_ids == 0)  # True where padded
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
    """
    arc_scores: [B, L, L]
    label_scores: [B, L, L, n_labels]
    heads: [B, L] (gold head indices)
    labels: [B, L] (gold label indices)
    mask: [B, L] (True for real tokens)
    """
    B, L = heads.shape

    # Arc loss: cross-entropy over possible heads
    # For each token, predict which position is its head
    arc_loss = F.cross_entropy(
        arc_scores[mask].view(-1, L),
        heads[mask].clamp(0, L-1),
        reduction='mean'
    )

    # Label loss: given gold head, predict label
    # Gather label scores at gold head positions
    # label_scores[b, head[b,i], i, :] for each token i
    head_indices = heads.clamp(0, L-1)  # [B, L]
    # Gather: for each (b, i), get label_scores[b, head[b,i], i, :]
    head_idx_expanded = head_indices.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, label_scores.shape[-1])
    # label_scores is [B, L_head, L_dep, n_labels]
    # We want label_scores[b, head[b,i], i, :] for dep position i
    # Rearrange: label_scores[b, :, i, :] then gather at head position
    label_at_gold = torch.gather(
        label_scores.transpose(1, 2),  # [B, L_dep, L_head, n_labels]
        2,
        head_idx_expanded
    ).squeeze(2)  # [B, L, n_labels]

    label_loss = F.cross_entropy(
        label_at_gold[mask],
        labels[mask],
        reduction='mean'
    )

    return arc_loss + label_loss


def evaluate(model, dataloader, max_len):
    model.eval()
    correct_arcs = 0
    correct_labels = 0
    total = 0

    with torch.no_grad():
        for token_ids, heads, labels, mask in dataloader:
            arc_scores, label_scores = model(token_ids)
            B, L = heads.shape

            # Predict arcs
            pred_heads = arc_scores.argmax(dim=2)  # [B, L]

            # Predict labels at predicted heads
            pred_head_idx = pred_heads.unsqueeze(2).unsqueeze(3).expand(-1, -1, 1, label_scores.shape[-1])
            label_at_pred = torch.gather(
                label_scores.transpose(1, 2),
                2,
                pred_head_idx
            ).squeeze(2)
            pred_labels = label_at_pred.argmax(dim=-1)

            # Skip root (position 0) and padding
            eval_mask = mask.clone()
            eval_mask[:, 0] = False  # don't evaluate root

            correct_arcs += ((pred_heads == heads) & eval_mask).sum().item()
            correct_labels += ((pred_heads == heads) & (pred_labels == labels) & eval_mask).sum().item()
            total += eval_mask.sum().item()

    uas = correct_arcs / total if total > 0 else 0
    las = correct_labels / total if total > 0 else 0
    return uas, las


def train_model(model, train_loader, val_loader, n_steps=5000, lr=2e-3, eval_every=500, model_name="model"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*60}")
    print(f"Training {model_name} ({n_params:,} params)")
    print(f"{'='*60}")

    log = {'steps': [], 'train_losses': [], 'val_uas': [], 'val_las': []}
    step = 0
    t_start = time.time()
    max_len = 65  # max_len + 1 for root

    model.train()
    while step < n_steps:
        for token_ids, heads, labels, mask in train_loader:
            if step >= n_steps:
                break

            # LR warmup
            if step < 200:
                cur_lr = lr * step / 200
            else:
                progress = (step - 200) / (n_steps - 200)
                cur_lr = lr * 0.5 * (1 + math.cos(math.pi * progress))
            for pg in optimizer.param_groups:
                pg['lr'] = cur_lr

            optimizer.zero_grad(set_to_none=True)
            arc_scores, label_scores = model(token_ids)
            loss = compute_loss(arc_scores, label_scores, heads, labels, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % eval_every == 0:
                uas, las = evaluate(model, val_loader, max_len)
                elapsed = time.time() - t_start
                speed = (step + 1) / elapsed if elapsed > 0 else 0
                print(f"Step {step:5d} | loss={loss.item():.4f} | UAS={uas:.4f} | LAS={las:.4f} | {speed:.1f} step/s",
                      flush=True)
                log['steps'].append(step)
                log['train_losses'].append(loss.item())
                log['val_uas'].append(uas)
                log['val_las'].append(las)
                model.train()

            step += 1

    # Final eval
    final_uas, final_las = evaluate(model, val_loader, max_len)
    elapsed = time.time() - t_start
    print(f"\nFinal: UAS={final_uas:.4f} LAS={final_las:.4f} ({elapsed/60:.1f} min)")

    log['final_uas'] = final_uas
    log['final_las'] = final_las
    log['training_minutes'] = elapsed / 60
    log['params'] = n_params
    return log


# ============================================================
# Main
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, choices=['biic', 'transformer'])
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_path', type=str, default='/data/biic/results/phase6_dep_parsing.json')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print(f"Phase 6: Dependency Parsing MVP - {args.model.upper()}")
    print("=" * 60)

    # Load data
    os.environ['http_proxy'] = 'http://proxy.mornai.cn:7890'
    os.environ['https_proxy'] = 'http://proxy.mornai.cn:7890'

    # Load from CoNLL-U file, split 90/10 for train/dev
    conllu_path = '/data/biic/ud_english_ewt/en_ewt-ud-train.conllu'
    full_ds = UDDataset(conllu_path, max_len=64)

    # Split 90/10
    n_total = len(full_ds)
    n_train = int(n_total * 0.9)
    train_ds = UDDataset.__new__(UDDataset)
    train_ds.word2id = full_ds.word2id
    train_ds.label2id = full_ds.label2id
    train_ds.vocab_size = full_ds.vocab_size
    train_ds.n_labels = full_ds.n_labels
    train_ds.max_len = 64
    train_ds.sentences = full_ds.sentences[:n_train]
    train_ds.heads = full_ds.heads[:n_train]
    train_ds.labels = full_ds.labels[:n_train]

    val_ds = UDDataset.__new__(UDDataset)
    val_ds.word2id = full_ds.word2id
    val_ds.label2id = full_ds.label2id
    val_ds.vocab_size = full_ds.vocab_size
    val_ds.n_labels = full_ds.n_labels
    val_ds.max_len = 64
    val_ds.sentences = full_ds.sentences[n_train:]
    val_ds.heads = full_ds.heads[n_train:]
    val_ds.labels = full_ds.labels[n_train:]

    print(f"Train: {len(train_ds.sentences)}, Val: {len(val_ds.sentences)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_fn)

    # Build model
    if args.model == 'biic':
        encoder = BIICDepEncoder(train_ds.vocab_size, n_channels=8, n_layers=6)
    else:
        # Match BIIC param count approximately
        # BIIC: vocab*8*32 + 6*layer_params + biaffine
        # Use d_model=80 to get similar param count
        encoder = TransformerDepEncoder(train_ds.vocab_size, d_model=80, n_layers=6, n_heads=4)

    model = DepParsingModel(encoder, train_ds.n_labels).to(device)

    # Train
    log = train_model(model, train_loader, val_loader,
                     n_steps=args.n_steps, lr=args.lr,
                     eval_every=500, model_name=args.model)

    log['model'] = args.model
    log['config'] = vars(args)
    log['vocab_size'] = train_ds.vocab_size
    log['n_labels'] = train_ds.n_labels
    log['n_train'] = len(train_ds)
    log['n_val'] = len(val_ds.sentences)

    # Save
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {args.save_path}")


if __name__ == '__main__':
    main()
