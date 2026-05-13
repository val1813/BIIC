"""
Phase 5.5: Relative Invariant Attention
Phase 5.6: Segmented Eraser

Combined script for both experiments.
Run with --experiment rel_attn or --experiment seg_eraser
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import json
import time
import math
import argparse
import numpy as np

sys.path.insert(0, '/data/biic')
sys.path.insert(0, '/data/biic/phase2')

from clifford_cl41 import GRADE_SLICES, N_BLADES
from token_to_ic import TokenToImmutableCore
from all_grade_decoder import AllGradeDecoder
from mutable_state import BIICLayer
from eraser_ops import GradeAwareEraser

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RelativeInvariantAttention(nn.Module):
    """
    5.5: Attention using grade-0 (semantic) + grade-2 (relative relation) scores.
    alpha controls how much the model uses grade-2 relative information.
    If alpha rises from ~0, the model is actively learning to use equivariant components.
    """
    def __init__(self, n_channels, window_size=16):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size

        # Grade-0 projections (semantic score)
        self.proj_q = nn.Linear(n_channels, n_channels)
        self.proj_k = nn.Linear(n_channels, n_channels)

        # Grade-2 value projection
        g2_dim = 10 * n_channels
        self.proj_v = nn.Linear(g2_dim, g2_dim)
        self.proj_out = nn.Linear(g2_dim, g2_dim)
        self.ln = nn.LayerNorm(g2_dim)

        # Alpha: controls relative score contribution, init=-4 (sigmoid~0.018)
        self.alpha = nn.Parameter(torch.tensor([-4.0]))
        # Gate for output, init=-4
        self.gate = nn.Parameter(torch.tensor([-4.0]))

    def forward(self, mv):
        B, L, C, D = mv.shape
        g0_start, g0_end = GRADE_SLICES[0]
        g2_start, g2_end = GRADE_SLICES[2]

        grade0 = mv[..., g0_start:g0_end].squeeze(-1)  # [B, L, C]
        grade2 = mv[..., g2_start:g2_end]  # [B, L, C, 10]

        g0_flat = grade0.reshape(B, L, -1)  # [B, L, C]
        g2_flat = grade2.reshape(B, L, -1)  # [B, L, C*10]

        # Semantic score from grade-0
        Q = self.proj_q(g0_flat)
        K = self.proj_k(g0_flat)
        scale = math.sqrt(Q.shape[-1])
        score_sem = torch.bmm(Q, K.transpose(1, 2)) / scale

        # Relative score from grade-2 (low-rank: use first 8*C dims)
        g2_small = g2_flat[..., :8 * self.n_channels]  # [B, L, 8*C]
        score_rel = torch.bmm(g2_small, g2_small.transpose(1, 2)) / (8 * self.n_channels) ** 0.5

        # Combined score
        alpha_val = torch.sigmoid(self.alpha)
        score = score_sem + alpha_val * score_rel

        # Causal + window mask
        mask = torch.ones(L, L, device=mv.device, dtype=torch.bool)
        for i in range(L):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i+1] = False
        score = score.masked_fill(mask.unsqueeze(0), float('-inf'))

        attn = F.softmax(score, dim=-1)
        V = self.proj_v(g2_flat)
        context = torch.bmm(attn, V)

        # Gated output to grade-2
        gate_val = torch.sigmoid(self.gate)
        g2_updated = g2_flat + gate_val * self.proj_out(self.ln(context))
        g2_updated = torch.clamp(g2_updated, -10.0, 10.0)

        mv_new = mv.clone()
        mv_new[..., g2_start:g2_end] = g2_updated.reshape(B, L, C, 10)
        return mv_new, alpha_val.item(), gate_val.item()


class BIICModelRelAttn(nn.Module):
    """Model with Relative Invariant Attention inserted at middle layer"""
    def __init__(self, vocab_size, n_channels, n_layers, d_hidden, seq_len=128):
        super().__init__()
        self.encoder = TokenToImmutableCore(vocab_size, n_channels)
        self.layers = nn.ModuleList([BIICLayer(n_channels, use_residual=True, use_eraser=True) for _ in range(n_layers)])
        self.rel_attn = RelativeInvariantAttention(n_channels, window_size=min(16, seq_len))
        self.decoder = AllGradeDecoder(vocab_size, n_channels, d_hidden)
        self.n_layers = n_layers

    def forward(self, token_ids):
        mv = self.encoder(token_ids)
        mv_prior = mv.clone().detach()
        alpha_val, gate_val = 0.0, 0.0
        for i, layer in enumerate(self.layers):
            mv = layer(mv, mv_prior)
            if i == self.n_layers // 2:
                mv, alpha_val, gate_val = self.rel_attn(mv)
        logits = self.decoder(mv)
        return logits, alpha_val, gate_val


class BIICModelSegEraser(nn.Module):
    """Model with Segmented Eraser: only erase every 4 layers, never erase grade-2"""
    def __init__(self, vocab_size, n_channels, n_layers, d_hidden):
        super().__init__()
        self.encoder = TokenToImmutableCore(vocab_size, n_channels)
        # Layers WITHOUT built-in eraser (we control eraser manually)
        self.layers = nn.ModuleList([BIICLayer(n_channels, use_residual=True, use_eraser=False) for _ in range(n_layers)])
        # Separate eraser for segmented use
        self.eraser = GradeAwareEraser(n_channels)
        self.decoder = AllGradeDecoder(vocab_size, n_channels, d_hidden)
        self.n_layers = n_layers

    def forward(self, token_ids):
        mv = self.encoder(token_ids)
        mv_prior = mv.clone().detach()

        g2_norms = []
        for i, layer in enumerate(self.layers):
            mv = layer(mv, mv_prior)

            # Record grade-2 norm at each layer
            g2_start, g2_end = GRADE_SLICES[2]
            g2_norm = mv[..., g2_start:g2_end].norm(dim=-1).mean().item()
            g2_norms.append(g2_norm)

            # Segmented eraser: only at block boundaries (every 4 layers)
            # Never erase grade-0 (invariant) or grade-2 (relation carrier)
            if (i + 1) % 4 == 0 and i < self.n_layers - 1:
                # Erase grade-1, 3, 4 only
                g1_start, g1_end = GRADE_SLICES[1]
                g3_start, g3_end = GRADE_SLICES[3]
                g4_start, g4_end = GRADE_SLICES[4]

                # decay_logits is [C, D], we need a scalar decay per grade slice
                decay = torch.sigmoid(self.eraser.decay_logits.mean())
                # Apply decay to grade-1, 3, 4 toward prior
                for gs, ge in [(g1_start, g1_end), (g3_start, g3_end), (g4_start, g4_end)]:
                    mv_slice = mv[..., gs:ge]
                    prior_slice = mv_prior[..., gs:ge]
                    mv = mv.clone()
                    mv[..., gs:ge] = mv_slice * (1 - decay) + prior_slice * decay

        logits = self.decoder(mv)
        return logits, g2_norms


class WikiTextDataLoader:
    def __init__(self, seq_len, batch_size, split='train'):
        from transformers import GPT2Tokenizer
        print("Loading tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('/data/biic/gpt2_tokenizer')
        self.seq_len = seq_len
        self.batch_size = batch_size
        print("Loading WikiText-103 %s..." % split)
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-103-v1', split=split,
                         cache_dir='/data/biic/hf_cache')
        text = " ".join([t for t in ds['text'] if len(t.strip()) > 20])
        print("Tokenizing...")
        self.tokens = self.tokenizer.encode(text)
        print("Total tokens: %d" % len(self.tokens))
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        self.pos = 0

    def get_batch(self):
        B, L = self.batch_size, self.seq_len
        if self.pos + B * (L + 1) > len(self.tokens):
            self.pos = 0
        inputs, targets = [], []
        for _ in range(B):
            chunk = self.tokens[self.pos:self.pos + L + 1]
            inputs.append(chunk[:-1])
            targets.append(chunk[1:])
            self.pos += L
        return torch.stack(inputs).to(device), torch.stack(targets).to(device)


def train_rel_attn(args):
    """Experiment 5.5: Relative Invariant Attention"""
    print("=" * 60)
    print("Experiment 5.5: Relative Invariant Attention")
    print("=" * 60)
    print("Config: n_channels=%d, seq=%d, batch=%d, steps=%d" % (
        args.n_channels, args.seq_len, args.batch_size, args.n_steps))

    try:
        data_loader = WikiTextDataLoader(args.seq_len, args.batch_size)
    except Exception:
        os.environ['http_proxy'] = 'http://proxy.mornai.cn:7890'
        os.environ['https_proxy'] = 'http://proxy.mornai.cn:7890'
        data_loader = WikiTextDataLoader(args.seq_len, args.batch_size)

    model = BIICModelRelAttn(
        vocab_size=50257, n_channels=args.n_channels, n_layers=6,
        d_hidden=128, seq_len=args.seq_len
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("Params: %d" % n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    torch.manual_seed(42)
    np.random.seed(42)

    log = {"steps": [], "losses": [], "alphas": [], "gates": [], "g2_grad_norms": []}
    t_start = time.time()

    for step in range(args.n_steps):
        if step < 200:
            lr = 3e-4 * step / 200
        else:
            progress = (step - 200) / (args.n_steps - 200)
            lr = 3e-4 * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        inputs, targets = data_loader.get_batch()
        optimizer.zero_grad(set_to_none=True)
        logits, alpha_val, gate_val = model(inputs)
        loss = F.cross_entropy(logits.view(-1, 50257), targets.view(-1))
        loss.backward()

        # Get grade-2 gradient norm
        g2_grad_norm = 0.0
        for name, param in model.rel_attn.named_parameters():
            if 'proj_v' in name and param.grad is not None:
                g2_grad_norm = param.grad.norm().item()
                break

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            ppl = math.exp(loss.item()) if loss.item() < 20 else float('inf')
            print("Step %4d | loss=%.4f | ppl=%.1f | alpha=%.4f | gate=%.4f | g2_grad=%.4f" % (
                step, loss.item(), ppl, alpha_val, gate_val, g2_grad_norm), flush=True)
            log["steps"].append(step)
            log["losses"].append(loss.item())
            log["alphas"].append(alpha_val)
            log["gates"].append(gate_val)
            log["g2_grad_norms"].append(g2_grad_norm)

    elapsed = time.time() - t_start
    final_loss = np.mean(log["losses"][-5:])
    log["final_loss"] = final_loss
    log["final_ppl"] = float(math.exp(final_loss))
    log["final_alpha"] = alpha_val
    log["final_gate"] = gate_val
    log["training_minutes"] = elapsed / 60
    log["params"] = n_params
    log["config"] = vars(args)

    print("\nDONE: Relative Invariant Attention")
    print("  Final loss: %.4f (PPL %.1f)" % (final_loss, math.exp(final_loss)))
    print("  Alpha: %.4f, Gate: %.4f" % (alpha_val, gate_val))

    os.makedirs('/data/biic/results', exist_ok=True)
    with open('/data/biic/results/phase5_relative_attn.json', 'w') as f:
        json.dump(log, f, indent=2)
    print("Saved: /data/biic/results/phase5_relative_attn.json")


def train_seg_eraser(args):
    """Experiment 5.6: Segmented Eraser"""
    print("=" * 60)
    print("Experiment 5.6: Segmented Eraser")
    print("=" * 60)
    print("Config: n_channels=%d, seq=%d, batch=%d, steps=%d" % (
        args.n_channels, args.seq_len, args.batch_size, args.n_steps))

    try:
        data_loader = WikiTextDataLoader(args.seq_len, args.batch_size)
    except Exception:
        os.environ['http_proxy'] = 'http://proxy.mornai.cn:7890'
        os.environ['https_proxy'] = 'http://proxy.mornai.cn:7890'
        data_loader = WikiTextDataLoader(args.seq_len, args.batch_size)

    model = BIICModelSegEraser(
        vocab_size=50257, n_channels=args.n_channels, n_layers=6, d_hidden=128
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("Params: %d" % n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    torch.manual_seed(42)
    np.random.seed(42)

    log = {"steps": [], "losses": [], "g2_norms_per_layer": []}
    t_start = time.time()

    for step in range(args.n_steps):
        if step < 200:
            lr = 3e-4 * step / 200
        else:
            progress = (step - 200) / (args.n_steps - 200)
            lr = 3e-4 * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        inputs, targets = data_loader.get_batch()
        optimizer.zero_grad(set_to_none=True)
        logits, g2_norms = model(inputs)
        loss = F.cross_entropy(logits.view(-1, 50257), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            ppl = math.exp(loss.item()) if loss.item() < 20 else float('inf')
            norms_str = " ".join(["%.3f" % n for n in g2_norms])
            print("Step %4d | loss=%.4f | ppl=%.1f | g2_norms=[%s]" % (
                step, loss.item(), ppl, norms_str), flush=True)
            log["steps"].append(step)
            log["losses"].append(loss.item())
            log["g2_norms_per_layer"].append(g2_norms)

    elapsed = time.time() - t_start
    final_loss = np.mean(log["losses"][-5:])
    log["final_loss"] = final_loss
    log["final_ppl"] = float(math.exp(final_loss))
    log["training_minutes"] = elapsed / 60
    log["params"] = n_params
    log["config"] = vars(args)

    print("\nDONE: Segmented Eraser")
    print("  Final loss: %.4f (PPL %.1f)" % (final_loss, math.exp(final_loss)))
    print("  Grade-2 norms per layer: %s" % str(g2_norms))

    os.makedirs('/data/biic/results', exist_ok=True)
    with open('/data/biic/results/phase5_segmented_eraser.json', 'w') as f:
        json.dump(log, f, indent=2)
    print("Saved: /data/biic/results/phase5_segmented_eraser.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True, choices=['rel_attn', 'seg_eraser'])
    parser.add_argument('--n_channels', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_steps', type=int, default=2000)
    args = parser.parse_args()

    if args.experiment == 'rel_attn':
        train_rel_attn(args)
    else:
        train_seg_eraser(args)
