"""
Phase 5 Unified Experiment Script
Supports all combinations: Cohesin, Relative Attention, Segmented Eraser
Configurable: n_channels, seq_len, n_layers, n_steps, save_path

Usage examples:
  # Relative attention 10k steps
  python biic_with_cohesin.py --use_wikitext True --use_cohesin True --use_relative_attn True --n_steps 10000 --save_path /data/biic/results/p5_rel_attn_10k.json

  # Original baseline (no mechanisms)
  python biic_with_cohesin.py --use_wikitext True --use_cohesin False --use_relative_attn False --n_steps 10000 --save_path /data/biic/results/p5_original_10k.json
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


class GradeAwareCohesin(nn.Module):
    """Cohesin: cross-token attention on grade-0, updates grade-2"""
    def __init__(self, n_channels, window_size=16):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        g2_dim = 10 * n_channels
        self.proj_q = nn.Linear(n_channels, n_channels)
        self.proj_k = nn.Linear(n_channels, n_channels)
        self.proj_v = nn.Linear(g2_dim, g2_dim)
        self.proj_out = nn.Linear(g2_dim, g2_dim)
        self.gate = nn.Parameter(torch.zeros(1))
        self.ln = nn.LayerNorm(g2_dim)

    def forward(self, mv):
        B, L, C, D = mv.shape
        g0_start, g0_end = GRADE_SLICES[0]
        g2_start, g2_end = GRADE_SLICES[2]
        grade0 = mv[..., g0_start:g0_end].squeeze(-1)
        grade2 = mv[..., g2_start:g2_end]
        g0_flat = grade0.reshape(B, L, -1)
        g2_flat = grade2.reshape(B, L, -1)
        Q = self.proj_q(g0_flat)
        K = self.proj_k(g0_flat)
        V = self.proj_v(g2_flat)
        scale = math.sqrt(Q.shape[-1])
        scores = torch.bmm(Q, K.transpose(1, 2)) / scale
        mask = torch.ones(L, L, device=mv.device, dtype=torch.bool)
        for i in range(L):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i+1] = False
        scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(attn, V)
        g2_updated = g2_flat + torch.sigmoid(self.gate) * self.proj_out(self.ln(context))
        g2_updated = torch.clamp(g2_updated, -10.0, 10.0)
        mv_new = mv.clone()
        mv_new[..., g2_start:g2_end] = g2_updated.reshape(B, L, C, 10)
        return mv_new, torch.sigmoid(self.gate).item()


class RelativeInvariantAttention(nn.Module):
    """Attention using grade-0 (semantic) + grade-2 (relative relation) scores."""
    def __init__(self, n_channels, window_size=16):
        super().__init__()
        self.n_channels = n_channels
        self.window_size = window_size
        self.proj_q = nn.Linear(n_channels, n_channels)
        self.proj_k = nn.Linear(n_channels, n_channels)
        g2_dim = 10 * n_channels
        self.proj_v = nn.Linear(g2_dim, g2_dim)
        self.proj_out = nn.Linear(g2_dim, g2_dim)
        self.ln = nn.LayerNorm(g2_dim)
        self.alpha = nn.Parameter(torch.tensor([-4.0]))
        self.gate = nn.Parameter(torch.tensor([-4.0]))

    def forward(self, mv):
        B, L, C, D = mv.shape
        g0_start, g0_end = GRADE_SLICES[0]
        g2_start, g2_end = GRADE_SLICES[2]
        grade0 = mv[..., g0_start:g0_end].squeeze(-1)
        grade2 = mv[..., g2_start:g2_end]
        g0_flat = grade0.reshape(B, L, -1)
        g2_flat = grade2.reshape(B, L, -1)

        Q = self.proj_q(g0_flat)
        K = self.proj_k(g0_flat)
        scale = math.sqrt(Q.shape[-1])
        score_sem = torch.bmm(Q, K.transpose(1, 2)) / scale

        g2_small = g2_flat[..., :8 * self.n_channels]
        score_rel = torch.bmm(g2_small, g2_small.transpose(1, 2)) / (8 * self.n_channels) ** 0.5

        alpha_val = torch.sigmoid(self.alpha)
        score = score_sem + alpha_val * score_rel

        mask = torch.ones(L, L, device=mv.device, dtype=torch.bool)
        for i in range(L):
            start = max(0, i - self.window_size + 1)
            mask[i, start:i+1] = False
        score = score.masked_fill(mask.unsqueeze(0), float('-inf'))

        attn = F.softmax(score, dim=-1)
        V = self.proj_v(g2_flat)
        context = torch.bmm(attn, V)

        gate_val = torch.sigmoid(self.gate)
        g2_updated = g2_flat + gate_val * self.proj_out(self.ln(context))
        g2_updated = torch.clamp(g2_updated, -10.0, 10.0)

        mv_new = mv.clone()
        mv_new[..., g2_start:g2_end] = g2_updated.reshape(B, L, C, 10)
        return mv_new, alpha_val.item(), gate_val.item()


class SegmentedEraser(nn.Module):
    """Segmented Eraser: erase grade-1,3,4 every 4 layers, preserve grade-0,2"""
    def __init__(self, n_channels):
        super().__init__()
        self.eraser = GradeAwareEraser(n_channels)

    def apply(self, mv, mv_prior, layer_idx, n_layers):
        if (layer_idx + 1) % 4 == 0 and layer_idx < n_layers - 1:
            g1_start, g1_end = GRADE_SLICES[1]
            g3_start, g3_end = GRADE_SLICES[3]
            g4_start, g4_end = GRADE_SLICES[4]
            decay = torch.sigmoid(self.eraser.decay_logits.mean())
            for gs, ge in [(g1_start, g1_end), (g3_start, g3_end), (g4_start, g4_end)]:
                mv_slice = mv[..., gs:ge]
                prior_slice = mv_prior[..., gs:ge]
                mv = mv.clone()
                mv[..., gs:ge] = mv_slice * (1 - decay) + prior_slice * decay
        return mv


class UnifiedBIICModel(nn.Module):
    """Unified model supporting all mechanism combinations"""
    def __init__(self, vocab_size, n_channels, n_layers, d_hidden, seq_len=128,
                 use_cohesin=False, use_relative_attn=False, use_seg_eraser=False):
        super().__init__()
        self.use_cohesin = use_cohesin
        self.use_relative_attn = use_relative_attn
        self.use_seg_eraser = use_seg_eraser
        self.n_layers = n_layers

        self.encoder = TokenToImmutableCore(vocab_size, n_channels)
        # If seg_eraser, layers don't have built-in eraser
        use_layer_eraser = not use_seg_eraser
        self.layers = nn.ModuleList([
            BIICLayer(n_channels, use_residual=True, use_eraser=use_layer_eraser)
            for _ in range(n_layers)
        ])

        if use_cohesin:
            self.cohesin = GradeAwareCohesin(n_channels, window_size=min(16, seq_len))
        if use_relative_attn:
            self.rel_attn = RelativeInvariantAttention(n_channels, window_size=min(16, seq_len))
        if use_seg_eraser:
            self.seg_eraser = SegmentedEraser(n_channels)

        self.decoder = AllGradeDecoder(vocab_size, n_channels, d_hidden)

    def forward(self, token_ids):
        mv = self.encoder(token_ids)
        mv_prior = mv.clone().detach()

        alpha_val, gate_val, cohesin_gate = 0.0, 0.0, 0.0
        g2_norms = []

        for i, layer in enumerate(self.layers):
            mv = layer(mv, mv_prior)

            # Record grade-2 norms
            g2_start, g2_end = GRADE_SLICES[2]
            g2_norm = mv[..., g2_start:g2_end].norm(dim=-1).mean().item()
            g2_norms.append(g2_norm)

            # Segmented eraser at block boundaries
            if self.use_seg_eraser:
                mv = self.seg_eraser.apply(mv, mv_prior, i, self.n_layers)

            # Cohesin at middle layer
            if self.use_cohesin and i == self.n_layers // 2:
                mv, cohesin_gate = self.cohesin(mv)

            # Relative attention at middle layer
            if self.use_relative_attn and i == self.n_layers // 2:
                mv, alpha_val, gate_val = self.rel_attn(mv)

        logits = self.decoder(mv)
        return logits, {
            'alpha': alpha_val,
            'gate': gate_val,
            'cohesin_gate': cohesin_gate,
            'g2_norms': g2_norms,
        }


class WikiTextDataLoader:
    def __init__(self, seq_len, batch_size, split='train'):
        import os as _os
        self.seq_len = seq_len
        self.batch_size = batch_size
        pt_path = '/data/biic/hf_cache/wikitext103_tokens.pt'
        if _os.path.exists(pt_path):
            print("Loading pre-tokenized data from %s..." % pt_path)
            self.tokens = torch.load(pt_path, weights_only=True)
            print("Total tokens: %d" % len(self.tokens))
        else:
            from transformers import GPT2Tokenizer
            print("Loading tokenizer...")
            self.tokenizer = GPT2Tokenizer.from_pretrained('/data/biic/gpt2_tokenizer')
            print("Loading WikiText-103 %s..." % split)
            from datasets import load_dataset
            try:
                ds = load_dataset('wikitext', 'wikitext-103-v1', split=split,
                                 cache_dir='/data/biic/hf_cache')
            except Exception:
                _os.environ['http_proxy'] = 'http://proxy.mornai.cn:7890'
                _os.environ['https_proxy'] = 'http://proxy.mornai.cn:7890'
                ds = load_dataset('wikitext', 'wikitext-103-v1', split=split,
                                 cache_dir='/data/biic/hf_cache')
            text = " ".join([t for t in ds['text'] if len(t.strip()) > 20])
            print("Tokenizing...")
            self.tokens = self.tokenizer.encode(text)
            print("Total tokens: %d" % len(self.tokens))
            self.tokens = torch.tensor(self.tokens, dtype=torch.long)
            torch.save(self.tokens, pt_path)
            print("Saved tokens to %s" % pt_path)
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


def train(args):
    print("=" * 60)
    mechanisms = []
    if args.use_cohesin: mechanisms.append("Cohesin")
    if args.use_relative_attn: mechanisms.append("RelAttn")
    if args.use_seg_eraser: mechanisms.append("SegEraser")
    if not mechanisms: mechanisms.append("Original")
    title = "Phase 5: " + "+".join(mechanisms)
    print(title)
    print("=" * 60)
    print("Device:", device)
    print("Config: n_channels=%d, seq=%d, batch=%d, steps=%d, layers=%d" % (
        args.n_channels, args.seq_len, args.batch_size, args.n_steps, args.n_layers))

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    os.makedirs('/data/biic/logs', exist_ok=True)

    # Data
    try:
        data_loader = WikiTextDataLoader(args.seq_len, args.batch_size)
    except Exception:
        os.environ['http_proxy'] = 'http://proxy.mornai.cn:7890'
        os.environ['https_proxy'] = 'http://proxy.mornai.cn:7890'
        data_loader = WikiTextDataLoader(args.seq_len, args.batch_size)

    # Model
    model = UnifiedBIICModel(
        vocab_size=50257, n_channels=args.n_channels, n_layers=args.n_layers,
        d_hidden=128, seq_len=args.seq_len,
        use_cohesin=args.use_cohesin,
        use_relative_attn=args.use_relative_attn,
        use_seg_eraser=args.use_seg_eraser,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("Params: %d" % n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.reset_peak_memory_stats()

    log = {
        "experiment": title,
        "config": {
            "n_channels": args.n_channels,
            "seq_len": args.seq_len,
            "batch_size": args.batch_size,
            "n_steps": args.n_steps,
            "n_layers": args.n_layers,
            "use_cohesin": args.use_cohesin,
            "use_relative_attn": args.use_relative_attn,
            "use_seg_eraser": args.use_seg_eraser,
        },
        "params": n_params,
        "steps": [],
        "losses": [],
        "alphas": [],
        "gates": [],
        "cohesin_gates": [],
        "g2_norms_per_layer": [],
    }
    t_start = time.time()

    for step in range(args.n_steps):
        # LR warmup + cosine
        if step < 200:
            lr = 3e-4 * step / 200
        else:
            progress = (step - 200) / (args.n_steps - 200)
            lr = 3e-4 * 0.5 * (1 + math.cos(math.pi * progress))
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        inputs, targets = data_loader.get_batch()
        optimizer.zero_grad(set_to_none=True)
        logits, info = model(inputs)
        loss = F.cross_entropy(logits.view(-1, 50257), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 100 == 0:
            ppl = math.exp(loss.item()) if loss.item() < 20 else float('inf')
            elapsed = time.time() - t_start
            speed = (step + 1) / elapsed if elapsed > 0 else 0
            extra = ""
            if args.use_relative_attn:
                extra += " alpha=%.4f gate=%.4f" % (info['alpha'], info['gate'])
            if args.use_cohesin:
                extra += " coh_gate=%.4f" % info['cohesin_gate']
            if args.use_seg_eraser:
                norms_str = " ".join(["%.2f" % n for n in info['g2_norms']])
                extra += " g2=[%s]" % norms_str
            print("Step %5d | loss=%.4f | ppl=%.1f |%s | %.2f step/s" % (
                step, loss.item(), ppl, extra, speed), flush=True)

            log["steps"].append(step)
            log["losses"].append(loss.item())
            log["alphas"].append(info['alpha'])
            log["gates"].append(info['gate'])
            log["cohesin_gates"].append(info['cohesin_gate'])
            log["g2_norms_per_layer"].append(info['g2_norms'])

    # Final summary
    elapsed = time.time() - t_start
    final_loss = float(np.mean(log["losses"][-5:]))
    final_ppl = float(math.exp(final_loss)) if final_loss < 20 else float('inf')
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    log["final_loss"] = final_loss
    log["final_ppl"] = final_ppl
    log["final_alpha"] = info['alpha']
    log["final_gate"] = info['gate']
    log["final_cohesin_gate"] = info['cohesin_gate']
    log["training_minutes"] = elapsed / 60
    log["peak_vram_mb"] = peak_mem

    print("\n" + "=" * 60)
    print("DONE: %s" % title)
    print("  Final loss: %.4f (PPL %.1f)" % (final_loss, final_ppl))
    print("  Time: %.1f min, Peak VRAM: %.0f MB" % (elapsed / 60, peak_mem))
    if args.use_relative_attn:
        print("  Alpha: %.4f, Gate: %.4f" % (info['alpha'], info['gate']))
    if args.use_cohesin:
        print("  Cohesin gate: %.4f" % info['cohesin_gate'])

    with open(args.save_path, 'w', encoding='utf-8') as f:
        json.dump(log, f, indent=2)
    print("Saved: %s" % args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BIIC Phase 5 Unified Experiment')
    parser.add_argument('--use_wikitext', type=str, default='True')
    parser.add_argument('--use_cohesin', type=str, default='False')
    parser.add_argument('--use_relative_attn', type=str, default='False')
    parser.add_argument('--use_seg_eraser', type=str, default='False')
    parser.add_argument('--n_channels', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_steps', type=int, default=2000)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--save_path', type=str, default='/data/biic/results/p5_experiment.json')
    args = parser.parse_args()

    # Parse boolean args
    args.use_wikitext = args.use_wikitext.lower() in ('true', '1', 'yes')
    args.use_cohesin = args.use_cohesin.lower() in ('true', '1', 'yes')
    args.use_relative_attn = args.use_relative_attn.lower() in ('true', '1', 'yes')
    args.use_seg_eraser = args.use_seg_eraser.lower() in ('true', '1', 'yes')

    train(args)
