"""
Phase 5.4: Cohesin vs Original on WikiText-103
Compare BIICLayer vs BIICLayerWithCohesin on real language data.
Hypothesis: Cohesin should help on real text (syntax/deps) even if toy task showed no gain.
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
        # Causal + window mask
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
        return mv_new


class BIICWikiTextModel(nn.Module):
    def __init__(self, vocab_size, n_channels, n_layers, d_hidden, use_cohesin=False, seq_len=128):
        super().__init__()
        self.use_cohesin = use_cohesin
        self.encoder = TokenToImmutableCore(vocab_size, n_channels)
        self.layers = nn.ModuleList([BIICLayer(n_channels, use_residual=True, use_eraser=True) for _ in range(n_layers)])
        if use_cohesin:
            self.cohesin = GradeAwareCohesin(n_channels, window_size=min(16, seq_len))
        self.decoder = AllGradeDecoder(vocab_size, n_channels, d_hidden)

    def forward(self, token_ids):
        mv = self.encoder(token_ids)
        mv_prior = mv.clone().detach()
        for i, layer in enumerate(self.layers):
            mv = layer(mv, mv_prior)
            if self.use_cohesin and i == len(self.layers) // 2:
                mv = self.cohesin(mv)
        logits = self.decoder(mv)
        return logits

    def get_gate_value(self):
        if self.use_cohesin:
            return torch.sigmoid(self.cohesin.gate).item()
        return 0.0


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


def train(args):
    print("=" * 60)
    print("Phase 5.4: BIIC WikiText-103 (%s)" % ("WITH Cohesin" if args.use_cohesin else "Original"))
    print("=" * 60)
    print("Device:", device)
    print("Config: n_channels=%d, seq=%d, batch=%d, steps=%d" % (
        args.n_channels, args.seq_len, args.batch_size, args.n_steps))

    os.makedirs('/data/biic/results', exist_ok=True)
    os.makedirs('/data/biic/logs', exist_ok=True)

    # Data
    try:
        data_loader = WikiTextDataLoader(args.seq_len, args.batch_size)
    except Exception:
        os.environ['http_proxy'] = 'http://proxy.mornai.cn:7890'
        os.environ['https_proxy'] = 'http://proxy.mornai.cn:7890'
        data_loader = WikiTextDataLoader(args.seq_len, args.batch_size)

    # Model
    model = BIICWikiTextModel(
        vocab_size=50257, n_channels=args.n_channels, n_layers=6,
        d_hidden=128, use_cohesin=args.use_cohesin, seq_len=args.seq_len
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print("Params: %d" % n_params)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.reset_peak_memory_stats()

    log = {"steps": [], "losses": [], "gates": [], "config": vars(args)}
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
        logits = model(inputs)
        loss = F.cross_entropy(logits.view(-1, 50257), targets.view(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()

        if step % 100 == 0:
            gate = model.get_gate_value()
            elapsed = time.time() - t_start
            speed = (step + 1) / elapsed if elapsed > 0 else 0
            ppl = math.exp(loss_val) if loss_val < 20 else float('inf')
            print("Step %4d | loss=%.4f | ppl=%.1f | gate=%.4f | %.2f step/s" % (
                step, loss_val, ppl, gate, speed), flush=True)
            log["steps"].append(step)
            log["losses"].append(loss_val)
            log["gates"].append(gate)

    # Final
    elapsed = time.time() - t_start
    final_loss = np.mean([l for l in log["losses"][-5:]])
    peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    print("\n" + "=" * 60)
    print("DONE: %s" % ("Cohesin" if args.use_cohesin else "Original"))
    print("  Final loss: %.4f (PPL %.1f)" % (final_loss, math.exp(final_loss)))
    print("  Gate: %.4f" % model.get_gate_value())
    print("  Time: %.1f min" % (elapsed / 60))
    print("  Peak VRAM: %.0f MB" % peak_mem)

    log["final_loss"] = final_loss
    log["final_ppl"] = float(math.exp(final_loss))
    log["final_gate"] = model.get_gate_value()
    log["training_minutes"] = elapsed / 60
    log["peak_vram_mb"] = peak_mem
    log["params"] = n_params

    suffix = "cohesin" if args.use_cohesin else "original"
    result_path = "/data/biic/results/phase5_wikitext_%s.json" % suffix
    with open(result_path, 'w') as f:
        json.dump(log, f, indent=2)
    print("Saved: %s" % result_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_channels', type=int, default=8)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_steps', type=int, default=2000)
    parser.add_argument('--use_cohesin', type=str, default='False')
    args = parser.parse_args()
    args.use_cohesin = args.use_cohesin.lower() in ('true', '1', 'yes')
    train(args)
