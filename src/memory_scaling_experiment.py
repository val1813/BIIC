"""
显存对比实验：BIIC vs Transformer 在不同seq_len下的峰值显存
目标：证明BIIC没有KV Cache，显存不随序列长度线性增长
"""

import torch
import torch.nn as nn
import sys
import os
import json
import time
import math
import numpy as np

sys.path.insert(0, '/data/biic')
sys.path.insert(0, '/data/biic/phase2')

from clifford_cl41 import GRADE_SLICES, N_BLADES
from token_to_ic import TokenToImmutableCore
from all_grade_decoder import AllGradeDecoder
from mutable_state import BIICLayer

device = torch.device('cuda')
torch.backends.cudnn.benchmark = True

SEQ_LENS = [256, 512, 1024, 2048]
BATCH_SIZE = 4
VOCAB_SIZE = 50257


# ═══════════════════════════════════════════════════════════
# BIIC Model (same as v0.2 but variable seq_len)
# ═══════════════════════════════════════════════════════════

class PositionEncoding(nn.Module):
    def __init__(self, max_len=4096, n_channels=64):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, n_channels * 10)
        self.n_channels = n_channels

    def forward(self, mv):
        B, L, C, D = mv.shape
        pos_ids = torch.arange(L, device=mv.device)
        pos_vec = self.pos_embed(pos_ids).reshape(L, C, 10) * 0.1
        g2_start, g2_end = GRADE_SLICES[2]
        mv_out = mv.clone()
        mv_out[:, :, :, g2_start:g2_end] = mv[:, :, :, g2_start:g2_end] + pos_vec.unsqueeze(0)
        return mv_out


class SimpleTokenMixer(nn.Module):
    def __init__(self, n_channels, kernel_size=16):
        super().__init__()
        dim = n_channels * N_BLADES
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size-1, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, mv):
        B, L, C, D = mv.shape
        x = mv.reshape(B, L, C * D)
        x_normed = self.norm(x)
        x_t = x_normed.transpose(1, 2)
        conv_out = self.conv(x_t)[:, :, :L]
        conv_out = conv_out.transpose(1, 2)
        gate = torch.sigmoid(self.gate(x_normed))
        mixed = x + gate * conv_out
        return mixed.reshape(B, L, C, D)


class BIICLMBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.mixer = SimpleTokenMixer(n_channels, kernel_size=16)
        self.biic_layer = BIICLayer(n_channels, use_residual=True, use_eraser=True)

    def forward(self, mv, mv_prior):
        mv = self.mixer(mv)
        mv = self.biic_layer(mv, mv_prior)
        return mv


class BIICModel(nn.Module):
    def __init__(self, n_channels=64, n_layers=12):
        super().__init__()
        self.encoder = TokenToImmutableCore(VOCAB_SIZE, n_channels)
        self.pos_enc = PositionEncoding(max_len=4096, n_channels=n_channels)
        self.blocks = nn.ModuleList([BIICLMBlock(n_channels) for _ in range(n_layers)])
        self.decoder = AllGradeDecoder(VOCAB_SIZE, n_channels, 256)

    def forward(self, token_ids):
        mv = self.encoder(token_ids)
        mv = self.pos_enc(mv)
        mv_prior = mv.clone().detach()
        for block in self.blocks:
            mv = block(mv, mv_prior)
        logits = self.decoder(mv)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════
# Transformer Model (same param count)
# ═══════════════════════════════════════════════════════════

class TransformerModel(nn.Module):
    def __init__(self, d_model=512, n_heads=8, n_layers=8, d_ff=2048):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_embed = nn.Embedding(4096, d_model)
        self.embed_scale = math.sqrt(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.1, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.output.weight = self.embed.weight

    def forward(self, token_ids):
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device)
        x = self.embed(token_ids) * self.embed_scale + self.pos_embed(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=token_ids.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.output(x)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════
# 显存测量
# ═══════════════════════════════════════════════════════════

def measure_memory(model, seq_len, batch_size, n_forward=3):
    """测量模型在给定seq_len下的峰值显存"""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    token_ids = torch.randint(0, VOCAB_SIZE, (batch_size, seq_len)).to(device)

    # Warmup
    with torch.no_grad():
        try:
            _ = model(token_ids)
        except RuntimeError as e:
            if "out of memory" in str(e):
                torch.cuda.empty_cache()
                return None  # OOM
            raise

    torch.cuda.reset_peak_memory_stats()

    # Measure
    with torch.no_grad():
        for _ in range(n_forward):
            _ = model(token_ids)

    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    return peak_mb


def run_experiment():
    print("=" * 60)
    print("Memory Scaling Experiment: BIIC vs Transformer")
    print("=" * 60)
    print("Device:", torch.cuda.get_device_name(0))
    print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
    print("Batch size:", BATCH_SIZE)
    print("Seq lengths:", SEQ_LENS)
    print()

    results = {"biic": {}, "transformer": {}, "config": {
        "batch_size": BATCH_SIZE, "seq_lens": SEQ_LENS,
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
    }}

    # Build models
    print("Building BIIC model (n_channels=64, n_layers=12)...")
    biic = BIICModel(n_channels=64, n_layers=12).to(device).eval()
    biic_params = biic.count_params()
    print("  BIIC params: {:,}".format(biic_params))
    results["config"]["biic_params"] = biic_params

    print("Building Transformer model (d=512, heads=8, layers=8)...")
    transformer = TransformerModel(d_model=512, n_heads=8, n_layers=8, d_ff=2048).to(device).eval()
    tf_params = transformer.count_params()
    print("  Transformer params: {:,}".format(tf_params))
    results["config"]["transformer_params"] = tf_params
    print()

    # Measure BIIC
    print("--- BIIC Memory ---")
    for seq_len in SEQ_LENS:
        torch.cuda.empty_cache()
        mem = measure_memory(biic, seq_len, BATCH_SIZE)
        if mem is not None:
            results["biic"][str(seq_len)] = mem
            print("  seq_len=%4d: %.0f MB" % (seq_len, mem))
        else:
            results["biic"][str(seq_len)] = "OOM"
            print("  seq_len=%4d: OOM" % seq_len)

    # Free BIIC
    del biic
    torch.cuda.empty_cache()

    # Measure Transformer
    print("\n--- Transformer Memory ---")
    for seq_len in SEQ_LENS:
        torch.cuda.empty_cache()
        mem = measure_memory(transformer, seq_len, BATCH_SIZE)
        if mem is not None:
            results["transformer"][str(seq_len)] = mem
            print("  seq_len=%4d: %.0f MB" % (seq_len, mem))
        else:
            results["transformer"][str(seq_len)] = "OOM"
            print("  seq_len=%4d: OOM" % seq_len)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("\n  seq_len | BIIC (MB) | Transformer (MB) | Ratio")
    print("  --------|-----------|------------------|------")
    for seq_len in SEQ_LENS:
        b = results["biic"].get(str(seq_len), "OOM")
        t = results["transformer"].get(str(seq_len), "OOM")
        if isinstance(b, (int, float)) and isinstance(t, (int, float)):
            ratio = t / b
            print("  %7d | %9.0f | %16.0f | %.2fx" % (seq_len, b, t, ratio))
        else:
            print("  %7d | %9s | %16s | -" % (seq_len, str(b), str(t)))

    # Save
    os.makedirs('/data/biic/memory_experiment', exist_ok=True)
    save_path = '/data/biic/memory_experiment/results.json'
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved:", save_path)

    return results


if __name__ == '__main__':
    run_experiment()
