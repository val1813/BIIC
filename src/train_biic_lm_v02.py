"""
BIIC Language Model v0.2 - Optimized Version
优化点：
1. torch.cuda.amp.autocast 混合精度 (预计提速30-50%)
2. torch.backends.cudnn.benchmark = True
3. 目标10000步（足够证明收敛）
4. 数据预加载到内存避免重复tokenize
"""

import torch
import torch.nn as nn
import sys
import os
import json
import time
import random
import numpy as np

sys.path.insert(0, '/data/biic')
sys.path.insert(0, '/data/biic/phase2')

from clifford_cl41 import GRADE_SLICES, N_BLADES
from rotor_utils import sandwich_product, exp_bivector, normalize_rotor
from eraser_ops import GradeAwareEraser
from token_to_ic import TokenToImmutableCore
from all_grade_decoder import AllGradeDecoder
from mutable_state import BIICLayer

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIG = {
    'vocab_size': 50257,
    'n_channels': 64,
    'n_layers': 12,
    'd_hidden': 256,
    'batch_size': 8,
    'seq_len': 256,
    'n_steps': 10000,
    'lr': 3e-4,
    'warmup_steps': 1000,
    'grad_clip': 1.0,
    'eval_every': 1000,
    'save_every': 5000,
    'log_every': 50,
    'eraser_init': 0.0,
    'save_dir': '/data/biic/biic_lm_v02/checkpoints',
    'log_dir': '/data/biic/biic_lm_v02/logs',
}


class PositionEncoding(nn.Module):
    def __init__(self, max_len=2048, n_channels=64):
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
        self.kernel_size = kernel_size

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


class BIICLanguageModelV02(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        C = config['n_channels']
        self.encoder = TokenToImmutableCore(config['vocab_size'], C)
        self.pos_enc = PositionEncoding(max_len=2048, n_channels=C)
        self.blocks = nn.ModuleList([BIICLMBlock(C) for _ in range(config['n_layers'])])
        for block in self.blocks:
            if hasattr(block.biic_layer.mutable_layer, 'eraser'):
                with torch.no_grad():
                    block.biic_layer.mutable_layer.eraser.decay_logits.fill_(config['eraser_init'])
        self.decoder = AllGradeDecoder(config['vocab_size'], C, config['d_hidden'])

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


class WikiTextDataLoader:
    def __init__(self, seq_len, batch_size, split='train'):
        from transformers import GPT2Tokenizer
        print("Loading tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.seq_len = seq_len
        self.batch_size = batch_size

        print("Loading WikiText-103 %s split..." % split)
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-103-v1', split=split)

        print("Tokenizing (one-time)...")
        all_text = "\n".join([t for t in ds['text'] if len(t.strip()) > 20])
        self.tokens = self.tokenizer.encode(all_text)
        print("Total tokens: {:,}".format(len(self.tokens)))

        # 预切分为chunks避免每次get_batch时切片
        self.chunks = []
        pos = 0
        while pos + seq_len + 1 <= len(self.tokens):
            self.chunks.append(self.tokens[pos:pos + seq_len + 1])
            pos += seq_len
        print("Total chunks: {:,}".format(len(self.chunks)))
        self.idx = 0

    def get_batch(self):
        batch = []
        for _ in range(self.batch_size):
            if self.idx >= len(self.chunks):
                self.idx = 0
                random.shuffle(self.chunks)
            batch.append(self.chunks[self.idx])
            self.idx += 1
        batch = torch.tensor(batch, dtype=torch.long, device=device)
        return batch[:, :-1], batch[:, 1:]


def get_lr(step, warmup_steps, max_lr, total_steps):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return max_lr * 0.5 * (1 + np.cos(np.pi * progress))


def train():
    print("=" * 60)
    print("BIIC Language Model v0.2 - Optimized")
    print("=" * 60)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
        print("cudnn.benchmark:", torch.backends.cudnn.benchmark)
        print("AMP: enabled")

    config = CONFIG
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # 数据
    try:
        data_loader = WikiTextDataLoader(config['seq_len'], config['batch_size'], split='train')
    except Exception as e:
        print("Download failed, using proxy...")
        os.environ['http_proxy'] = 'http://proxy.mornai.cn:7890'
        os.environ['https_proxy'] = 'http://proxy.mornai.cn:7890'
        data_loader = WikiTextDataLoader(config['seq_len'], config['batch_size'], split='train')

    # 模型
    model = BIICLanguageModelV02(config).to(device)
    n_params = model.count_params()
    print("Params: {:,}".format(n_params))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    loss_history = []
    best_loss = float('inf')
    t_start = time.time()

    print("\nTraining started (%d steps)...\n" % config['n_steps'], flush=True)

    for step in range(config['n_steps']):
        lr = get_lr(step, config['warmup_steps'], config['lr'], config['n_steps'])
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        inputs, targets = data_loader.get_batch()

        optimizer.zero_grad(set_to_none=True)

        # AMP混合精度
        with torch.cuda.amp.autocast():
            logits = model(inputs)
            loss = nn.CrossEntropyLoss()(
                logits.reshape(-1, config['vocab_size']),
                targets.reshape(-1)
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % config['log_every'] == 0:
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            ppl = np.exp(loss_val) if loss_val < 20 else float('inf')
            eta_hours = (config['n_steps'] - step) / steps_per_sec / 3600 if steps_per_sec > 0 else 0
            print("Step %5d | loss=%.4f | ppl=%.1f | lr=%.2e | %.2f step/s | mem=%.0fMB | ETA=%.1fh" % (
                step, loss_val, ppl, lr, steps_per_sec, mem, eta_hours), flush=True)

        if step > 0 and step % config['eval_every'] == 0:
            avg_recent = np.mean(loss_history[-200:])
            ppl_recent = np.exp(avg_recent)
            print("\n  [Eval] step=%d, avg_loss=%.4f, PPL=%.1f" % (step, avg_recent, ppl_recent), flush=True)
            if avg_recent < best_loss:
                best_loss = avg_recent
                print("  [Eval] New best: loss=%.4f, PPL=%.1f" % (best_loss, np.exp(best_loss)), flush=True)
            print("", flush=True)

        if step > 0 and step % config['save_every'] == 0:
            ckpt_path = os.path.join(config['save_dir'], "step_%d.pt" % step)
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss_history': loss_history[-5000:],
                'config': config,
            }, ckpt_path)
            print("  [Save] %s" % ckpt_path, flush=True)

    # Final
    elapsed = time.time() - t_start
    final_path = os.path.join(config['save_dir'], "final.pt")
    torch.save({
        'step': config['n_steps'],
        'model_state': model.state_dict(),
        'loss_history': loss_history,
        'config': config,
    }, final_path)

    final_loss = np.mean(loss_history[-200:])
    final_ppl = np.exp(final_loss)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("  Total time: %.1f hours" % (elapsed / 3600))
    print("  Params: {:,}".format(n_params))
    print("  Final loss: %.4f" % final_loss)
    print("  Final PPL: %.1f" % final_ppl)
    print("  Best loss: %.4f (PPL %.1f)" % (best_loss, np.exp(best_loss)))
    if torch.cuda.is_available():
        print("  Peak VRAM: %.0f MB" % (torch.cuda.max_memory_allocated() / 1e6))

    log_path = os.path.join(config['log_dir'], "train_log_optimized.json")
    with open(log_path, 'w') as f:
        json.dump({
            'config': config,
            'loss_history_sampled': loss_history[::50],
            'total_time_hours': elapsed / 3600,
            'final_loss': float(final_loss),
            'final_ppl': float(final_ppl),
            'best_loss': float(best_loss),
            'params': n_params,
            'peak_vram_mb': torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
            'optimizations': ['AMP', 'cudnn.benchmark', 'pre-chunked data', 'set_to_none=True'],
        }, f, indent=2)
    print("  Log: %s" % log_path)


if __name__ == '__main__':
    train()
