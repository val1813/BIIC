"""
Transformer Baseline - Extended to 50000 steps with resume support
Same architecture as original (d=512, 8 heads, 8 layers, ~52M params)
"""
import torch
import torch.nn as nn
import sys
import os
import json
import time
import random
import math
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True

CONFIG = {
    'vocab_size': 50257,
    'd_model': 512,
    'n_heads': 8,
    'n_layers': 8,
    'd_ff': 2048,
    'dropout': 0.1,
    'batch_size': 8,
    'seq_len': 256,
    'n_steps': 50000,
    'lr': 3e-4,
    'warmup_steps': 2000,
    'grad_clip': 1.0,
    'eval_every': 2000,
    'save_every': 10000,
    'log_every': 50,
    'save_dir': '/data/biic/transformer_baseline/checkpoints',
    'log_dir': '/data/biic/transformer_baseline/logs',
}


class TransformerLM(nn.Module):
    """Must match original checkpoint architecture: ln_f + output (not head)"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config['d_model']
        self.embed = nn.Embedding(config['vocab_size'], d)
        self.pos_embed = nn.Embedding(2048, d)
        self.embed_scale = math.sqrt(d)
        self.dropout = nn.Dropout(config['dropout'])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=config['n_heads'],
            dim_feedforward=config['d_ff'],
            dropout=config['dropout'], batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, config['n_layers'])
        self.ln_f = nn.LayerNorm(d)
        self.output = nn.Linear(d, config['vocab_size'], bias=False)
        self.output.weight = self.embed.weight  # weight tying

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device)
        h = self.embed(x) * self.embed_scale + self.pos_embed(pos)
        h = self.dropout(h)
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.transformer(h, mask=mask, is_causal=True)
        h = self.ln_f(h)
        return self.output(h)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


class WikiTextDataLoader:
    def __init__(self, seq_len, batch_size, split='train'):
        from transformers import GPT2Tokenizer
        print("Loading tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained('/data/biic/gpt2_tokenizer')
        self.seq_len = seq_len
        self.batch_size = batch_size
        print("Loading WikiText-103...")
        from datasets import load_dataset
        ds = load_dataset('wikitext', 'wikitext-103-v1', split=split,
                         cache_dir='/data/biic/hf_cache')
        text = " ".join([t for t in ds['text'] if len(t.strip()) > 0])
        print("Tokenizing (this takes a minute)...")
        self.tokens = self.tokenizer.encode(text)
        print("Total tokens: %d" % len(self.tokens))
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        self.pos = 0

    def get_batch(self):
        B, L = self.batch_size, self.seq_len
        if self.pos + B * (L + 1) > len(self.tokens):
            self.pos = 0
        batch_inputs = []
        batch_targets = []
        for _ in range(B):
            start = self.pos
            chunk = self.tokens[start:start + L + 1]
            batch_inputs.append(chunk[:-1])
            batch_targets.append(chunk[1:])
            self.pos += L
        return torch.stack(batch_inputs).to(device), torch.stack(batch_targets).to(device)


def get_lr(step, warmup, max_lr, total_steps):
    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / (total_steps - warmup)
    return max_lr * 0.5 * (1 + math.cos(math.pi * progress))


if __name__ == '__main__':
    print("Transformer Baseline - Extended 50000 steps")
    print("=" * 60)

    config = CONFIG
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # Data
    try:
        data_loader = WikiTextDataLoader(config['seq_len'], config['batch_size'])
    except Exception:
        os.environ['http_proxy'] = 'http://proxy.mornai.cn:7890'
        os.environ['https_proxy'] = 'http://proxy.mornai.cn:7890'
        data_loader = WikiTextDataLoader(config['seq_len'], config['batch_size'])

    # Model
    model = TransformerLM(config).to(device)
    print("Params: {:,}".format(model.count_params()))

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()

    # Resume from checkpoint if available
    start_step = 0
    loss_history = []
    ckpts = sorted([f for f in os.listdir(config['save_dir']) if f.endswith('.pt')])
    if ckpts:
        latest = os.path.join(config['save_dir'], ckpts[-1])
        print("Loading checkpoint: %s" % latest)
        ckpt = torch.load(latest, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        start_step = ckpt['step'] + 1
        loss_history = ckpt.get('loss_history', [])
        print("Resuming from step %d" % start_step)

    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    torch.cuda.reset_peak_memory_stats()

    best_loss = min(loss_history[-200:]) if len(loss_history) >= 200 else float('inf')
    t_start = time.time()

    print("Training %d -> %d steps..." % (start_step, config['n_steps']), flush=True)

    for step in range(start_step, config['n_steps']):
        lr = get_lr(step, config['warmup_steps'], config['lr'], config['n_steps'])
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        inputs, targets = data_loader.get_batch()
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            logits = model(inputs)
            loss = nn.CrossEntropyLoss()(logits.reshape(-1, config['vocab_size']), targets.reshape(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        scaler.step(optimizer)
        scaler.update()

        loss_val = loss.item()
        loss_history.append(loss_val)

        if step % config['log_every'] == 0:
            elapsed = time.time() - t_start
            steps_done = step - start_step + 1
            steps_per_sec = steps_done / elapsed if elapsed > 0 else 0
            mem = torch.cuda.memory_allocated() / 1e6
            ppl = math.exp(loss_val) if loss_val < 20 else float('inf')
            eta_hours = (config['n_steps'] - step) / steps_per_sec / 3600 if steps_per_sec > 0 else 0
            print("Step %5d | loss=%.4f | ppl=%.1f | lr=%.2e | %.1f step/s | mem=%.0fMB | ETA=%.1fh" % (
                step, loss_val, ppl, lr, steps_per_sec, mem, eta_hours), flush=True)

        if step > 0 and step % config['eval_every'] == 0:
            avg_recent = np.mean(loss_history[-500:])
            ppl_recent = np.exp(avg_recent)
            print("  [Eval] step=%d, avg_loss=%.4f, PPL=%.1f" % (step, avg_recent, ppl_recent), flush=True)
            if avg_recent < best_loss:
                best_loss = avg_recent
                print("  [Eval] New best: loss=%.4f, PPL=%.1f" % (best_loss, np.exp(best_loss)), flush=True)

        if step > 0 and step % config['save_every'] == 0:
            sp = os.path.join(config['save_dir'], "step_%d.pt" % step)
            torch.save({'step': step, 'model_state': model.state_dict(),
                       'loss_history': loss_history[-10000:], 'config': config}, sp)
            print("  [Save] %s" % sp, flush=True)

    # Final save
    elapsed = time.time() - t_start
    final_loss = np.mean(loss_history[-500:])
    final_ppl = np.exp(final_loss)
    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("  Total time: %.1f hours" % (elapsed / 3600))
    print("  Params: {:,}".format(model.count_params()))
    print("  Final loss: %.4f" % final_loss)
    print("  Final PPL: %.1f" % final_ppl)
    print("  Best loss: %.4f (PPL %.1f)" % (best_loss, np.exp(best_loss)))
    print("  Peak VRAM: %.0f MB" % peak_mem)

    # Save final checkpoint
    torch.save({'step': config['n_steps'], 'model_state': model.state_dict(),
               'loss_history': loss_history[-10000:], 'config': config},
              os.path.join(config['save_dir'], 'final.pt'))

    # Save results JSON
    results = {
        'model': 'transformer_baseline',
        'params': model.count_params(),
        'final_loss': final_loss,
        'final_ppl': final_ppl,
        'best_loss': best_loss,
        'best_ppl': float(np.exp(best_loss)),
        'total_steps': config['n_steps'],
        'peak_vram_mb': peak_mem,
        'training_hours': elapsed / 3600,
        'config': {k: v for k, v in config.items() if not k.endswith('_dir')},
    }
    os.makedirs('/data/biic/results', exist_ok=True)
    with open('/data/biic/results/phase4_transformer_baseline_full.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Results saved to /data/biic/results/phase4_transformer_baseline_full.json")
