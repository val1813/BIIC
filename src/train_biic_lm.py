"""
BIIC Language Model v0.1 - Proof of Concept
用BIIC多向量替代token embedding的最小语言模型
目标：证明loss能下降，next-token prediction能跑通
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ═══════════════════════════════════════════════════════════
# 配置
# ═══════════════════════════════════════════════════════════

CONFIG = {
    'vocab_size': 50257,
    'n_channels': 32,
    'n_layers': 6,
    'd_hidden': 128,
    'batch_size': 16,
    'seq_len': 128,
    'n_steps': 10000,
    'lr': 3e-4,
    'warmup_steps': 500,
    'grad_clip': 1.0,
    'eval_every': 500,
    'save_every': 2000,
    'log_every': 10,
    'eraser_init': 0.0,
    'save_dir': '/data/biic/biic_lm/checkpoints',
    'log_dir': '/data/biic/biic_lm/logs',
}


# ═══════════════════════════════════════════════════════════
# 模型
# ═══════════════════════════════════════════════════════════

class PositionEncoding(nn.Module):
    """简单位置编码：注入到grade-2分量"""
    def __init__(self, max_len=2048, n_channels=32):
        super().__init__()
        self.pos_embed = nn.Embedding(max_len, n_channels * 10)  # grade-2有10维
        self.n_channels = n_channels

    def forward(self, mv):
        B, L, C, D = mv.shape
        pos_ids = torch.arange(L, device=mv.device)
        pos_vec = self.pos_embed(pos_ids)  # [L, C*10]
        pos_vec = pos_vec.reshape(L, C, 10) * 0.1  # scale down
        # 注入到grade-2 (index 6:16)
        g2_start, g2_end = GRADE_SLICES[2]
        mv_out = mv.clone()
        mv_out[:, :, :, g2_start:g2_end] = mv[:, :, :, g2_start:g2_end] + pos_vec.unsqueeze(0)
        return mv_out


class SimpleTokenMixer(nn.Module):
    """简单的token间信息混合（替代attention的v0.1方案）
    用因果卷积实现局部上下文混合，O(L)复杂度
    """
    def __init__(self, n_channels, kernel_size=8):
        super().__init__()
        dim = n_channels * N_BLADES  # C * 32
        self.conv = nn.Conv1d(dim, dim, kernel_size, padding=kernel_size-1, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.kernel_size = kernel_size

    def forward(self, mv):
        B, L, C, D = mv.shape
        x = mv.reshape(B, L, C * D)  # [B, L, C*32]
        x_t = x.transpose(1, 2)  # [B, C*32, L]
        # 因果卷积：只看过去
        conv_out = self.conv(x_t)[:, :, :L]  # 截断未来
        conv_out = conv_out.transpose(1, 2)  # [B, L, C*32]
        # 门控
        gate = torch.sigmoid(self.gate(x))
        mixed = x + gate * conv_out
        return mixed.reshape(B, L, C, D)


class BIICLMBlock(nn.Module):
    """BIIC语言模型的单个block
    = TokenMixer + BIICLayer(Writer+Eraser)
    """
    def __init__(self, n_channels):
        super().__init__()
        self.mixer = SimpleTokenMixer(n_channels)
        self.biic_layer = BIICLayer(n_channels, use_residual=True, use_eraser=True)

    def forward(self, mv, mv_prior):
        # 1. Token间混合（获取上下文）
        mv = self.mixer(mv)
        # 2. BIIC变换（等变演化 + 遗忘）
        mv = self.biic_layer(mv, mv_prior)
        return mv


class BIICLanguageModel(nn.Module):
    """BIIC语言模型 v0.1"""
    def __init__(self, config):
        super().__init__()
        self.config = config
        C = config['n_channels']

        # 编码器
        self.encoder = TokenToImmutableCore(config['vocab_size'], C)
        self.pos_enc = PositionEncoding(max_len=2048, n_channels=C)

        # 推理层
        self.blocks = nn.ModuleList([
            BIICLMBlock(C) for _ in range(config['n_layers'])
        ])

        # 修复Eraser初始化
        for block in self.blocks:
            if hasattr(block.biic_layer.mutable_layer, 'eraser'):
                with torch.no_grad():
                    block.biic_layer.mutable_layer.eraser.decay_logits.fill_(config['eraser_init'])

        # 解码器
        self.decoder = AllGradeDecoder(config['vocab_size'], C, config['d_hidden'])

    def forward(self, token_ids):
        """
        token_ids: [B, L]
        returns: logits [B, L, vocab_size]
        """
        # 编码
        mv = self.encoder(token_ids)
        mv = self.pos_enc(mv)

        # 保存不变核作为Eraser先验
        mv_prior = mv.clone().detach()

        # 推理
        for block in self.blocks:
            mv = block(mv, mv_prior)

        # 解码
        logits = self.decoder(mv)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════════════════
# 训练
# ═══════════════════════════════════════════════════════════

def get_lr(step, warmup_steps, max_lr):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    return max_lr


def train():
    print("=" * 60)
    print("BIIC Language Model v0.1 Training")
    print("=" * 60)
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")

    config = CONFIG
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)

    # 模型
    model = BIICLanguageModel(config).to(device)
    print("Params:", "{:,}".format(model.count_params()))
    print("Config:", json.dumps({k: v for k, v in config.items() if not k.endswith('_dir')}, indent=2))

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)

    # 训练
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    loss_history = []
    best_loss = float('inf')
    t_start = time.time()

    print("\nTraining started...\n", flush=True)

    for step in range(config['n_steps']):
        # 学习率warmup
        lr = get_lr(step, config['warmup_steps'], config['lr'])
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        # 生成随机数据（v0.1用随机token验证流程）
        token_ids = torch.randint(0, config['vocab_size'],
                                  (config['batch_size'], config['seq_len'])).to(device)
        inputs = token_ids[:, :-1]
        targets = token_ids[:, 1:]

        # 前向
        optimizer.zero_grad()
        logits = model(inputs)

        # Loss
        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, config['vocab_size']),
            targets.reshape(-1)
        )

        # 反向
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        optimizer.step()

        loss_val = loss.item()
        loss_history.append(loss_val)

        # 日志
        if step % config['log_every'] == 0:
            elapsed = time.time() - t_start
            steps_per_sec = (step + 1) / elapsed if elapsed > 0 else 0
            mem = torch.cuda.memory_allocated() / 1e6 if torch.cuda.is_available() else 0
            print("Step %5d | loss=%.4f | lr=%.2e | %.2f step/s | mem=%.0fMB" % (
                step, loss_val, lr, steps_per_sec, mem), flush=True)

        # 评估
        if step > 0 and step % config['eval_every'] == 0:
            avg_recent = np.mean(loss_history[-100:])
            avg_first = np.mean(loss_history[:100])
            improvement = (avg_first - avg_recent) / avg_first * 100
            print("\n  [Eval] step=%d, avg_loss_last100=%.4f, improvement=%.1f%%" % (
                step, avg_recent, improvement), flush=True)

            if avg_recent < best_loss:
                best_loss = avg_recent
                print("  [Eval] New best loss: %.4f" % best_loss, flush=True)
            print("", flush=True)

        # 保存
        if step > 0 and step % config['save_every'] == 0:
            ckpt_path = os.path.join(config['save_dir'], "step_%d.pt" % step)
            torch.save({
                'step': step,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss_history': loss_history,
                'config': config,
            }, ckpt_path)
            print("  [Save] %s" % ckpt_path, flush=True)

    # 最终保存
    elapsed = time.time() - t_start
    final_path = os.path.join(config['save_dir'], "final.pt")
    torch.save({
        'step': config['n_steps'],
        'model_state': model.state_dict(),
        'loss_history': loss_history,
        'config': config,
    }, final_path)

    # 结果
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("  Total time: %.1f minutes" % (elapsed / 60))
    print("  Steps: %d" % config['n_steps'])
    print("  Initial loss: %.4f" % np.mean(loss_history[:100]))
    print("  Final loss: %.4f" % np.mean(loss_history[-100:]))
    print("  Best loss: %.4f" % best_loss)
    print("  Improvement: %.1f%%" % ((np.mean(loss_history[:100]) - np.mean(loss_history[-100:])) / np.mean(loss_history[:100]) * 100))
    if torch.cuda.is_available():
        print("  Peak VRAM: %.0f MB" % (torch.cuda.max_memory_allocated() / 1e6))
    print("  Saved: %s" % final_path)

    # 保存训练日志
    log_path = os.path.join(config['log_dir'], "train_log.json")
    with open(log_path, 'w') as f:
        json.dump({
            'config': config,
            'loss_history': loss_history,
            'total_time_minutes': elapsed / 60,
            'initial_loss': float(np.mean(loss_history[:100])),
            'final_loss': float(np.mean(loss_history[-100:])),
            'best_loss': float(best_loss),
            'params': model.count_params(),
            'peak_vram_mb': torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0,
        }, f, indent=2)
    print("  Log: %s" % log_path)

    return loss_history


if __name__ == '__main__':
    train()
