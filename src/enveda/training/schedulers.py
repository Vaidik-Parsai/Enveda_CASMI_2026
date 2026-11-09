"""Learning rate schedulers with warmup and cosine annealing for retrieval training."""

import math
from typing import Any, Dict
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """Linear warmup followed by cosine annealing decay to minimum learning rate.
    
    Args:
        optimizer: Wrapped PyTorch optimizer.
        warmup_steps: Number of initial steps for linear warmup.
        total_steps: Total training steps for decay cycle.
        min_lr: Minimum learning rate floor at completion (default: 1e-6).
        last_epoch: The index of last step (defaults to -1).
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_steps: int = 500,
        total_steps: int = 5000,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(self.warmup_steps + 1, total_steps)
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        lrs = []
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                # Linear warmup from min_lr to base_lr
                lr = self.min_lr + (base_lr - self.min_lr) * (float(step) / float(self.warmup_steps))
            elif step >= self.total_steps:
                # Floor at min_lr
                lr = self.min_lr
            else:
                # Cosine decay from base_lr to min_lr
                progress = float(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine
            lrs.append(lr)
        return lrs

