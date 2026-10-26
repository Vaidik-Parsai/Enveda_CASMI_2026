"""Modular batch trainer supporting AMP mixed precision and safe checkpointing."""

import logging
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from enveda.training.checkpointing import save_checkpoint
from enveda.training.losses import multi_positive_contrastive_loss


class Trainer:
    """Production batch trainer for the two-tower retrieval model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        config: Dict[str, Any],
        device: torch.device,
        train_iterator: Optional[Iterator[Dict[str, torch.Tensor]]] = None,
        train_dataset: Optional[Any] = None,
        val_iterator: Optional[Iterator[Dict[str, torch.Tensor]]] = None,
        val_evaluator: Optional[Callable[[], Dict[str, float]]] = None,
        initial_loss_history: Optional[List[float]] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.device = device
        self.train_dataset = train_dataset
        self.train_iterator = train_iterator or (iter(train_dataset) if train_dataset is not None else None)
        self.val_iterator = val_iterator
        self.val_evaluator = val_evaluator
        self.logger = logger or logging.getLogger("enveda")

        self.mixed_precision = self.config.get("training", {}).get("mixed_precision", True) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)

        self.max_steps = self.config.get("training", {}).get("max_steps", 1000)
        self.max_grad_norm = self.config.get("training", {}).get("max_grad_norm", 1.0)
        self.log_interval = self.config.get("training", {}).get("log_interval", 10)
        self.checkpoint_interval = self.config.get("training", {}).get("checkpoint_interval", 500)
        self.eval_interval = self.config.get("training", {}).get("eval_interval", 200)

        self.checkpoint_dir = Path(self.config.get("training", {}).get("checkpoint_dir", "checkpoints"))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.loss_history: List[float] = list(initial_loss_history) if initial_loss_history is not None else []
        self.best_val_metric: Optional[float] = None

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute a single forward-backward-optimizer step with bounded memory."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Move only this batch to GPU
        spectrum = batch["spectrum"].to(self.device, non_blocking=True)
        metadata = batch["metadata"].to(self.device, non_blocking=True)
        fingerprint = batch["fingerprint"].to(self.device, non_blocking=True)
        molecule_ids = batch["molecule_ids"].to(self.device, non_blocking=True)

        # Autocast for mixed precision
        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        with torch.autocast(device_type=device_type, enabled=self.mixed_precision):
            spec_emb, mol_emb = self.model(spectrum, metadata, fingerprint)
            loss = multi_positive_contrastive_loss(
                spec_emb,
                mol_emb,
                molecule_ids,
                temperature=self.config.get("model", {}).get("temperature", 0.07),
            )

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss ({loss.item()}) encountered at step {self.global_step}")

        # Backward pass with scaler
        self.scaler.scale(loss).backward()

        # Unscale before gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()

        loss_val = float(loss.item())
        del spectrum, metadata, fingerprint, molecule_ids, spec_emb, mol_emb, loss
        return loss_val

    def train(self, start_step: int = 0) -> List[float]:
        """Run training loop up to self.max_steps."""
        self.global_step = start_step
        self.logger.info(f"Starting training from step {start_step + 1} to {self.max_steps} on {self.device}")
        start_time = time.time()

        try:
            while self.global_step < self.max_steps:
                self.global_step += 1
                try:
                    batch = next(self.train_iterator)
                except StopIteration:
                    if self.train_dataset is not None:
                        self.logger.info("Dataset epoch completed; resetting iterator for next epoch.")
                        self.train_iterator = iter(self.train_dataset)
                        batch = next(self.train_iterator)
                    else:
                        self.logger.info("Train iterator exhausted; terminating.")
                        break

                step_loss = self.train_step(batch)
                self.loss_history.append(step_loss)

                # Periodic logging
                if self.global_step % self.log_interval == 0:
                    window = min(self.log_interval, len(self.loss_history))
                    mean_loss = float(np.mean(self.loss_history[-window:]))
                    mem_str = ""
                    if self.device.type == "cuda":
                        mem_mb = torch.cuda.max_memory_allocated() / (1024**2)
                        mem_str = f" | GPU Mem: {mem_mb:.1f} MB"
                    self.logger.info(
                        f"Step {self.global_step:5d}/{self.max_steps} | "
                        f"Mean Loss ({window}): {mean_loss:.4f}{mem_str}"
                    )

                # Periodic evaluation
                if self.val_evaluator is not None and self.global_step % self.eval_interval == 0:
                    self.model.eval()
                    self.logger.info(f"Running evaluation at step {self.global_step}...")
                    metrics = self.val_evaluator()
                    self.logger.info(f"Validation metrics at step {self.global_step}: {metrics}")

                # Periodic checkpointing
                if self.global_step % self.checkpoint_interval == 0 or self.global_step == self.max_steps:
                    ckpt_path = self.checkpoint_dir / f"spectrum_molecule_step_{self.global_step}.pt"
                    save_checkpoint(
                        filepath=ckpt_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        scaler=self.scaler,
                        step=self.global_step,
                        loss_history=self.loss_history,
                        best_metric=self.best_val_metric,
                        config=dict(self.config),
                    )
                    self.logger.info(f"Checkpoint saved: {ckpt_path}")

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user! Saving emergency checkpoint...")
            emergency_path = self.checkpoint_dir / f"spectrum_molecule_interrupted_step_{self.global_step}.pt"
            save_checkpoint(
                filepath=emergency_path,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                step=self.global_step,
                loss_history=self.loss_history,
                config=dict(self.config),
            )
            self.logger.info(f"Emergency checkpoint saved to {emergency_path}")

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s. Total steps: {self.global_step}")
        return self.loss_history
