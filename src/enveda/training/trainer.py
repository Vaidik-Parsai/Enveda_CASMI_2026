"""Modular batch trainer supporting AMP mixed precision and safe checkpointing."""

import logging
import shutil
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer

from enveda.training.checkpointing import save_checkpoint
from enveda.training.losses import (
    multi_positive_contrastive_loss,
    normalized_supervised_contrastive_loss,
)


class Trainer:
    """Production batch trainer for the two-tower retrieval model with chemistry-aware negatives and schedulers."""

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
        scheduler: Optional[Any] = None,
        negative_pool: Optional[Any] = None,
        loss_fn_type: str = "multi_positive",
        negatives_per_anchor: int = 2,
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
        self.scheduler = scheduler
        self.negative_pool = negative_pool
        self.loss_fn_type = loss_fn_type
        self.negatives_per_anchor = negatives_per_anchor
        self.logger = logger or logging.getLogger("enveda")

        self.mixed_precision = self.config.get("training", {}).get("mixed_precision", True) and device.type == "cuda"
        init_scale = float(self.config.get("training", {}).get("scaler_init_scale", 8192.0))
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.mixed_precision, init_scale=init_scale)

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

        # Detailed configuration logging
        sched_name = type(self.scheduler).__name__ if self.scheduler is not None else "None (constant LR)"
        pool_status = (
            f"Active ({self.negative_pool.num_molecules:,} candidates, {self.negatives_per_anchor} neg/anchor)"
            if self.negative_pool is not None
            else "Disabled (in-batch negatives only)"
        )
        self.logger.info(
            f"Trainer Configuration Initialized:\n"
            f"  - Loss Objective:         {self.loss_fn_type}\n"
            f"  - Negative Sampling:      {pool_status}\n"
            f"  - LR Scheduler:           {sched_name}\n"
            f"  - Mixed Precision (AMP):  {self.mixed_precision} (init_scale={init_scale})\n"
            f"  - Checkpoint Directory:   {self.checkpoint_dir}\n"
            f"  - Target Training Steps:  {self.max_steps:,}"
        )

    def train_step(self, batch: Dict[str, Any]) -> float:
        """Execute a single forward-backward-optimizer step with bounded memory."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Move only this batch to GPU
        spectrum = batch["spectrum"].to(self.device, non_blocking=True)
        metadata = batch["metadata"].to(self.device, non_blocking=True)
        fingerprint = batch["fingerprint"].to(self.device, non_blocking=True)
        molecule_ids = batch["molecule_ids"].to(self.device, non_blocking=True)

        device_type = "cuda" if self.device.type == "cuda" else "cpu"
        hard_neg_weight = self.config.get("training", {}).get("hard_negative_weight", 1.0)
        hard_tanimoto_thresh = self.config.get("training", {}).get("hard_tanimoto_threshold", 0.4)

        # Chemistry-aware external negative sampling
        ext_neg_embs = None
        if self.negative_pool is not None and "inchikeys" in batch:
            inchikeys = batch["inchikeys"]
            pmzs = batch.get("precursor_mzs", [0.0] * len(inchikeys))
            adducts = batch.get("adducts", [""] * len(inchikeys))
            modes = batch.get("ionization_modes", [""] * len(inchikeys))
            fps_np = fingerprint.cpu().numpy()

            sampled_neg_indices = set()
            for idx_a in range(len(inchikeys)):
                neg_idxs, _ = self.negative_pool.sample_negatives_for_anchor(
                    anchor_inchikey=inchikeys[idx_a],
                    anchor_fp=fps_np[idx_a],
                    anchor_precursor_mz=float(pmzs[idx_a]),
                    anchor_adduct=adducts[idx_a],
                    anchor_mode=modes[idx_a],
                    num_negatives=self.negatives_per_anchor,
                    anchor_idx=idx_a,
                )
                sampled_neg_indices.update(neg_idxs)

            if len(sampled_neg_indices) > 0:
                neg_fps_tensor = torch.from_numpy(
                    self.negative_pool.fingerprints[list(sampled_neg_indices)]
                ).float().to(self.device, non_blocking=True)
                with torch.autocast(device_type=device_type, enabled=self.mixed_precision):
                    ext_neg_embs = self.model.encode_molecule(neg_fps_tensor)

        with torch.autocast(device_type=device_type, enabled=self.mixed_precision):
            spec_emb, mol_emb = self.model(spectrum, metadata, fingerprint)

            if self.loss_fn_type == "normalized_supcon":
                loss = normalized_supervised_contrastive_loss(
                    spec_emb,
                    mol_emb,
                    molecule_ids,
                    temperature=self.config.get("model", {}).get("temperature", 0.07),
                    fingerprints=fingerprint if hard_neg_weight > 1.0 else None,
                    hard_negative_weight=hard_neg_weight,
                    hard_tanimoto_threshold=hard_tanimoto_thresh,
                    negative_molecule_embeddings=ext_neg_embs,
                )
            else:
                loss = multi_positive_contrastive_loss(
                    spec_emb,
                    mol_emb,
                    molecule_ids,
                    temperature=self.config.get("model", {}).get("temperature", 0.07),
                    fingerprints=fingerprint if hard_neg_weight > 1.0 else None,
                    hard_negative_weight=hard_neg_weight,
                    hard_tanimoto_threshold=hard_tanimoto_thresh,
                    negative_molecule_embeddings=ext_neg_embs,
                )

        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss ({loss.item()}) encountered at step {self.global_step}")

        # Backward pass with scaler
        self.scaler.scale(loss).backward()

        # Unscale before gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)

        # Optimizer and scaler step
        scale_before = self.scaler.get_scale()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = self.scaler.get_scale()

        # Step scheduler only if optimizer step was not skipped by GradScaler
        if self.scheduler is not None and scale_before <= scale_after:
            self.scheduler.step()

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
                    ckpt_path = self.checkpoint_dir / f"step_{self.global_step}.pt"
                    save_checkpoint(
                        filepath=ckpt_path,
                        model=self.model,
                        optimizer=self.optimizer,
                        scaler=self.scaler,
                        scheduler=self.scheduler,
                        step=self.global_step,
                        loss_history=self.loss_history,
                        best_metric=self.best_val_metric,
                        config=dict(self.config),
                    )
                    # Also keep spectrum_molecule_step_X.pt alias for compatibility
                    alias_path = self.checkpoint_dir / f"spectrum_molecule_step_{self.global_step}.pt"
                    if alias_path != ckpt_path:
                        shutil.copyfile(str(ckpt_path), str(alias_path))
                    self.logger.info(f"Checkpoint saved: {ckpt_path}")

        except KeyboardInterrupt:
            self.logger.warning("Training interrupted by user! Saving emergency checkpoint...")
            emergency_path = self.checkpoint_dir / f"spectrum_molecule_interrupted_step_{self.global_step}.pt"
            save_checkpoint(
                filepath=emergency_path,
                model=self.model,
                optimizer=self.optimizer,
                scaler=self.scaler,
                scheduler=self.scheduler,
                step=self.global_step,
                loss_history=self.loss_history,
                config=dict(self.config),
            )
            self.logger.info(f"Emergency checkpoint saved to {emergency_path}")

        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.1f}s. Total steps: {self.global_step}")
        return self.loss_history
