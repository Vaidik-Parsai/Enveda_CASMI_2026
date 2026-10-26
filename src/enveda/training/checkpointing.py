"""Checkpoint loading, saving, and resumption management."""

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    step: int = 0,
    epoch: int = 0,
    loss_history: Optional[List[float]] = None,
    best_metric: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
    is_best: bool = False,
    overwrite: bool = True,
) -> Path:
    """Save model and optimizer state preserving baseline checkpoint structure.
    
    Args:
        filepath: Destination path.
        model: PyTorch model.
        optimizer: PyTorch optimizer.
        scaler: AMP GradScaler.
        step: Global step count.
        epoch: Epoch count.
        loss_history: List of recorded loss floats.
        best_metric: Best validation score.
        config: Training configuration dictionary.
        is_best: If True, also copies to best.pt in same directory.
        overwrite: If False, raises FileExistsError if destination exists.
        
    Returns:
        Path to the saved checkpoint.
    """
    path = Path(filepath)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Checkpoint already exists at {path} and overwrite=False.")

    path.parent.mkdir(parents=True, exist_ok=True)

    state: Dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "loss_history": loss_history if loss_history is not None else [],
    }

    if optimizer is not None:
        state["optimizer_state_dict"] = optimizer.state_dict()

    if scaler is not None and hasattr(scaler, "state_dict"):
        state["scaler_state_dict"] = scaler.state_dict()

    if best_metric is not None:
        state["best_metric"] = best_metric

    if config is not None:
        state["config"] = config

    torch.save(state, str(path))

    if is_best:
        best_path = path.parent / "best.pt"
        shutil.copyfile(str(path), str(best_path))

    return path


def load_checkpoint(
    filepath: Union[str, Path],
    model: nn.Module,
    optimizer: Optional[Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    device: Union[str, torch.device] = "cpu",
    strict: bool = True,
) -> Dict[str, Any]:
    """Load model and optimizer states with full backwards compatibility.
    
    Args:
        filepath: Path to checkpoint file.
        model: Model into which weights will be loaded.
        optimizer: Optional optimizer to restore.
        scaler: Optional GradScaler to restore.
        device: Device to map tensors to.
        strict: Whether to strictly enforce key matching in state_dict.
        
    Returns:
        Dictionary containing loaded checkpoint metadata.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {filepath}")

    try:
        checkpoint = torch.load(str(path), map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(path), map_location=device)

    # Load model weights
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=strict)

    # Load optimizer state if available and requested
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scaler state if available and requested
    if scaler is not None and "scaler_state_dict" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    return checkpoint
