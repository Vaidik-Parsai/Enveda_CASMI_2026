"""Batch collation and device transfer helpers."""

from typing import Dict, List, Optional
import torch


def collate_spectrum_batch(
    batch: List[Dict[str, torch.Tensor]],
    device: Optional[torch.device] = None,
) -> Dict[str, torch.Tensor]:
    """Collate individual sample dictionaries into a batched dictionary on device."""
    keys = batch[0].keys()
    collated = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        collated[k] = torch.stack(tensors, dim=0)
        if device is not None:
            collated[k] = collated[k].to(device, non_blocking=True)
    return collated
