"""Scale Depth Anything 3 depth into the current reconstruction."""

from __future__ import annotations

import torch


def align_reference_depth(rendered: torch.Tensor, estimated: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Multiply ``estimated`` by the median ratio of positive rendered depth.

    Both maps are the same shape. The ratio puts metric depth into the
    reconstruction's camera units. Non-positive or non-finite pixels are ignored
    when choosing the ratio and are left unchanged in the output.
    """
    if rendered.shape != estimated.shape:
        raise ValueError("depth maps must have the same shape")
    valid = (rendered > 0) & (estimated > 0) & torch.isfinite(rendered) & torch.isfinite(estimated)
    if not bool(valid.any()):
        raise ValueError("no positive finite overlap between the depth maps")
    ratio = torch.median(rendered[valid] / estimated[valid])
    aligned = torch.where(valid, estimated * ratio, estimated)
    return aligned, ratio
