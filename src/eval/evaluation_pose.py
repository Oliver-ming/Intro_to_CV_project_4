"""Photometric evaluation poses. Each test view gets 500 Adam steps."""

from __future__ import annotations

from collections.abc import Callable

import torch

EVALUATION_POSE_STEPS = 500


def _numeric_pose_grad(delta: torch.Tensor, target: torch.Tensor, render, eps: float = 1e-3) -> torch.Tensor:
    """Central difference is unnecessary; a forward difference is enough for Adam.

    The 3DGS rasterizer does not return a gradient for the camera, and the
    scoring renderer detaches its image. The six-vector is still the parameter.
    """
    base = delta.detach()
    base_image = render(base).detach()
    aligned = target.to(device=base_image.device, dtype=base_image.dtype)
    base_loss = torch.mean((base_image - aligned) ** 2)
    grad = torch.zeros_like(base)
    for index in range(base.shape[0]):
        bumped = base.clone()
        bumped[index] = bumped[index] + eps
        image = render(bumped).detach()
        loss = torch.mean((image - aligned) ** 2)
        grad[index] = (loss - base_loss) / eps
    return grad


def align_evaluation_pose(
    image: torch.Tensor,
    target: torch.Tensor,
    render: Callable[[torch.Tensor], torch.Tensor],
    steps: int = EVALUATION_POSE_STEPS,
) -> torch.Tensor:
    """Move one camera so ``render(delta)`` matches ``target``.

    ``image`` is unused except to fix the device. ``delta`` is a 6-vector
    (rotation, translation) starting at zero. The reconstruction itself is not
    an argument and must stay fixed inside ``render``.
    """
    del image
    delta = torch.zeros(6, device=target.device, dtype=target.dtype, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=1e-2)
    for _ in range(steps):
        optimizer.zero_grad()
        rendered = render(delta)
        aligned_target = target.to(device=rendered.device, dtype=rendered.dtype)
        loss = torch.mean((rendered - aligned_target) ** 2)
        if loss.requires_grad:
            loss.backward()
        else:
            delta.grad = _numeric_pose_grad(delta, target, render)
        optimizer.step()
    return delta.detach()
