"""Part 3 entry: one 30k run with two generation rounds, plus the control."""

from __future__ import annotations

from pathlib import Path

from src.reconstruct.schedule import CONTROL_ITERS, closed_loop_event
from src.run.part1 import gaussian_training_command


def control_iterations() -> list[int]:
    return list(CONTROL_ITERS)


def closed_loop_actions() -> dict[int, str]:
    return {
        iteration: event
        for iteration in range(1, 30001)
        if (event := closed_loop_event(iteration)) is not None
    }


def control_command(
    source: Path,
    model: Path,
    python: str,
    train_script: Path,
    start_checkpoint: Path | None = None,
    sh_degree: int | None = None,
) -> list[str]:
    """Sparse Views only, same 30k budget, checkpoints at every comparison iteration.

    ``start_checkpoint`` is the exported i3dgs Gaussian file. Without it, 3DGS
    would throw those Gaussians away and grow a new cloud from the placeholder points.
    """
    command = gaussian_training_command(
        source,
        model,
        python,
        train_script,
        test_iterations=CONTROL_ITERS,
        checkpoint_iterations=CONTROL_ITERS,
    )
    if sh_degree is not None:
        command.extend(["--sh_degree", str(sh_degree)])
    if start_checkpoint is not None:
        command.extend(["--start_checkpoint", str(start_checkpoint)])
    return command
