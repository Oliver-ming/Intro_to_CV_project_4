"""The standard 3DGS command for one frozen-pose initialization."""

from __future__ import annotations

from pathlib import Path


def gaussian_training_command(
    source: Path,
    model: Path,
    python: str,
    train_script: Path,
    test_iterations: tuple[int, ...] = (7000, 30000),
    checkpoint_iterations: tuple[int, ...] = (),
) -> list[str]:
    """30k iterations, viewer off, poses left frozen.

    Part 1 reports at 7000 and 30000. The sparse-only control also checkpoints
    the comparison iterations.
    """
    command = [
        python,
        str(train_script),
        "-s",
        str(source),
        "-m",
        str(model),
        "--iterations",
        "30000",
        "--test_iterations",
        *[str(step) for step in test_iterations],
        "--save_iterations",
        *[str(step) for step in test_iterations],
        "--disable_viewer",
    ]
    if checkpoint_iterations:
        command.append("--checkpoint_iterations")
        command.extend(str(step) for step in checkpoint_iterations)
    return command
