"""Split the closed loop into resumed 3DGS runs at the fixed iteration points."""

from __future__ import annotations

from pathlib import Path

from src.run.part1 import gaussian_training_command


def closed_loop_segments() -> list[dict]:
    """Three resumed runs. Generation happens after the first two checkpoints."""
    return [
        {"iterations": 1500, "checkpoint_iterations": [1500], "after": "insert_round_1"},
        {"iterations": 3000, "checkpoint_iterations": [3000], "after": "insert_round_2"},
        {
            "iterations": 30000,
            "checkpoint_iterations": [4500, 7000, 30000],
            "after": "finish",
        },
    ]


def segment_command(
    source: Path,
    model: Path,
    python: str,
    train_script: Path,
    iterations: int,
    checkpoint_iterations: list[int],
    start_checkpoint: Path | None,
) -> list[str]:
    command = gaussian_training_command(source, model, python, train_script)
    command.extend(["--iterations", str(iterations)])
    command.append("--checkpoint_iterations")
    command.extend(str(step) for step in checkpoint_iterations)
    if start_checkpoint is not None:
        command.extend(["--start_checkpoint", str(start_checkpoint)])
    return command
