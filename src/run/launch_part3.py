"""Part 3 entry: 30k 3DGS with two generation rounds between resumed segments."""

from __future__ import annotations

from pathlib import Path

from src.generate.state import GenerationState
from src.run.execute_closed_loop import run_closed_loop
from src.run.launch_part1 import TRAIN_SCRIPT


def launch_part3(
    source: Path,
    model: Path,
    python: str,
    state: GenerationState,
    run,
    start_checkpoint: Path | None = None,
) -> dict:
    """Run the three segments. ``run`` is timed by the caller as reconstruction time.

    ``state`` inserts Generated Views after iterations 1500 and 3000. Its
    ``generation_seconds`` is the Generation Time and is not part of training.
    ``start_checkpoint`` is the exported i3dgs Gaussian file for the first segment.
    """
    run_closed_loop(source, model, python, TRAIN_SCRIPT, run, state, start_checkpoint)
    return {
        "generation_seconds": state.generation_seconds,
        "failure": state.failure,
    }
