"""Run the three 3DGS segments and call back when a generation round is due."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.run.closed_loop import closed_loop_segments, segment_command


def run_closed_loop(
    source: Path,
    model: Path,
    python: str,
    train_script: Path,
    run: Callable[[list[str]], None],
    insert_round: Callable[[int], None],
    start_checkpoint: Path | None = None,
) -> None:
    """Train to 1500, insert round 1, train to 3000, insert round 2, train to 30000.

    ``run`` executes one 3DGS command. ``insert_round`` receives 1 or 2 and must
    add Generated Views before the next segment starts. Poses stay frozen because
    the underlying command does not turn on pose optimization. ``start_checkpoint``
    is the i3dgs export and is loaded by the first segment.
    """
    checkpoint = start_checkpoint
    for segment in closed_loop_segments():
        command = segment_command(
            source,
            model,
            python,
            train_script,
            segment["iterations"],
            segment["checkpoint_iterations"],
            checkpoint,
        )
        run(command)
        if segment["after"] == "insert_round_1":
            insert_round(1)
            checkpoint = model / "chkpnt1500.pth"
        elif segment["after"] == "insert_round_2":
            insert_round(2)
            checkpoint = model / "chkpnt3000.pth"
