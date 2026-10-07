"""Reload the 3DGS checkpoint before each generation round."""

from __future__ import annotations

from pathlib import Path

from src.generate.state import GenerationState
from src.reconstruct.render_view import generation_renderers, load_gaussians

ROUND_CHECKPOINT = {1: 1500, 2: 3000}


def checkpoint_insert(
    model_dir: Path,
    state: GenerationState,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
    sh_degree: int,
):
    """Return the callback ``run_closed_loop`` calls after iterations 1500 and 3000.

    Each call loads the checkpoint just written, so the second round renders the
    Gaussians trained after the first round, not the export from i3dgs.
    """

    def insert(round_index: int) -> None:
        step = ROUND_CHECKPOINT[round_index]
        gaussians = load_gaussians(model_dir / f"chkpnt{step}.pth", sh_degree)
        state.render_color, state.render_depth = generation_renderers(
            gaussians, state.poses, width, height, focal_x, focal_y
        )
        state(round_index)

    return insert
