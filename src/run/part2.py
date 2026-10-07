"""i3dgs command: known focal length, no ground-truth extrinsics."""

from __future__ import annotations

from pathlib import Path

from src.data.splits import MAX_WIDTH


def scaled_focal(focal_x: float, width: int, max_width: int = MAX_WIDTH) -> float:
    """Focal length after the shared width cap, in pixels of the resized image."""
    if width <= 0 or focal_x <= 0:
        raise ValueError("width and focal length must be positive")
    if width <= max_width:
        return focal_x
    return focal_x * max_width / width


def i3dgs_command(
    source: Path,
    model: Path,
    python: str,
    train_script: Path,
    focal_x: float,
) -> list[str]:
    """Images only. ``--fix_focal`` keeps the dataset intrinsic from being re-estimated."""
    return [
        python,
        str(train_script),
        "-s",
        str(source),
        "-m",
        str(model),
        "--fix_focal",
        "--init_focal",
        str(focal_x),
        "--viewer_mode",
        "none",
    ]
