"""Feed-forward initializers. Optional bundle adjustment stays off."""

from __future__ import annotations

from pathlib import Path


def vggt_command(scene_dir: Path, python: str, demo_script: Path) -> list[str]:
    """VGGT's COLMAP export. ``--use_ba`` is omitted on purpose."""
    return [python, str(demo_script), "--scene_dir", str(scene_dir)]


def launch_vggt(scene_dir: Path, python: str, demo_script: Path, run) -> Path:
    """Run VGGT, then move its model into ``sparse/0`` for 3DGS."""
    from src.init.sparse_layout import nest_sparse_model

    run(vggt_command(scene_dir, python, demo_script))
    return nest_sparse_model(scene_dir)
