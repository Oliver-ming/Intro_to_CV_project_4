"""PSNR, SSIM, and LPIPS for a finished i3dgs export.

The checkpoint is the exported i3dgs Gaussians, not a 7000-step 3DGS run.
Joint Reconstruction Time comes from the pose summary already written for that split.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.eval.score_scene import write_checkpoint_metrics
from src.run.launch_part1 import ground_truth_centers


def joint_reconstruction_seconds(pose_metrics: Path, split: str) -> float:
    """Wall clock recorded for ``i3dgs-dense`` or ``i3dgs-sparse``."""
    run = f"i3dgs-{split}"
    for line in pose_metrics.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("run") == run:
            return float(record["joint_reconstruction_seconds"])
    raise KeyError(f"{run} is not in {pose_metrics}")


def score_i3dgs_export(
    scene: str,
    prepared: Path,
    source: Path,
    model_dir: Path,
    metrics_path: Path,
    split: str,
    joint_seconds: float,
) -> dict:
    """Score test views of one exported i3dgs split. Iteration 0 is the export."""
    if split not in {"dense", "sparse"}:
        raise ValueError("split must be dense or sparse")
    return write_checkpoint_metrics(
        scene,
        prepared,
        source,
        model_dir,
        metrics_path,
        ground_truth_centers(scene, prepared),
        0,
        0.0,
        0.0,
        training_split=split,
        run_name=f"i3dgs-{split}",
        joint_reconstruction_seconds=joint_seconds,
    )
