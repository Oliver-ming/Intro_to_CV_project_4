"""Write one metrics record for a scene and a named run."""

from __future__ import annotations

import json
from pathlib import Path


def save_metrics(path: Path, record: dict) -> None:
    """Append ``record`` as one JSON object per line.

    Required keys are checked here so later tables cannot silently drop a clock
    or a rendering metric.
    """
    required = {
        "scene",
        "run",
        "iteration",
        "psnr",
        "ssim",
        "lpips_vgg",
        "pose_error",
        "dropped_views",
        "initialization_seconds",
        "reconstruction_seconds",
        "generation_seconds",
        "joint_reconstruction_seconds",
    }
    missing = required - record.keys()
    if missing:
        raise KeyError(f"metrics record missing {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
