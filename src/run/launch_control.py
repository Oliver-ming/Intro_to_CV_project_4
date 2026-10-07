"""Sparse-only control: the exported i3dgs Gaussians, no generated views."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from src.eval.score_scene import write_checkpoint_metrics
from src.run.launch_part1 import TRAIN_SCRIPT, ground_truth_centers
from src.run.part3 import control_command, control_iterations


def execute_control(
    prepared: Path,
    source: Path,
    model: Path,
    python: str,
    start_checkpoint: Path,
    sh_degree: int,
    metrics_path: Path,
) -> dict:
    """Train 30000 iterations from the export and score every comparison checkpoint.

    The reconstruction clock is the whole control run. Generation time stays 0.
    Success is ``TRAIN_CONTROL_OK`` next to the model.
    """
    manifest = json.loads((prepared / "split.json").read_text())
    scene = manifest["scene"]
    command = control_command(
        source,
        model,
        python,
        TRAIN_SCRIPT,
        start_checkpoint,
        sh_degree,
    )
    model.mkdir(parents=True, exist_ok=True)
    log_path = model / "train.log"
    started = time.perf_counter()
    with log_path.open("w") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    reconstruction_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"sparse-only control exited {completed.returncode}; see {log_path}")
    centers = ground_truth_centers(scene, prepared)
    records = []
    for iteration in control_iterations():
        records.append(
            write_checkpoint_metrics(
                scene,
                prepared,
                source,
                model,
                metrics_path,
                centers,
                iteration,
                0.0,
                reconstruction_seconds,
                training_split="sparse",
                run_name="sparse-control",
            )
        )
    (model / "TRAIN_CONTROL_OK").write_text("sparse-control\n")
    return {"reconstruction_seconds": reconstruction_seconds, "records": len(records)}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sh-degree", type=int, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    execute_control(
        args.prepared,
        args.source,
        args.model,
        args.python,
        args.checkpoint,
        args.sh_degree,
        args.metrics,
    )


if __name__ == "__main__":
    main()
