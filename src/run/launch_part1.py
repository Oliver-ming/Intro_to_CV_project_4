"""Run COLMAP on Dense Views, then standard 3DGS with frozen poses."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from src.data.intrinsics import colmap_reader_args
from src.data.link_split import link_split
from src.eval.ground_truth import dl3dv_camera_centers, waymo_camera_centers
from src.init.colmap import pinhole_training_source, run_colmap
from src.run.part1 import gaussian_training_command

TRAIN_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "gaussian-splatting"
    / "train.py"
)


def reader_args_for_prepared(scene: str, prepared: Path) -> list[str]:
    """COLMAP camera from the prepared size, plus DL3DV fractions when present."""
    manifest = json.loads((prepared / "split.json").read_text())
    normalized = None
    cameras = prepared / "cameras.json"
    if cameras.is_file():
        payload = json.loads(cameras.read_text())
        item = payload[0] if isinstance(payload, list) else payload
        normalized = (item["fx"], item["fy"], item["cx"], item["cy"])
    return colmap_reader_args(scene, manifest["width"], manifest["height"], normalized)


def ground_truth_centers(scene: str, prepared: Path):
    """Training-order camera centers. Waymo reads the copied ``gt`` folder."""
    if scene == "waymo-405841":
        return waymo_camera_centers(prepared.parent.parent / "external" / scene / "gt")
    return dl3dv_camera_centers(prepared / "cameras.json")


def launch_part1(scene: str, prepared: Path, work: Path, python: str) -> None:
    """Initialize with COLMAP and train 3DGS. Writes ``TRAIN_PART1_OK`` on success."""
    source = work / scene / "colmap"
    link_split(prepared, "dense", source / "images")
    clocks_path = work / scene / "clocks.json"
    fresh = run_colmap(
        source / "images",
        source,
        reader_args=reader_args_for_prepared(scene, prepared),
    )
    if fresh.get("skipped"):
        clocks = json.loads(clocks_path.read_text()) if clocks_path.is_file() else {"initialization_seconds": 0.0}
    else:
        clocks = {"initializer": "colmap", "initialization_seconds": fresh["initialization_seconds"]}
        clocks_path.parent.mkdir(parents=True, exist_ok=True)
        clocks_path.write_text(json.dumps(clocks, indent=2) + "\n")
    train_source = pinhole_training_source(source)
    model = work / scene / "3dgs"
    log_path = work / scene / "train.log"
    command = gaussian_training_command(
        train_source,
        model,
        python,
        TRAIN_SCRIPT,
        checkpoint_iterations=(7000, 30000),
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with log_path.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    reconstruction_seconds = time.perf_counter() - started
    if completed.returncode != 0:
        raise RuntimeError(f"3DGS exited {completed.returncode}; see {log_path}")
    from src.eval.score_scene import write_checkpoint_metrics

    metrics_path = work / scene / "metrics.jsonl"
    centers = ground_truth_centers(scene, prepared)
    for iteration in (7000, 30000):
        write_checkpoint_metrics(
            scene,
            prepared,
            train_source,
            model,
            metrics_path,
            centers,
            iteration,
            clocks["initialization_seconds"],
            reconstruction_seconds,
        )
    (work / scene / "TRAIN_PART1_OK").write_text("colmap\n")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--python", default="python")
    args = parser.parse_args()
    launch_part1(args.scene, args.prepared, args.work, args.python)


if __name__ == "__main__":
    main()
