"""Launch i3dgs with its own extensions, not the Part 1 rasterizer."""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.data.intrinsics import focal_for_prepared
from src.data.link_split import link_split
from src.run.part2 import i3dgs_command

I3DGS_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "i3dgs"
PY_EXT = I3DGS_ROOT / "py_ext"
DA3_SRC = I3DGS_ROOT.parent / "Depth-Anything-3" / "src"


def part2_environ(environ: dict[str, str] | None = None) -> dict[str, str]:
    """Put the i3dgs extension build ahead of the Part 1 packages."""
    base = dict(os.environ if environ is None else environ)
    prefix = f"{PY_EXT}{os.pathsep}{I3DGS_ROOT}{os.pathsep}{DA3_SRC}"
    previous = base.get("PYTHONPATH")
    base["PYTHONPATH"] = prefix if not previous else prefix + os.pathsep + previous
    return base


def part2_command(source: Path, model: Path, python: str, focal_x: float) -> list[str]:
    return i3dgs_command(source, model, python, I3DGS_ROOT / "train.py", focal_x)


def launch_part2(prepared: Path, work: Path, split_name: str, python: str, run, cameras_path: Path | None = None) -> float:
    """Link one split and run i3dgs. ``run`` receives the command and the environment.

    The focal length is the dataset value on the prepared image. No ground-truth
    extrinsics are passed. ``split_name`` is ``dense`` or ``sparse``.
    """
    if split_name not in {"dense", "sparse"}:
        raise ValueError("Part 2 trains on dense or sparse views")
    manifest = json.loads((prepared / "split.json").read_text())
    normalized = None
    if cameras_path is not None:
        payload = json.loads(cameras_path.read_text())
        item = payload[0] if isinstance(payload, list) else payload
        normalized = (item["fx"], item["fy"], item["cx"], item["cy"])
    focal = focal_for_prepared(manifest["scene"], manifest["width"], manifest["height"], normalized)
    source = (work / split_name).resolve()
    link_split(prepared, split_name, source / "images")
    run(part2_command(source, (work / f"model-{split_name}").resolve(), python, focal), part2_environ())
    return focal


def execute_part2(
    prepared: Path,
    work: Path,
    split_name: str,
    python: str,
    cameras_path: Path | None = None,
    spawn=None,
) -> dict:
    """Run one i3dgs split and record the joint reconstruction clock."""
    import subprocess
    import time

    started = time.perf_counter()
    log_path = work / f"{split_name}.log"

    def run(command, env):
        work.mkdir(parents=True, exist_ok=True)
        if spawn is not None:
            spawn(command, env)
            return
        with log_path.open("w") as handle:
            completed = subprocess.run(
                command,
                env=env,
                cwd=I3DGS_ROOT,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"i3dgs exited {completed.returncode}; see {log_path}")

    focal = launch_part2(prepared, work, split_name, python, run, cameras_path)
    record = {
        "split": split_name,
        "focal": focal,
        "joint_reconstruction_seconds": time.perf_counter() - started,
    }
    (work / f"{split_name}_clock.json").write_text(json.dumps(record, indent=2) + "\n")
    (work / f"TRAIN_PART2_{split_name.upper()}_OK").write_text(f"{split_name}\n")
    return record
