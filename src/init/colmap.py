"""COLMAP initialization. The clock stops when cameras and points are written."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path


def run_colmap(
    image_dir: Path,
    workspace: Path,
    colmap_bin: str = "colmap",
    reader_args: list[str] | None = None,
) -> dict[str, float]:
    """Run feature extraction, matching, and mapping. Return the initialization clock.

    ``image_dir`` must already be the Dense View or Sparse View set. Test views
    are not passed in here.
    """
    database = workspace / "database.db"
    sparse = workspace / "sparse"
    model = sparse / "0"
    if (model / "cameras.txt").exists():
        return {"initialization_seconds": 0.0, "skipped": 1.0}
    sparse.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    _run(colmap_bin, [
        "feature_extractor",
        "--database_path", str(database),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        *(reader_args or []),
    ])
    _run(colmap_bin, [
        "exhaustive_matcher",
        "--database_path", str(database),
    ])
    _run(colmap_bin, [
        "mapper",
        "--database_path", str(database),
        "--image_path", str(image_dir),
        "--output_path", str(sparse),
    ])
    model = sparse / "0"
    _run(colmap_bin, [
        "model_converter",
        "--input_path", str(model),
        "--output_path", str(model),
        "--output_type", "TXT",
    ])
    drop_binary_model(model)
    elapsed = time.perf_counter() - started
    if not (model / "cameras.txt").exists():
        raise RuntimeError(f"COLMAP wrote no text model in {model}")
    return {"initialization_seconds": elapsed}


def camera_model_name(cameras_text: str) -> str:
    """The model name on the first camera line of a COLMAP ``cameras.txt``."""
    for line in cameras_text.splitlines():
        if line and not line.startswith("#"):
            return line.split()[1]
    raise ValueError("cameras.txt has no camera")


def pinhole_training_source(workspace: Path, colmap_bin: str = "colmap") -> Path:
    """Return a 3DGS scene whose camera is PINHOLE.

    An OPENCV model is undistorted into ``workspace/pinhole``. A model that is
    already PINHOLE is returned unchanged. 3DGS rejects every other camera model.
    """
    cameras_path = workspace / "sparse" / "0" / "cameras.txt"
    if camera_model_name(cameras_path.read_text()) == "PINHOLE":
        return workspace
    output = workspace / "pinhole"
    output_cameras = output / "sparse" / "0" / "cameras.txt"
    if output_cameras.is_file() and camera_model_name(output_cameras.read_text()) == "PINHOLE":
        return output
    output.mkdir(parents=True, exist_ok=True)
    _run(colmap_bin, [
        "image_undistorter",
        "--image_path", str(workspace / "images"),
        "--input_path", str(workspace / "sparse" / "0"),
        "--output_path", str(output),
        "--output_type", "COLMAP",
    ])
    from src.init.sparse_layout import nest_sparse_model

    nested = nest_sparse_model(output)
    _run(colmap_bin, [
        "model_converter",
        "--input_path", str(nested),
        "--output_path", str(nested),
        "--output_type", "TXT",
    ])
    drop_binary_model(nested)
    if camera_model_name((nested / "cameras.txt").read_text()) != "PINHOLE":
        raise RuntimeError(f"undistorted model in {nested} is not PINHOLE")
    return output


def drop_binary_model(model: Path) -> None:
    """Remove COLMAP binary files so later text edits are what 3DGS reads.

    The reader prefers ``images.bin`` whenever that file exists. Generated views
    are appended to ``images.txt``, so the binary copies must not remain.
    """
    for name in ("images.bin", "cameras.bin", "points3D.bin"):
        path = model / name
        if path.exists():
            path.unlink()


def _run(binary: str, args: list[str]) -> None:
    completed = subprocess.run([binary, *args], check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr[-4000:] or completed.stdout[-4000:])
