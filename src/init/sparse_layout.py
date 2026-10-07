"""Put a COLMAP model where standard 3DGS looks for it."""

from __future__ import annotations

from pathlib import Path

MODEL_FILES = (
    "cameras.bin",
    "images.bin",
    "points3D.bin",
    "cameras.txt",
    "images.txt",
    "points3D.txt",
)


def nest_sparse_model(scene_dir: Path) -> Path:
    """Move ``sparse/*.bin`` into ``sparse/0`` when an exporter wrote the parent.

    VGGT writes the model directly under ``sparse/``. 3DGS reads ``sparse/0``.
    A model that is already nested is left in place.
    """
    sparse = scene_dir / "sparse"
    nested = sparse / "0"
    parent_cameras = sparse / "cameras.txt"
    if parent_cameras.is_file():
        nested.mkdir(parents=True, exist_ok=True)
        for name in MODEL_FILES:
            source = sparse / name
            if source.is_file():
                source.replace(nested / name)
        return nested
    if (nested / "cameras.bin").exists() or (nested / "cameras.txt").exists():
        return nested
    nested.mkdir(parents=True, exist_ok=True)
    moved = False
    for name in MODEL_FILES:
        source = sparse / name
        if source.is_file():
            source.rename(nested / name)
            moved = True
    if not moved:
        raise RuntimeError(f"no COLMAP model in {sparse}")
    return nested
