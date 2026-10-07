"""Read the leaf Gaussians out of an i3dgs hierarchy PLY."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch


def load_hierarchy_gaussians(path: Path) -> dict:
    """Return raw 3DGS parameters for nodes that have no children.

    Internal hierarchy nodes are coarser copies of the same scene. The flat
    3DGS run keeps the leaves. Scales and opacities stay in the stored
    log / logit form. Spherical harmonics are coeff-major, matching the
    hierarchy writer.
    """
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    names = vertex.dtype.names
    child_names = sorted(
        (name for name in names if name.startswith("child_")),
        key=lambda name: int(name[6:]),
    )
    if child_names:
        children = np.column_stack([np.asarray(vertex[name]) for name in child_names])
        keep = np.all(children < 0, axis=1)
    else:
        keep = np.ones(len(vertex), dtype=bool)
    chosen = vertex[keep]
    if len(chosen) == 0:
        raise RuntimeError(f"no leaf Gaussians in {path}")
    rest_names = sorted(
        (name for name in names if name.startswith("f_rest_")),
        key=lambda name: int(name[7:]),
    )
    if len(rest_names) % 3 != 0:
        raise RuntimeError("f_rest fields are not RGB triplets")
    num_rest = len(rest_names) // 3
    sh_degree = math.isqrt(num_rest + 1) - 1
    if (sh_degree + 1) ** 2 - 1 != num_rest:
        raise RuntimeError(f"cannot infer an SH degree from {num_rest} rest coefficients")
    count = len(chosen)
    xyz = np.stack([chosen["x"], chosen["y"], chosen["z"]], axis=1).astype(np.float32)
    features_dc = np.stack([chosen[f"f_dc_{channel}"] for channel in range(3)], axis=1)
    features_dc = features_dc.reshape(count, 1, 3).astype(np.float32)
    if num_rest:
        flat = np.stack([chosen[name] for name in rest_names], axis=1).astype(np.float32)
        features_rest = flat.reshape(count, num_rest, 3)
    else:
        features_rest = np.zeros((count, 0, 3), dtype=np.float32)
    scaling = np.stack([chosen[f"scale_{channel}"] for channel in range(3)], axis=1).astype(np.float32)
    rotation = np.stack([chosen[f"rot_{channel}"] for channel in range(4)], axis=1).astype(np.float32)
    opacity = np.asarray(chosen["opacity"], dtype=np.float32).reshape(count, 1)
    return {
        "xyz": torch.from_numpy(xyz),
        "features_dc": torch.from_numpy(features_dc),
        "features_rest": torch.from_numpy(features_rest),
        "scaling": torch.from_numpy(scaling),
        "rotation": torch.from_numpy(rotation),
        "opacity": torch.from_numpy(opacity),
        "sh_degree": sh_degree,
    }
