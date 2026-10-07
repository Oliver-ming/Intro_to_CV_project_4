"""Write an official 3DGS checkpoint from exported Gaussian parameters."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2] / "third_party" / "gaussian-splatting"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scene.gaussian_model import GaussianModel  # noqa: E402


def _training_args() -> Namespace:
    return Namespace(
        iterations=30_000,
        position_lr_init=0.00016,
        position_lr_final=0.0000016,
        position_lr_delay_mult=0.01,
        position_lr_max_steps=30_000,
        feature_lr=0.0025,
        opacity_lr=0.025,
        scaling_lr=0.005,
        rotation_lr=0.001,
        exposure_lr_init=0.01,
        exposure_lr_final=0.001,
        exposure_lr_delay_steps=0,
        exposure_lr_delay_mult=0.0,
        percent_dense=0.01,
    )


def write_gaussian_checkpoint(
    path: Path,
    xyz: torch.Tensor,
    features_dc: torch.Tensor,
    features_rest: torch.Tensor,
    scaling: torch.Tensor,
    rotation: torch.Tensor,
    opacity: torch.Tensor,
    sh_degree: int,
) -> None:
    """Save ``(capture(), 0)`` so ``train.py --start_checkpoint`` restores these Gaussians.

    Tensors use the layout stored on ``GaussianModel``, not the activated values.
    The iteration is 0 because this is the export, not a later closed-loop resume.
    """
    model = GaussianModel(sh_degree)
    model.active_sh_degree = sh_degree
    model.spatial_lr_scale = 1.0
    model._xyz = nn.Parameter(xyz.detach().clone())
    model._features_dc = nn.Parameter(features_dc.detach().clone())
    model._features_rest = nn.Parameter(features_rest.detach().clone())
    model._scaling = nn.Parameter(scaling.detach().clone())
    model._rotation = nn.Parameter(rotation.detach().clone())
    model._opacity = nn.Parameter(opacity.detach().clone())
    model.max_radii2D = torch.zeros(xyz.shape[0])
    model._exposure = nn.Parameter(torch.eye(3, 4).unsqueeze(0))
    model.training_setup(_training_args())
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save((model.capture(), 0), path)
