"""Render one frozen camera from an official 3DGS model."""

from __future__ import annotations

import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.reconstruct.gaussian_camera import make_minicam

ROOT = Path(__file__).resolve().parents[2] / "third_party" / "gaussian-splatting"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gaussian_renderer import render  # noqa: E402


def render_view(
    model,
    camera_to_world: np.ndarray,
    width: int,
    height: int,
    focal_x: float,
    focal_y: float,
) -> dict[str, torch.Tensor]:
    """Color is ``3,H,W`` in ``[0, 1]``. Depth is ``H,W`` from the same rasterizer."""
    camera = make_minicam(camera_to_world, width, height, focal_x, focal_y)
    camera.world_view_transform = camera.world_view_transform.cuda()
    camera.full_proj_transform = camera.full_proj_transform.cuda()
    camera.camera_center = torch.inverse(camera.world_view_transform)[3, :3]
    pipe = Namespace(
        convert_SHs_python=False,
        compute_cov3D_python=False,
        debug=False,
        antialiasing=False,
    )
    package = render(camera, model, pipe, torch.zeros(3, device="cuda"))
    depth = package["depth"]
    if depth.ndim == 3:
        depth = depth[0]
    return {"color": package["render"], "depth": depth}


def color_to_image(color: torch.Tensor) -> Image.Image:
    """``3,H,W`` float image to a PIL RGB image."""
    array = (color.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def load_gaussians(path: Path, sh_degree: int):
    """Load a checkpoint written by ``write_gaussian_checkpoint`` onto CUDA."""
    from torch import nn

    from src.reconstruct.export_gaussians import ROOT
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from scene.gaussian_model import GaussianModel

    captured, _iteration = torch.load(path, map_location="cuda", weights_only=False)
    model = GaussianModel(sh_degree)
    (
        model.active_sh_degree,
        xyz,
        features_dc,
        features_rest,
        scaling,
        rotation,
        opacity,
        max_radii2D,
        model.xyz_gradient_accum,
        model.denom,
        _optimizer,
        model.spatial_lr_scale,
    ) = captured
    model._xyz = nn.Parameter(xyz.detach())
    model._features_dc = nn.Parameter(features_dc.detach())
    model._features_rest = nn.Parameter(features_rest.detach())
    model._scaling = nn.Parameter(scaling.detach())
    model._rotation = nn.Parameter(rotation.detach())
    model._opacity = nn.Parameter(opacity.detach())
    model.max_radii2D = max_radii2D.detach()
    return model


def generation_renderers(model, poses: dict[int, torch.Tensor], width: int, height: int, focal_x: float, focal_y: float):
    """Color for a target pose, and depth for a registered training index."""

    def render_color(pose: torch.Tensor) -> Image.Image:
        rendered = render_view(model, pose.detach().cpu().numpy(), width, height, focal_x, focal_y)
        return color_to_image(rendered["color"])

    def render_depth(index: int) -> torch.Tensor:
        rendered = render_view(model, poses[index].detach().cpu().numpy(), width, height, focal_x, focal_y)
        return rendered["depth"].detach().cpu()

    return render_color, render_depth
