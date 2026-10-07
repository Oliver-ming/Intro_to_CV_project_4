"""Keep generated views so the second round can pair them."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.generate.round import run_generation_round


class GenerationState:
    """Training views seen by both closed-loop rounds."""

    def __init__(
        self,
        indices: list[int],
        poses: dict[int, torch.Tensor],
        images: dict[int, Image.Image],
        intrinsics: np.ndarray,
        scene_dir: Path,
        camera_id: int,
        first_image_id: int,
        render_color: Callable[[torch.Tensor], Image.Image],
        render_depth: Callable[[int], torch.Tensor],
        estimate_depth: Callable[[Image.Image], torch.Tensor],
        restore: Callable[..., Image.Image],
    ) -> None:
        self.indices = list(indices)
        self.poses = dict(poses)
        self.images = dict(images)
        self.intrinsics = intrinsics
        self.scene_dir = scene_dir
        self.camera_id = camera_id
        self.next_image_id = first_image_id
        self.render_color = render_color
        self.render_depth = render_depth
        self.estimate_depth = estimate_depth
        self.restore = restore
        self.generation_seconds = 0.0
        self.failure = None

    def __call__(self, round_index: int) -> None:
        result = run_generation_round(
            round_index,
            self.indices,
            self.poses,
            self.images,
            self.intrinsics,
            self.scene_dir,
            self.camera_id,
            self.next_image_id,
            self.render_color,
            self.render_depth,
            self.estimate_depth,
            self.restore,
        )
        self.generation_seconds += result["generation_seconds"]
        if "failure" in result:
            self.failure = result["failure"]
            return
        self.indices[:] = result["indices"]
        self.poses.update(result["poses"])
        self.images.update(result["images"])
        self.next_image_id = result["next_image_id"]
