"""Checks that one Gaussian in front of the camera produces a non-black image."""

import unittest

import numpy as np
import torch
from torch import nn

from src.reconstruct.render_view import color_to_image, render_view


class RenderViewTest(unittest.TestCase):
    def test_a_gaussian_in_front_is_visible(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parents[1] / "third_party" / "gaussian-splatting"
        sys.path.insert(0, str(root))
        from scene.gaussian_model import GaussianModel

        model = GaussianModel(0)
        model.active_sh_degree = 0
        model._xyz = nn.Parameter(torch.tensor([[0.0, 0.0, 4.0]], device="cuda"))
        model._features_dc = nn.Parameter(torch.ones(1, 1, 3, device="cuda"))
        model._features_rest = nn.Parameter(torch.zeros(1, 0, 3, device="cuda"))
        model._scaling = nn.Parameter(torch.full((1, 3), 0.0, device="cuda"))
        model._rotation = nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cuda"))
        model._opacity = nn.Parameter(torch.tensor([[5.0]], device="cuda"))
        rendered = render_view(model, np.eye(4), 32, 16, 16.0, 16.0)
        self.assertEqual(tuple(rendered["color"].shape), (3, 16, 32))
        self.assertEqual(tuple(rendered["depth"].shape), (16, 32))
        self.assertGreater(float(rendered["color"].detach().max()), 0.0)
        self.assertEqual(color_to_image(rendered["color"]).size, (32, 16))

    def test_checkpoint_reload_still_renders(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        import tempfile
        from pathlib import Path

        from src.reconstruct.export_gaussians import write_gaussian_checkpoint
        from src.reconstruct.render_view import generation_renderers, load_gaussians

        with tempfile.TemporaryDirectory() as raw:
            path = Path(raw) / "chkpnt0.pth"
            write_gaussian_checkpoint(
                path,
                torch.tensor([[0.0, 0.0, 4.0]]),
                torch.ones(1, 1, 3),
                torch.zeros(1, 0, 3),
                torch.zeros(1, 3),
                torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                torch.tensor([[5.0]]),
                sh_degree=0,
            )
            model = load_gaussians(path, 0)
        render_color, render_depth = generation_renderers(model, {0: torch.eye(4)}, 32, 16, 16.0, 16.0)
        image = render_color(torch.eye(4))
        depth = render_depth(0)
        self.assertEqual(image.size, (32, 16))
        self.assertEqual(tuple(depth.shape), (16, 32))
        self.assertGreater(max(image.convert("L").getextrema()), 0)


if __name__ == "__main__":
    unittest.main()
