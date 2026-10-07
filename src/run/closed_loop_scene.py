"""Run one scene's closed loop from an exported i3dgs checkpoint."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import torch

from src.generate.models import estimate_with_da3, restore_with_geoquery
from src.generate.state import GenerationState
from src.reconstruct.render_view import color_to_image, load_gaussians, render_view
from src.run.launch_part3 import launch_part3
from src.run.scene_views import load_training_views, read_intrinsics

ROOT = Path(__file__).resolve().parents[2]
DA3_SRC = ROOT / "third_party" / "Depth-Anything-3" / "src"
GEOQUERY_SRC = ROOT / "third_party" / "GeoQuery" / "src"


def latest_checkpoint(model_dir: Path) -> Path:
    """The highest-iteration 3DGS checkpoint in ``model_dir``."""
    paths = list(model_dir.glob("chkpnt*.pth"))
    if not paths:
        raise FileNotFoundError(f"no checkpoint in {model_dir}")
    return max(paths, key=lambda path: int(path.stem.removeprefix("chkpnt")))


def _camera_size(cameras_path: Path) -> tuple[int, int]:
    for line in cameras_path.read_text().splitlines():
        if line and not line.startswith("#"):
            parts = line.split()
            return int(parts[2]), int(parts[3])
    raise ValueError(f"no camera in {cameras_path}")


def _live_renderers(model_dir: Path, state: GenerationState, width: int, height: int, focal_x: float, focal_y: float, sh_degree: int):
    """Render from the newest checkpoint and the poses the rounds have added."""
    cache: dict = {}

    def model():
        path = latest_checkpoint(model_dir)
        if cache.get("path") != path:
            cache["path"] = path
            cache["model"] = load_gaussians(path, sh_degree)
        return cache["model"]

    def render_color(pose: torch.Tensor):
        rendered = render_view(model(), pose.detach().cpu().numpy(), width, height, focal_x, focal_y)
        return color_to_image(rendered["color"])

    def render_depth(index: int) -> torch.Tensor:
        pose = state.poses[index]
        rendered = render_view(model(), pose.detach().cpu().numpy(), width, height, focal_x, focal_y)
        return rendered["depth"].detach().cpu()

    return render_color, render_depth


def load_da3():
    """Depth Anything 3 base model. Inference stays on one reference view."""
    if str(DA3_SRC) not in sys.path:
        sys.path.insert(0, str(DA3_SRC))
    from depth_anything_3.api import DepthAnything3

    model = DepthAnything3.from_pretrained("depth-anything/DA3-BASE")
    return model.to("cuda").eval()


def load_geoquery(checkpoint: Path):
    """GeoQuery restoration. The public file is the window-size-3, down-1 checkpoint.

    The constructor defaults are neighborhood 5 and every UNet block. This file
    only contains the ``down_1`` modules, and its position table is 3 by 3.
    """
    if str(GEOQUERY_SRC) not in sys.path:
        sys.path.insert(0, str(GEOQUERY_SRC))
    from model import GeoQuery

    return GeoQuery(
        pretrained_path=str(checkpoint),
        neighborhood_size=3,
        low_res_only=True,
    )


def _restore_without_da3(da3, geoquery, *args):
    """GeoQuery needs the card. Depth Anything 3 is idle during the diffusion step."""
    da3.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    try:
        return restore_with_geoquery(geoquery, *args)
    finally:
        da3.to("cuda")


def execute_closed_loop_scene(
    source: Path,
    model_dir: Path,
    python: str,
    start_checkpoint: Path,
    sh_degree: int,
    da3,
    geoquery,
) -> dict:
    """Train the three segments and insert both generation rounds.

    Reconstruction time is only the 3DGS processes. Generation time is the two
    Depth Anything 3 and GeoQuery rounds.
    """
    indices, poses, images = load_training_views(source)
    intrinsics = read_intrinsics(source / "sparse" / "0" / "cameras.txt")
    width, height = _camera_size(source / "sparse" / "0" / "cameras.txt")
    focal_x = float(intrinsics[0, 0])
    focal_y = float(intrinsics[1, 1])
    state = GenerationState(
        indices,
        poses,
        images,
        intrinsics,
        source,
        1,
        1_000_000,
        lambda _pose: None,
        lambda _index: torch.zeros(1, 1),
        lambda image: estimate_with_da3(da3, image),
        lambda *args: _restore_without_da3(da3, geoquery, *args),
    )
    render_color, render_depth = _live_renderers(model_dir, state, width, height, focal_x, focal_y, sh_degree)
    state.render_color = render_color
    state.render_depth = render_depth
    log_path = model_dir / "train.log"
    model_dir.mkdir(parents=True, exist_ok=True)

    def run(command: list[str]) -> None:
        # The 3DGS process needs the card. Diffusion weights stay in memory
        # and move back before the next generation round.
        da3.to("cpu")
        geoquery.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            with log_path.open("a") as handle:
                completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
        finally:
            da3.to("cuda")
            geoquery.to("cuda")
        if completed.returncode != 0:
            raise RuntimeError(f"3DGS exited {completed.returncode}; see {log_path}")

    started = time.perf_counter()
    result = launch_part3(source, model_dir, python, state, run, start_checkpoint)
    result["reconstruction_seconds"] = time.perf_counter() - started - result["generation_seconds"]
    return result


def main() -> None:
    """Train both generation rounds, then score the closed-loop checkpoints."""
    import argparse
    import json

    from src.eval.score_scene import write_checkpoint_metrics
    from src.run.launch_part1 import ground_truth_centers

    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--geoquery", type=Path, required=True)
    parser.add_argument("--python", required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--sh-degree", type=int, default=3)
    args = parser.parse_args()
    da3 = load_da3()
    geoquery = load_geoquery(args.geoquery)
    result = execute_closed_loop_scene(
        args.source,
        args.model,
        args.python,
        args.checkpoint,
        args.sh_degree,
        da3,
        geoquery,
    )
    clocks = args.model / "clocks.json"
    clocks.write_text(json.dumps({
        "reconstruction_seconds": result["reconstruction_seconds"],
        "generation_seconds": result["generation_seconds"],
        "failure": result["failure"],
    }, indent=2) + "\n")
    if result["failure"] is not None:
        raise SystemExit(result["failure"])
    centers = ground_truth_centers(args.scene, args.prepared)
    for iteration in (1500, 3000, 4500, 7000, 30000):
        write_checkpoint_metrics(
            args.scene,
            args.prepared,
            args.source,
            args.model,
            args.metrics,
            centers,
            iteration,
            0.0,
            result["reconstruction_seconds"],
            training_split="sparse",
            run_name="closed-loop",
            generation_seconds=result["generation_seconds"],
        )
    (args.model / "CLOSED_LOOP_OK").write_text("two-rounds\n")


if __name__ == "__main__":
    main()
