# Pinned upstream checkouts

Recorded 2026-10-01. These checkouts were not installed with their `requirements.txt`, so `torch` and `numpy` in `../env_cv` stayed unchanged.

| Path | Commit | Source |
| --- | --- | --- |
| `vggt` | `a288dd0f14786c93483e45524328726ab7b1b4ce` | https://github.com/facebookresearch/vggt |
| `vggt-omega` | `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6` | https://github.com/facebookresearch/vggt-omega |
| `gaussian-splatting` | `54c035f7834b564019656c3e3fcc3646292f727d` | https://github.com/graphdeco-inria/gaussian-splatting |

`gaussian-splatting` is the GitHub archive of that commit. A direct `git clone` stalled on this network, so that directory has no `.git`. Its CUDA submodules are the gitlinks from that commit:

- `submodules/diff-gaussian-rasterization` at `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`
- `submodules/fused-ssim` at `1272e21a282342e89537159e4bad508b19b34157`
- `submodules/simple-knn` at `86710c2d4b46680c02301765dd79e465819c8f19`

`SIBR_viewers` is the interactive viewer and is not present.

These Part 2 and Part 3 checkouts are GitHub archives of the recorded commits:

| Path | Commit | Source |
| --- | --- | --- |
| `i3dgs` | `cf4d5b9762359a1d6de76fb9abf7b3dc764c1a42` | https://github.com/graphdeco-inria/i3dgs |
| `GeoQuery` | `8439c4ffdf87783c9dbc29329c72df77ab13e01d` | https://github.com/Xiaoc7/GeoQuery |
| `Depth-Anything-3` | `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4` | https://github.com/ByteDance-Seed/Depth-Anything-3 |

