# Project 4

https://github.com/Oliver-ming/Intro_to_CV_project_4

Generative sparse-view 3D reconstruction. Domain language is in `CONTEXT.md`. Decisions are in `docs/adr/`. The work log is `docs/workload.md`.

`third_party/` holds pinned upstream checkouts. The commits are listed in `third_party/PINS.md`. Experiment logic stays in `src/`.

- `src/data` — view splits and input resolution
- `src/init` — COLMAP, VGGT, VGGT-Ω
- `src/reconstruct` — standard 3DGS and i3dgs
- `src/generate` — target views, Depth Anything 3, GeoQuery
- `src/eval` — pose error, evaluation poses, metrics, clocks
- `src/run` — per-scene entry points

`configs/` holds experiment configs. `data/`, `weights/`, and `outputs/` are not in git. The Python environment is `../env_cv`.

Put one scene's captured frames in `data/raw/<scene>/`, with file name order equal to time order. Then:

```bash
../env_cv/bin/python -m src.data.prepare \
  --scene waymo-405841 \
  --source data/raw/waymo-405841 \
  --destination data/prepared/waymo-405841
```

The same command accepts `dl3dv-2` and `re10k-1`. It writes resized images and `split.json`. Re10k-1 in the course zip is imported with `src.data.import_s3po.import_re10k`: png frames in filename order, `video.mp4` left out, and `cameras.json` kept beside the raw frames.

S3PO archives are imported automatically when their download logs contain `WAYMO_OK` or `DL3DV_OK`:

```bash
../env_cv/bin/python -m src.run.ingest_when_ready --scene waymo-405841 --project-root .
../env_cv/bin/python -m src.run.ingest_when_ready --scene dl3dv-2 --project-root .
```

Part 1 COLMAP is `src.run.launch_part1`. It trains on Dense Views with the dataset camera: Waymo uses `OPENCV` and its distortion, then undistorts that model to `PINHOLE` before 3DGS. DL3DV uses `PINHOLE` from `cameras.json`. The model is converted to text, the binary COLMAP files are removed, then standard 3DGS trains for 30000 iterations with the viewer off and poses frozen. Success is `outputs/part1/<scene>/TRAIN_PART1_OK`.

Part 1 also has two feed-forward initializers. VGGT is `src.init.vggt.launch_vggt`: it runs `demo_colmap.py` without `--use_ba`, then moves the model into `sparse/0`. VGGT-Ω would unproject its depth with `src.init.export_colmap.export_omega_scene`, keeping the dataset intrinsics at the shared image size. The public checkpoint `facebook/VGGT-Omega` is gated. This machine is not logged in, so the download returns 401 and no scene is initialized. That failure is `outputs/part1/vggt_omega_failure.json`.

Part 2 runs `src.run.wait_part2` after a scene is prepared and `graphdecoviewer` imports. It trains dense views, then sparse views, with `--fix_focal` and the dataset focal length scaled to the prepared width. Its CUDA extensions live in `third_party/i3dgs/py_ext` and must come first on `PYTHONPATH`, so they do not replace the Part 1 rasterizer. No ground-truth extrinsics are passed. Success is `TRAIN_PART2_DENSE_OK` and `TRAIN_PART2_SPARSE_OK`. The joint reconstruction clock is the whole i3dgs run. Rendering metrics use the exported checkpoint at iteration 0: `src.eval.score_part2.score_i3dgs_export` writes PSNR, SSIM, and VGG LPIPS on the test views, with dropped views left out of the pose error.

Part 3 starts from the sparse i3dgs export. `src.run.export_i3dgs_scene` keeps leaf Gaussians, recenters them with the registered cameras so the coordinates stay in a range 3DGS can rasterize, writes their checkpoint as `chkpnt0.pth`, and writes only the registered training views with those frozen poses. The sparse-only control is `src.run.launch_control`: 30000 iterations, no generated views, checkpoints at 1500, 3000, 4500, 7000, and 30000, then the same test-view metrics. Success is `TRAIN_CONTROL_OK`. The closed loop is `src.run.closed_loop_scene`: it loads that checkpoint, trains to 1500, inserts the first GeoQuery round, resumes to 3000, inserts the second round, then resumes through 30000. Depth Anything 3 runs on the nearest training view and is scaled by the median ratio to the rendered depth. The GeoQuery file is the window-size-3, down-1 checkpoint. Images wider than 512 pixels are restored at long side 512 and resampled back to the camera size, because the SD-Turbo UNet at 1600 pixels does not fit in 24 GB. Success is `CLOSED_LOOP_OK`.




