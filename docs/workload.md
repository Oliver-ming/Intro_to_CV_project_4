# Workload

按时间追加。每条记录对应一次会改变项目状态的决定或修改。

## 2026-10-01

- 读完 `Project_4_new__Generative_Sparse_View_3D_Reconstruction.pdf`。
- 实验机为 8 张 RTX 4090（每张 24GB）。Python 环境固定为 `../env_cv`（Python 3.10，当时尚未安装 PyTorch）。
- 术语：Closed-Loop Generative Reconstruction 定为必做阶段。它指把 Generated View 写回重建并至少做两轮，不是课程 PDF 里标成 Advanced 的一致性融合、视频扩散加速或主动视角规划。
- 尚未决定：Part 2 的重建骨架、Part 3 的生成路线、可选数据集是否纳入完成线。

## 2026-10-01 稍后

- Part 2 定为 Unposed Sparse Reconstruction，实现骨架是 i3dgs。同一场景比较全量 Captured View 与 Sparse View。
- 决定记在 `docs/adr/0001-i3dgs-for-unposed-sparse-reconstruction.md`。
- 尚未决定：Part 3 的生成路线、可选数据集是否纳入完成线。

## 2026-10-01 再稍后

- Part 3 定为 Image Diffusion Restoration：GeoQuery 修复 Target View 的渲染图，得到 Generated View，写回 i3dgs，至少两轮。
- BRPO 只约束抽帧和循环形状。不使用 BRPO 的 Difix3D 权重，也不走 GSFixer / GenFusion。
- 决定记在 `docs/adr/0002-geoquery-image-diffusion-for-the-closed-loop.md`。
- 查阅 BRPO 正文：Waymo 取 1/10 帧、DL3DV 每 30 帧取 1 帧作为输入；正文没有规定渲染指标的测试帧。
- 尚未决定：渲染指标打在哪些图上、可选数据集是否纳入完成线。

## 2026-10-01 测试帧

- 渲染指标只打在 Test View 上。Waymo-405841 为下标 % 10 == 5，Dl3dv-2 与 Re10k-1 为下标 % 30 == 15。帧按下标从 0 沿时间排序。
- Test View 不进入训练、位姿初始化、GeoQuery 参考，也不写回。Generated View 只能进训练。
- Part 1 与 Part 2 全视角设置的训练观测改称 Dense View：全部 Captured View 去掉 Test View。
- 决定记在 `docs/adr/0003-test-views-are-midpoint-captured-frames.md`。
- 尚未决定：ATE 怎么对齐、可选数据集是否纳入完成线。

## 2026-10-01 位姿误差

- Pose Error 定为 Sim(3) 对齐后的平移 RMSE。Part 1 与全视角设置算在 Dense View 上，稀疏设置算在 Sparse View 上。Test View 不计入。
- 决定记在 `docs/adr/0004-sim3-ate-on-training-views.md`。
- 尚未决定：Target View 的位姿从哪来、可选数据集是否纳入完成线。

## 2026-10-01 目标视角

- 新增 Training View：闭环开始时是 Sparse View，写回后还包括带着插值位姿的 Generated View。
- 每一轮、每一对相邻 Training View 之间放一个 Target View。位姿由估计位姿中点插值（平移中点、旋转球面插值），不用任何 Captured View 的真实相机。
- Generated View 写回时保留该插值位姿。
- 决定记在 `docs/adr/0005-target-poses-are-interpolated-from-estimates.md`。
- 尚未决定：GeoQuery 的参考图和深度从哪来、可选数据集是否纳入完成线。

## 2026-10-01 参考视角

- Reference View 定为离 Target View 最近的 Training View。GeoQuery 只查询这一张图。
- Reference Depth 定为当前 i3dgs 在该参考相机上渲染的深度，与估计位姿同一坐标系。不用 Test View、深度真值和外部深度模型。
- 决定记在 `docs/adr/0006-geoquery-uses-rendered-reference-depth.md`。
- 尚未决定：Generated View 写回后的监督权重、可选数据集是否纳入完成线。

## 2026-10-01 同等监督

- Generated View 写回后与 Sparse View 使用同一套光度损失，监督强度相同。
- 必做对比保留写回前、第一轮后、第二轮后。一致性融合仍不在必做范围内；第二轮变差就记成失败案例。
- 决定记在 `docs/adr/0007-equal-supervision-for-generated-views.md`。
- 尚未决定：可选数据集是否纳入完成线。

## 2026-10-01 完成线

- 当前完成线只有三个 Mandatory Scene：Waymo-405841、Dl3dv-2、Re10k-1。每个场景做 Part 1、Part 2，以及恰好两轮闭环。
- 可选数据集不在这轮完成线里。三个场景的表和图齐了之后，再考虑一个 Mip-NeRF 360 场景。
- 对照了 i3dgs（arXiv:2607.14481）和 GeoQuery（arXiv:2605.12399）：i3dgs 只收无序 RGB，并对新关键帧做匹配和局部 BA；GeoQuery 的写回是带着已知位姿继续优化 3DGS。
- 尚未决定：闭环两轮是写回 i3dgs，还是改到冻结位姿的标准 3DGS。

## 2026-10-01 闭环优化器

- 闭环两轮改到标准 3DGS。i3dgs 只负责 Part 2，输入只有 Dense View 或 Sparse View 的真实图像。
- Sparse View 位姿冻结在 i3dgs 估计上，Generated View 位姿冻结在插值上。生成图不送回 i3dgs。
- 依据：i3dgs（arXiv:2607.14481）对新关键帧做匹配和局部 BA；GeoQuery（arXiv:2605.12399）把修复图作为已知位姿观测继续优化 3DGS。
- 决定记在 `docs/adr/0008-closed-loop-uses-frozen-pose-3dgs.md`。
- 尚未决定：计算 PSNR 时，Test View 的真实相机如何变到重建坐标系。

## 2026-10-01 渲染相机调研

- [Protocol paper survey](936557b5-fcdd-43b3-93c2-d7b12f417607) 审阅了拟议规则：用训练集 Sim(3) 把 Test View 的真实相机变到重建坐标系再渲染。结论是不合规。
- CF-3DGS 与 InstantSplat 在算 PSNR/SSIM/LPIPS 时冻结重建，用测试图像只优化测试相机。i3dgs 让测试帧参与位姿训练、不参与高斯优化。Sim(3) 在这些论文里只用于 ATE。
- 尚未决定：本项目的渲染相机跟哪一条。

## 2026-10-01 评测位姿

- 渲染指标改在 Evaluation Pose 上计算。重建冻结，测试相机从最近训练位姿起步，只用该测试图像的光度误差更新，高斯和训练位姿不变。
- Test View 仍不进入训练、i3dgs 位姿估计和 GeoQuery。不采用 i3dgs「测试帧参与位姿训练」的做法。
- 决定记在 `docs/adr/0009-evaluation-poses-for-rendering-metrics.md`。
- 尚未决定：Unposed Sparse Reconstruction 能否使用数据集内参。

## 2026-10-01 内参

- Unposed Sparse Reconstruction 可以使用数据集的 Intrinsics（焦距和主点），不接收真实外参。
- 必做表不单开估计内参的一列。Part 2 因此不与 i3dgs、InstantSplat 原文的焦距设定逐项对应。
- 决定记在 `docs/adr/0010-known-intrinsics-unknown-extrinsics.md`。
- 尚未决定：标准 3DGS 的分辨率、迭代检查点和 LPIPS 主干。

## 2026-10-01 决定回溯

- 用论文回溯了 D1–D10。对照记在可打开的决定审计里。审计代理是 Decision audit against papers。
- 有论文依据：D4 训练集 Sim(3) ATE，D9 冻结模型后的光度测试相机。
- 有意偏离：D1 i3dgs 的稀疏程度、D2 不用 BRPO 的 Difix3D、D6 渲染深度、D7 同等监督、D8 冻结位姿、D10 锁死数据集内参。
- 论文里没有、但跟课程说明相容：D3 间隔中点测试帧，D5 估计位姿中点插值。
- 最站不住的是 D6。GeoQuery（arXiv:2605.12399）第 5.1 节用 Depth Anything 3 提供几何，训练深度来自 MVSFormer++，不是当前稀疏重建渲染的深度。
- 尚未改 D6。也尚未决定分辨率和迭代检查点。

## 2026-10-01 分辨率调研

- [Protocol paper survey](936557b5-fcdd-43b3-93c2-d7b12f417607) 审阅了「长边 1024、7k/30k、LPIPS-Alex」的拟议规则，结论是部分合规。
- Kerbl 的检查点是 7k 和 30k。宽度超过 1600 时把宽度缩到 1600，不是长边一律 1024。官方 `metrics.py` 的 LPIPS 用 VGG，不是 AlexNet。i3dgs 不强制跑 30k。
- 这条还没拿去问，因为 Reference Depth 要不要改成 Depth Anything 3 还没有答复。

## 2026-10-01 参考深度改为 Depth Anything 3

- Reference Depth 改为只在 Reference View 上跑 Depth Anything 3。Test View 不进入深度模型。
- 用参考图上渲染深度与 Depth Anything 3 深度的中位数比值做尺度系数，把深度对齐到当前重建的相机单位。
- 撤销 ADR 0006 的渲染深度。新决定记在 `docs/adr/0011-reference-depth-from-depth-anything-3.md`。
- 尚未决定：标准 3DGS 的分辨率、7k/30k 检查点和 LPIPS 是否用 VGG。

## 2026-10-01 分辨率与检查点

- Input Resolution：宽度超过 1600 时缩到 1600 并保持宽高比，否则保持原尺寸。COLMAP、VGGT、VGGT-Ω、i3dgs、标准 3DGS 共用这批像素。
- Reporting Checkpoint：标准 3DGS 在 7000 和 30000 次迭代记录 PSNR、SSIM、LPIPS（VGG）和耗时。三种初始化同一预算。i3dgs 不跑这个预算。
- 依据是 Kerbl 等人 2023 与官方 `metrics.py`。不与 S3PO-GS 的 Waymo 原分辨率或 DL3DV 256×256 逐项对应。
- 决定记在 `docs/adr/0012-kerbl-resolution-and-checkpoints.md`。
- 尚未决定：闭环的每一轮标准 3DGS 跑多少次迭代。

## 2026-10-01 闭环步数调研

- [Protocol paper survey](936557b5-fcdd-43b3-93c2-d7b12f417607) 否决了「每一轮生成之后再单独跑 30000 次」。
- DIFIX3D+ 大约每 1500 次迭代修复并写回一次。GeoQuery 沿用这条渐进日程，总步数可以到 60000，不是两段独立的 30k。GenFusion 大约每 1000 次生成一次。
- 尚未决定是否改成一次连续的 30000 次训练，并在约 1500 与 3000 次插入两轮生成。

## 2026-10-01 闭环插在同一次训练里

- 写回前基线是训完的 i3dgs。随后一次连续的 30000 次标准 3DGS，位姿冻结。
- 第 1500 次插入第一轮 Generated View。第 3000 次记第一轮之后并插入第二轮。第 4500 次记第二轮之后。7000 与 30000 次仍上报，此时两轮都已加入。
- 决定记在 `docs/adr/0013-two-fix-steps-inside-one-30k-run.md`。
- 尚未决定：有没有一条不插入生成图、但迭代数对齐的对照训练。

## 2026-10-01 无生成对照

- 增加 Sparse-Only Control：同一份 i3dgs 高斯、冻结位姿、只用 Sparse View、30000 次标准 3DGS，不插入 Generated View。
- 有无生成的对比对齐在第 1500、3000、4500、7000、30000 次。i3dgs 仍只代表 Part 2 和写回前基线。
- 决定记在 `docs/adr/0014-sparse-only-control-matches-iteration.md`。
- 尚未决定：Part 1 的标准 3DGS 是否冻结初始化位姿。

## 2026-10-01 Part 1 冻结位姿

- Initializer 指 COLMAP、VGGT、VGGT-Ω。它们只看见 Dense View。
- Part 1 的标准 3DGS 冻结这些位姿，只更新高斯。Pose Error 打在冻结位姿上。
- 初始高斯来自各自的点或深度，不共用 COLMAP 点云。
- 决定记在 `docs/adr/0015-part1-freezes-initializer-poses.md`。
- 尚未决定：初始化耗时、重建耗时和生成耗时各算哪一段。
- [Protocol paper survey](936557b5-fcdd-43b3-93c2-d7b12f417607) 认为分段计时符合 Kerbl、VGGT 与 i3dgs 的发表方式。Kerbl 只报高斯优化时间；VGGT 的可选 BA 不计入默认前向；i3dgs 用整段墙钟。这条尚未写入决定。

## 2026-10-01 分段耗时

- Initialization Time、Reconstruction Time、Generation Time、Joint Reconstruction Time 四段分列，不相加。
- VGGT 与 VGGT-Ω 的初始化只计默认前向。i3dgs 不拆成初始化加重建。
- 决定记在 `docs/adr/0016-separate-runtime-clocks.md`。
- 尚未决定：i3dgs 没能注册的训练视角如何计入 Pose Error 和失败案例。

## 2026-10-01 未注册视角

- Dropped View 不进入 Pose Error，不用真实位姿或 COLMAP 填上，并另报张数。渲染指标仍计算。
- 注册上的训练视角少于两张时，该场景的 Part 3 写成失败案例，不换别的位姿续跑。
- 决定记在 `docs/adr/0017-dropped-views-stay-out-of-pose-error.md`。
- 尚未决定：Evaluation Pose 的光度对齐停在什么时候。
- [Protocol paper survey](936557b5-fcdd-43b3-93c2-d7b12f417607) 指出 CF-3DGS 没有公布测试相机的步数；InstantSplat 是每张测试图 500 次。100 步没有出处。这条尚未写入决定。

## 2026-10-01 评测位姿步数

- Evaluation Pose 每张 Test View 单独做 500 次 Adam 更新后停止。不用真实测试位姿再对齐。
- 学习率沿用 InstantSplat 发布代码的默认值，不另设没有出处的数。
- 决定记在 `docs/adr/0018-evaluation-pose-stops-at-500-steps.md`。
- 度量规则到此齐了。尚未决定代码目录怎么分。

## 2026-10-01 目录

- 建了 `src/data`、`src/init`、`src/reconstruct`、`src/generate`、`src/eval`、`src/run`，以及 `third_party/`、`configs/`。`README.md` 说明边界。
- 决定记在 `docs/adr/0019-src-stages-and-third-party.md`。
- 尚未决定：`env_cv` 里安装哪一版 PyTorch。调研认为 VGGT 锁 `torch==2.3.1`，与 i3dgs 的 `2.7.1+cu128`、VGGT-Ω 的 `torch>=2.6` 冲突。

## 2026-10-01 已有的 PyTorch

- 查看终端 `Install PyTorch CUDA 12.8 into env_cv`：安装已经结束，退出码 0，不是仍在安装。工作目录是 Project_2，装进的是共用的 `env_cv`。
- 实际版本是 `torch 2.11.0+cu128`、`torchvision 0.26.0+cu128`、`numpy 2.2.6`。`torch.cuda.is_available()` 为真，可见 8 张 GPU。
- 这比先前建议的 `torch==2.7.1` 新，而且 numpy 已经大于等于 2。Project_2 正在用这个环境跑任务，此时降级会打断它。
- 尚未决定：Project 4 是否就用这一版，而不是改回 2.7.1。

## 2026-10-01 PyTorch 版本

- Project 4 使用 `env_cv` 里已有的 `torch 2.11.0+cu128`、`torchvision 0.26.0+cu128`。不降级，也不执行 VGGT 的 `torch==2.3.1`。
- `numpy 2.2.6` 保持不动。以后某个上游因 numpy 2.x 装不上时再单独处理。
- 决定记在 `docs/adr/0020-shared-torch-2.11.0-cu128.md`。
- 尚未决定：先把哪些上游仓库固定到 `third_party/`。

## 2026-10-01 上游固定

- `third_party/vggt` 为 `a288dd0f14786c93483e45524328726ab7b1b4ce`。
- `third_party/vggt-omega` 为 `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6`。
- `third_party/gaussian-splatting` 为 `54c035f7834b564019656c3e3fcc3646292f727d`。直接 `git clone` 在这台机器上停住，所以用该 commit 的 GitHub 归档。CUDA 子模块按该 commit 的 gitlink 放好。没有下载 SIBR 查看器。
- 没有执行这三个仓库的 `requirements.txt`。清单在 `third_party/PINS.md`。
- 尚未决定：是否用 `/usr/local/cuda/bin/nvcc` 编译 3DGS 的 CUDA 扩展。

## 2026-10-01 开始实现

- 目标是完成 Project 4 的三部分和三个必做场景。训练任务至少隔 15 分钟才查看一次，完成与否看进程退出、日志或完成标记。
- 已用 `env_cv` 开始编译 `diff-gaussian-rasterization`、`simple-knn`、`fused-ssim`，日志在 `outputs/build_3dgs_extensions.log`。`CUDA_HOME=/usr/local/cuda`（nvcc 12.9）。
- `src/data/splits.py` 实现了 Sparse / Test / Dense 划分和宽度 1600 的缩放。`tests/test_splits.py` 通过。
- 机器上还没有 Waymo-405841、Dl3dv-2、Re10k-1。COLMAP 3.8 在 `/usr/local/bin/colmap`。

## 2026-10-01 CUDA 扩展第一次编译失败

- `diff_gaussian_rasterization` 编译失败，退出码 1。缺 `glm/glm.hpp`。归档里的 `third_party/glm` 是空的嵌套子模块。
- 正在补 GLM `5c46b9c07008ae65cb81ab79cd677ecc1934b903`。补齐后再编。
- i3dgs、GeoQuery、Depth-Anything-3 的代码归档已经放在 `third_party/`。

## 2026-10-01 CUDA 扩展编译完成

- 补上 GLM 之后重新编译，日志出现 `BUILD_OK`，退出码 0。
- `diff_gaussian_rasterization`、`simple_knn`、`fused_ssim` 都能在 `env_cv` 里导入。

## 2026-10-01 i3dgs 扩展与生成写回

- i3dgs 的 `diff_gaussian_rasterization` 和 `simple_knn` 编进 `third_party/i3dgs/py_ext`。日志有 `I3DGS_EXT_OK`。不设置 `PYTHONPATH` 时，环境里的 Part 1 光栅器没有 `adamUpdate`。
- 生成视角写回 `images.txt` 之后可以读回相机到世界的位姿。这个位姿转成的 `world_view_transform` 与官方 3DGS 的 `getWorld2View2` 一致。
- Part 2 命令显式加上 `--viewer_mode none`。默认值本来就是 `none`，写出来是为了训练不依赖显示。
















