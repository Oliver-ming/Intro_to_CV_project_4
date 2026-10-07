# Generative Sparse-View 3D Reconstruction

AIAA 3201 Project 4 的领域语言：在观测不足时，用三维重建和生成观测补全同一场景。

## Language

### Views

**Captured View**:
数据集里实际拍到的图像及其对应相机。
_Avoid_: 输入帧, 原图

**Sparse View**:
按时间排序、下标从 0 起，被抽出来做稀疏训练的 Captured View。Waymo-405841 取下标能被 10 整除的帧，Dl3dv-2 与 Re10k-1 取下标能被 30 整除的帧。
_Avoid_: Generated View, 新视角

**Test View**:
只用于计算 PSNR、SSIM、LPIPS 的 Captured View。它不进入训练、i3dgs 的位姿估计、GeoQuery 参考，也不写回重建。渲染时使用它的 Evaluation Pose，不用它的真实相机。Waymo-405841 取下标除以 10 余 5 的帧，Dl3dv-2 与 Re10k-1 取下标除以 30 余 15 的帧。
_Avoid_: Held-out View, 验证帧, novel view

**Evaluation Pose**:
在冻结的重建上，为某个 Test View 求出的相机。它从最近的已注册训练位姿起步，只用这张测试图像的光度误差做 500 次 Adam 更新，然后停止。高斯和训练位姿都不因此改变。
_Avoid_: 位姿初始化, 真实测试相机, Sim(3) 测试相机

**Dense View**:
去掉 Test View 之后剩下的 Captured View。Part 1，以及 Part 2 的全视角设置，用它训练。
_Avoid_: 全量帧, all frames

**Training View**:
当前这一轮允许监督重建的视角。闭环开始时只有 Sparse View；写回之后还包括带着插值位姿的 Generated View。
_Avoid_: Dense View, Test View

**Target View**:
某一轮里，插在两个当前相邻 Training View 之间的相机。位姿由这两个视角的估计位姿做中点插值：平移取中点，旋转做球面插值。它不使用任何 Captured View 的真实相机，本身也还不是一张图像。
_Avoid_: Test View, novel view, 真实中间帧

**Reference View**:
离某个 Target View 最近的 Training View。GeoQuery 只能查询它的图像。它不能是 Test View。
_Avoid_: 最近的真实帧, 邻帧真值

**Reference Depth**:
在 Reference View 上由 Depth Anything 3 估计、再乘上一个尺度系数后的深度。该系数是这张参考图上渲染深度与 Depth Anything 3 深度的中位数比值，用来对齐到当前重建的相机单位。不使用 Test View。
_Avoid_: 稀疏重建渲染深度, 深度真值, 未对齐的度量深度

**Generated View**:
Image Diffusion Restoration 的输出。它带着 Target View 的插值位姿写回重建，并与 Sparse View 接受同等的光度监督。
_Avoid_: 伪视角, novel view, 补图, 降权伪视角

**Dropped View**:
i3dgs 跑完后仍没有估计位姿的训练视角。Pose Error 不计入它，也不用真实位姿或 COLMAP 填上，同时另报丢掉的张数。某个场景注册上的训练视角少于两张时，该场景不做闭环，并写成失败案例。
_Avoid_: 用真值补位姿, 静默跳过

### Cameras

**Intrinsics**:
数据集给出的焦距和主点。无位姿设置仍然把它当作已知标定，但不接收真实外参。
_Avoid_: 外参, 位姿, 自己估计的焦距

**Initializer**:
Part 1 里只看见 Dense View、并给出相机和初始几何的方法。三者是 COLMAP、VGGT、VGGT-Ω。随后的标准 3DGS 冻结它的位姿，初始高斯来自它自己的点或深度，不共用另一方法的点云。
_Avoid_: 位姿细化, 共用点云

**Frozen Pose**:
标准 3DGS 训练期间不再更新的相机。Part 1 用 Initializer 的估计。闭环和 Sparse-Only Control 用 i3dgs 的估计，Generated View 用插值位姿。
_Avoid_: 光度位姿细化, Gaussian BA

### Stages

**Unposed Sparse Reconstruction**:
不接收真实外参的重建。可以使用数据集的 Intrinsics。同一场景要比较 Dense View 训练和 Sparse View 训练，两者都不得使用 Test View。本项目这一阶段使用 i3dgs。
_Avoid_: 无标定重建, monocular 3DGS-SLAM, COLMAP 初始化

**Image Diffusion Restoration**:
用 GeoQuery 修复 Target View 上的渲染图。查询的几何来自一张 Reference View 和它的 Reference Depth。修复结果才是 Generated View。
_Avoid_: 视频扩散, GSFixer, GenFusion, Difix3D

**Closed-Loop Generative Reconstruction**:
从 i3dgs 在 Sparse View 上导出的高斯出发，做一次连续的 30000 次标准 3DGS。第 1500 次插入第一轮 Generated View，第 3000 次记为第一轮之后并插入第二轮，第 4500 次记为第二轮之后。位姿在这 30000 次里保持冻结，生成图不送回 i3dgs。
_Avoid_: 每轮单独 30000 次, 把生成图送回 i3dgs, Part 3 进阶项

**Sparse-Only Control**:
从同一份 i3dgs 高斯出发、位姿同样冻结、只使用 Sparse View 的 30000 次标准 3DGS。它不插入 Generated View。有无生成的对比在第 1500、3000、4500、7000 和 30000 次上对齐。它不是 Part 2 的 i3dgs 结果。
_Avoid_: i3dgs 基线, 无生成的 i3dgs

### Metrics

**Pose Error**:
估计位姿相对真实位姿的轨迹误差。先做一次 Sim(3) 对齐，再报告平移的 RMSE。Part 1 和全视角设置算在 Dense View 上，稀疏设置算在 Sparse View 上。Test View 不计入。
_Avoid_: SE(3) ATE, 绝对尺度误差

**Input Resolution**:
COLMAP、VGGT、VGGT-Ω、i3dgs 和标准 3DGS 共用的像素。图像宽度超过 1600 时，把宽度缩放到 1600 并保持宽高比；否则保持原尺寸。
_Avoid_: 长边 1024, 各方法各用各的分辨率

**Reporting Checkpoint**:
标准 3DGS 在 7000 次和 30000 次迭代时记录的 PSNR、SSIM、LPIPS 和耗时。LPIPS 使用 VGG。三种初始化使用同一个迭代预算。i3dgs 不使用这个预算。
_Avoid_: AlexNet LPIPS, 提前停止

**Initialization Time**:
从读入第一张图到 Initializer 写完相机和几何的墙钟。不含随后的 3DGS。VGGT 与 VGGT-Ω 只计默认前向，不计额外的光束法平差。
_Avoid_: 端到端时间, 含训练的初始化

**Reconstruction Time**:
只含标准 3DGS 训练循环的墙钟，在 7000 次和 30000 次各记一次。不含初始化、Depth Anything 3 和 GeoQuery。
_Avoid_: 端到端时间

**Generation Time**:
某一轮修复里 Depth Anything 3 与 GeoQuery 的墙钟。它单独记录，不并进 Reconstruction Time。
_Avoid_: 重建耗时

**Joint Reconstruction Time**:
i3dgs 一次完整运行的墙钟。位姿和高斯一起估计，不拆成 Initialization Time 和 Reconstruction Time。
_Avoid_: 初始化耗时, 重建耗时

### Scenes

**Mandatory Scene**:
当前完成线里的场景：Waymo-405841、Dl3dv-2、Re10k-1。每个场景都要做完 Part 1、Part 2，以及恰好两轮闭环。
_Avoid_: 必做数据, benchmark
