# Freeze Part 1 poses and keep each initializer's own geometry

Part 1 的标准 3DGS 不更新相机，只更新高斯。COLMAP、VGGT、VGGT-Ω 的 Dense View 位姿在 30000 次迭代里保持冻结。Pose Error 打在这些冻结位姿上，不打在光度细化之后的位姿上。初始高斯来自各自 Initializer 的点或深度，三列不共用一份 COLMAP 点云。

Kerbl 等人 2023 的 3DGS 把 SfM 相机当作固定输入。COLMAP 的光束法平差发生在导出之前。冻结位姿才能让收敛速度和最终渲染对应初始化差异。InstantSplat 等会在 3DGS 里继续修位姿，那种渲染不作为本表的做法。

## Considered Options

- 冻结 Initializer 位姿，各自使用自己的初始几何（采用）
- 训练中按光度误差联合细化位姿
- 三列共用 COLMAP 点云，只更换相机
