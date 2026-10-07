# Count unregistered training views instead of filling their poses

i3dgs 跑完即一次完成的运行，即使部分 Dense View 或 Sparse View 没有位姿。这些 Dropped View 不进入 Pose Error，也不用真实位姿或 COLMAP 填上，同时报告丢掉的张数。渲染指标仍在得到的模型上计算。注册上的训练视角少于两张时，该场景的 Part 3 写成失败案例，列出未注册的帧，不用其他位姿把闭环续上。

i3dgs 在没有重叠时会搁置关键帧，位姿表只统计估计出来的相机（arXiv:2607.14481）。InstantSplat 与 BRPO 默认每张稀疏训练图都有位姿。只在注册子集上算 Pose Error 会偏好看，所以丢掉的张数必须一起报。

## Considered Options

- 只在有估计的训练视角上算 Pose Error，并报告丢掉的张数（采用）
- 用真实位姿或 COLMAP 填上缺的相机
- 有丢帧就让整个 Part 2 失败
