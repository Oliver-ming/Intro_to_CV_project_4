# Report pose error as Sim(3) ATE on training views

Pose Error 是一次 Sim(3) 对齐之后的平移 RMSE。COLMAP 和 i3dgs 的尺度是任意的，只用旋转和平移对齐会让尺度差盖过轨迹形状。VGGT 即使带有尺度，也走同一套对齐，三列初始化才能并排比较。

Part 1 和 Part 2 全视角设置只在 Dense View 的估计位姿上计算。稀疏设置只在 Sparse View 上计算。Test View 没有参与位姿优化，不进入这个数字。

## Considered Options

- Sim(3) 对齐后的平移 RMSE，只在训练视角上计算（采用）
- 只做 SE(3) 对齐，保留尺度误差
- 把 Test View 的位姿也算进 ATE
