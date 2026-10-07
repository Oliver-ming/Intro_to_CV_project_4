# Continue the closed loop in standard 3DGS with frozen poses

Part 2 仍由 i3dgs 只根据 Dense View 或 Sparse View 的真实图像估计位姿和高斯。恰好两轮 Closed-Loop Generative Reconstruction 不把 Generated View 送回 i3dgs。i3dgs 会对每张新关键帧做匹配和局部光束法平差，从而改掉插值位姿；生成图上也常常没有足够的真实对应点（arXiv:2607.14481）。

两轮更新从导出的 i3dgs 结果开始，使用和 Part 1 相同的标准 3DGS。Sparse View 的位姿冻结在 i3dgs 的估计上，Generated View 的位姿冻结在插值上，两者光度监督相同。Target View 和 Reference Depth 由当前这份 3DGS 渲染。这与 GeoQuery 的写回方式一致：修复后的图像作为已知位姿的观测继续优化 3DGS（arXiv:2605.12399）。

## Considered Options

- 标准 3DGS，训练位姿在闭环期间冻结（采用）
- 把 Generated View 作为无序图像重新送入 i3dgs，让它自己估计位姿
