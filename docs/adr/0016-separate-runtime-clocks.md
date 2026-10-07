# Keep initialization, reconstruction, generation, and i3dgs clocks apart

四段耗时分列，不加进同一个格子。Initialization Time 从读入第一张图到 Initializer 写完相机和几何，不含 3DGS；VGGT 与 VGGT-Ω 只计默认前向。Reconstruction Time 只计标准 3DGS 循环，在 7000 和 30000 次各记一次。Generation Time 只计该轮 Depth Anything 3 与 GeoQuery。i3dgs 用 Joint Reconstruction Time，因为位姿和高斯是同一次运行。

Kerbl 等人 2023 只报告高斯优化时间。VGGT 把可选 BA 与前向分开。i3dgs 发表的是整段联合墙钟。把 COLMAP 和 3DGS 加在一起才能和 i3dgs 比端到端，那个和数不是 Kerbl 表里的格子，所以这里也不合并。

## Considered Options

- 四段墙钟分列（采用）
- 每个方法只报一个端到端时间
- 把 i3dgs 拆成初始化加重建
