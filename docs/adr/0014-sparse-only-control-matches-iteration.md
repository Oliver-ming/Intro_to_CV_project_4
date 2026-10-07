# Compare generation against a matched sparse-only 3DGS control

有无生成视角的对比使用 Sparse-Only Control，而不是拿 i3dgs 直接和闭环结果比。对照从同一份导出的 i3dgs 高斯出发，位姿冻结，只使用 Sparse View，跑 30000 次标准 3DGS，不插入 Generated View。两边都在第 1500、3000、4500、7000 和 30000 次记录指标。i3dgs 仍是 Part 2 的结果，也是写回前的基线。

DIFIX3D+、GeoQuery 和 GenFusion 的无生成对照都是同一套重建骨干、不加入生成图。它们通常只发表最终指标。这里额外保留中间步，是为了让课程要求的逐轮对比落在相同迭代数上。

## Considered Options

- 同一初始化、同一位姿、同一 30000 次日程的无生成对照（采用）
- 只用训完的 i3dgs 充当无生成结果
