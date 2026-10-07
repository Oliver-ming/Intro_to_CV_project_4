# Stop evaluation-pose alignment after 500 Adam steps

每张 Test View 的 Evaluation Pose 单独优化 500 次 Adam 步，只更新这一个相机，然后停止。重建保持冻结。起点是最近的已注册训练位姿。不用真实测试位姿再对齐一次。

CF-3DGS 没有公布测试相机的步数。500 次来自 InstantSplat（arXiv:2403.20309）。学习率取 InstantSplat 发布代码里的默认值，不另设一个没有出处的数。因此渲染指标对齐的是 InstantSplat 的测试相机预算，不保证与 CF-3DGS 的原表相同。

## Considered Options

- 每张测试图 500 次 Adam（采用）
- 100 次
- 用真实测试位姿放置相机
