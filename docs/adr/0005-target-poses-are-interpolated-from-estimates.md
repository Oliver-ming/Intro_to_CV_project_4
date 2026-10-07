# Interpolate target poses from estimated training views

每一轮在每一对当前相邻的 Training View 之间放一个 Target View。位姿只由这两个估计位姿做中点插值，平移取中点，旋转做球面插值。不用 Test View 的真实相机，也不用其他 Captured View 的真实相机。Generated View 写回时保留这个插值位姿。

Test View 正是每个抽帧间隔的时间中点。若用它的真实相机当 Target View，生成图会在测试视角上监督重建，渲染指标就不再是留出视角的结果。第二轮的相邻视角可以包括已经写回的 Generated View，新的 Target View 仍只从估计位姿插值。

## Considered Options

- 由相邻 Training View 的估计位姿中点插值（采用）
- 使用 Test View 或数据集中间帧的真实相机
- 每一轮只在整段轨迹中点放一个 Target View
