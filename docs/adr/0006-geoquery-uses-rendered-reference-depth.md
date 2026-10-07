# Condition GeoQuery on the nearest training view and rendered depth

Status: superseded by ADR 0011.

原先采用最近的 Training View，加上当前重建渲染的深度。该深度选择已撤销。参考图仍是最近的 Training View。

Image Diffusion Restoration 的参考图是离 Target View 最近的 Reference View，也就是一张 Training View。深度是当前 i3dgs 在该参考相机上渲染的 Reference Depth，与估计位姿同一坐标系。不使用 Test View，不使用数据集深度真值，也不另跑深度模型。

GeoQuery 原文可以用外部度量深度。那样会引入第二套尺度，也可能让测试帧的几何进入修复。渲染深度和插值位姿来自同一次重建，反投影才和 Target View 的相机一致。

## Considered Options

- 最近的 Training View，加上 i3dgs 渲染深度（采用）
- 用 Depth Anything 或 MVS 给参考图估深度
- 参考两张相邻训练图，沿用 BRPO 的双向融合
