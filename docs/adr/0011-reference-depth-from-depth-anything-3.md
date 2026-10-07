# Align Depth Anything 3 depth into the reconstruction scale

Reference Depth 改为在 Reference View 上运行 Depth Anything 3，不使用当前稀疏重建渲染出来的深度，也不把 Test View 送进深度模型。GeoQuery 用这份深度和估计位姿建立对应（arXiv:2605.12399，第 5.1 节；训练时的深度来自 MVSFormer++）。

Depth Anything 3 的深度是度量尺度，i3dgs 的相机是任意尺度。直接混用会对错像素。因此用一个尺度系数把 Depth Anything 3 的深度乘进当前重建的相机单位：该系数是参考图上渲染深度与 Depth Anything 3 深度的中位数比值。对应场仍用已经估计好的位姿。

## Considered Options

- Depth Anything 3，再按渲染深度做一次尺度对齐（采用）
- 直接使用重建渲染的深度
- 使用未对齐的度量深度
