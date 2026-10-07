# Use GeoQuery image diffusion for the closed loop

Closed-Loop Generative Reconstruction 用图像扩散，不用视频扩散。每一轮在两个相邻 Sparse View 之间放一个 Target View，用当前 i3dgs 渲染，再用 GeoQuery 做 Image Diffusion Restoration，把结果作为 Generated View 写回，至少两轮。

BRPO 只提供抽帧规则和「渲染—修复—写回」的循环形状。BRPO 自己的扩散权重是 Difix3D，本项目不采用。视频路线（GSFixer、GenFusion）要生成一整段轨迹，和课程默认的单视角插值不是同一种观测。

## Considered Options

- GeoQuery 图像扩散，循环形状跟随 BRPO（采用）
- GSFixer 或 GenFusion 视频扩散
- 直接使用 BRPO 的 Difix3D 权重
