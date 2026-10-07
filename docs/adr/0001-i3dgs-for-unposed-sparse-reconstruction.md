# Use i3dgs for unposed sparse reconstruction

Part 2 需要在没有 COLMAP 位姿的情况下重建 3DGS，并比较全量视角和稀疏视角。我们用 i3dgs，不用 HI-SLAM2 或 S3PO-GS。i3dgs 和 Part 1 一样落在 3DGS 优化上，Closed-Loop Generative Reconstruction 可以把 Generated View 写回同一次重建。单目 3DGS-SLAM 会把这次写回绑到跟踪、尺度和增量建图上。

## Considered Options

- i3dgs（采用）
- HI-SLAM2
- S3PO-GS
