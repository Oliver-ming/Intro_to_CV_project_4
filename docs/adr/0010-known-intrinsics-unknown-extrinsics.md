# Give i3dgs dataset intrinsics and withhold ground-truth poses

Unposed Sparse Reconstruction 只拒绝真实外参。i3dgs 接收数据集的焦距和主点，不接收真实位姿。必做表不再单开「估计内参」一列。

课程写的是相机位姿未知。CF-3DGS 的问题设定是无位姿图像加上内参；S3PO-GS 和 HI-SLAM2 也使用给定标定。i3dgs 与 InstantSplat 的原文会估计焦距，因此本项目的 Part 2 数字不与这两篇的原表逐项对应。锁死 Intrinsics 是为了和课程用语以及使用给定标定的无位姿方法对齐。

## Considered Options

- 使用数据集 Intrinsics，不给真实外参（采用）
- 让 i3dgs 自己估计焦距，以对齐其原文
- 内参和外参都不提供
