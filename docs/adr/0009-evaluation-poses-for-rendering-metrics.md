# Render test views at photometrically aligned evaluation poses

PSNR、SSIM、LPIPS 在 Evaluation Pose 上渲染。重建先冻结。每个 Test View 从最近的训练位姿起步，只用该测试图像的光度误差更新相机，高斯和训练位姿都不更新。不使用 Test View 的真实相机，也不用训练集 Sim(3) 去搬运真实测试相机。

CF-3DGS（CVPR 2024）和 InstantSplat（arXiv:2403.20309）用的是这套评测。i3dgs 还会让测试帧参与位姿训练；本项目不采用，因为 Test View 不得进入 i3dgs 的位姿估计。Pose Error 仍然只在训练视角上做 Sim(3) 对齐。

## Considered Options

- 冻结重建后，只用光度误差对齐测试相机（采用）
- 用训练集 Sim(3) 变换真实测试相机
- 让测试帧参与 i3dgs 的位姿训练，但不参与高斯优化
