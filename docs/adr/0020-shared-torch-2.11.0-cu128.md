# Use the shared env_cv torch 2.11.0+cu128

Project 4 使用已经装在 `../env_cv` 里的 `torch 2.11.0+cu128` 和 `torchvision 0.26.0+cu128`。不降级到 i3dgs 文档里的 `torch==2.7.1`，也不执行 VGGT 锁定的 `torch==2.3.1`。当前 `numpy` 是 2.2.6，现在不把它降到 2 以下。

这个环境同时在跑其他项目的任务。现在改版本会打断那些任务。VGGT-Ω 或 Depth Anything 3 如果之后因为 numpy 2.x 装不上，再针对那一次安装处理。

## Considered Options

- 保持 `torch 2.11.0+cu128`（采用）
- 降级到 `torch==2.7.1` 以对齐 i3dgs 文档
- 按 VGGT 的 `torch==2.3.1` 重装
