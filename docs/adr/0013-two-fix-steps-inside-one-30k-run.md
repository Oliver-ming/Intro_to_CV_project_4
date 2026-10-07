# Insert both generation rounds into one 30k 3DGS run

Closed-Loop Generative Reconstruction 是一次连续的 30000 次标准 3DGS，不是两段各自 30000 次的重训。写回前的基线是训完的 i3dgs。高斯从 i3dgs 导出，位姿冻结。第 1500 次插入第一轮 Generated View，第 3000 次记录第一轮之后并插入第二轮，第 4500 次记录第二轮之后。7000 和 30000 次仍按 Reporting Checkpoint 上报，此时两轮生成图都已在训练中。

DIFIX3D+ 约每 1500 次迭代修复并写回（arXiv:2503.01774）。GeoQuery 沿用这种渐进间隔，而不是每轮跑满 Kerbl 的 30000 次。两段独立的 30k 会比这些论文密一个数量级。

## Considered Options

- 一次 30000 次训练，在 1500 与 3000 次插入两轮生成（采用）
- 每一轮生成之后各自再训练 30000 次
