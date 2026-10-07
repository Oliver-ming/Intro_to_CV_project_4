# Give generated views the same photometric supervision as sparse views

Generated View 写回后与 Sparse View 使用同一套光度损失，监督强度相同。必做对比是写回之前、第一轮之后、第二轮之后的重建，不是置信度加权。课程里的一致性融合仍是进阶项。若第二轮变差，把变差写成失败案例，而不是事先把生成图降权。

降权或置信度掩码会把「生成视角有没有用」和「权重怎么调」绑在同一个数字里。同等监督让这两件事分开。

## Considered Options

- 与 Sparse View 同等的光度监督（采用）
- 固定降权
- 按 BRPO 的置信度掩码加权
