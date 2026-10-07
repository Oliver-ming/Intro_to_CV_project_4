# Generative Sparse-View 3D Reconstruction

AIAA 3201 Project 4 的领域语言：在观测不足时，用三维重建和生成观测补全同一场景。

## Language

**Closed-Loop Generative Reconstruction**:
把生成或修复得到的、拍摄时不存在的视角，当作同一场景的新观测写回重建，并再次更新场景。课程要求至少两轮，且必须完成。
_Avoid_: Part 3 进阶项, 一次性新视角合成, 加分生成

**Captured View**:
数据集里实际拍到的图像及其对应相机。
_Avoid_: 输入帧, 原图

**Generated View**:
由生成模型新合成或修复出来、再写回重建的视角。
_Avoid_: 伪视角, novel view, 补图
