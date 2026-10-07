# Score rendering only on midpoint captured frames

PSNR、SSIM、LPIPS 只在 Test View 上计算。Test View 是 Captured View 里每个抽帧间隔的中间帧：Waymo-405841 为下标除以 10 余 5，Dl3dv-2 与 Re10k-1 为下标除以 30 余 15。这些帧不进入训练、位姿初始化、GeoQuery 参考，也不作为 Generated View 写回。

Part 1 和 Part 2 的全视角设置改为用 Dense View 训练，也就是全部 Captured View 去掉 Test View。这样稠密、稀疏和闭环的渲染指标打在同一套图上。Generated View 只能进训练。

不用「稀疏输入以外的每一帧」做测试，因为那会让测试集比训练集成倍地密，和全视角设置的测试集也不再是同一批图。

## Considered Options

- 每个间隔的中间帧作为 Test View（采用）
- 稀疏输入以外的全部帧作为测试
- 全视角设置把包括测试帧在内的所有帧都拿去训练
