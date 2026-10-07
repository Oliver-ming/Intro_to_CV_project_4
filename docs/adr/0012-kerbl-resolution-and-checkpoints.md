# Match Kerbl's resolution and report at 7k and 30k

标准 3DGS 跟 Kerbl 等人 2023 的设置。所有方法共用 Input Resolution：宽度超过 1600 时把宽度缩到 1600 并保持宽高比，否则保持原尺寸。Reporting Checkpoint 是 7000 次和 30000 次迭代上的 PSNR、SSIM、LPIPS 和耗时。LPIPS 用 VGG，与官方 `metrics.py` 一致。三种初始化使用同一个 30000 次预算。i3dgs 保持自己的优化日程，只共用输入像素。

不把长边一律缩成 1024，也不用 AlexNet LPIPS。S3PO-GS 的公开配置是 Waymo 1920×1280、DL3DV 256×256，因此这套数字不与该表逐项对应。

## Considered Options

- 宽度上限 1600，7k/30k，LPIPS-VGG（采用）
- 长边 1024，LPIPS-AlexNet
- 按 S3PO-GS 的原分辨率或 256×256
