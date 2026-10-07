"""Rendering metrics. LPIPS uses the VGG network, matching official 3DGS."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean PSNR over a batch of images in ``[0, 1]`` with shape ``N,C,H,W``."""
    mse = torch.mean((prediction - target) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / mse.clamp_min(1e-10))


def ssim(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Channel-averaged SSIM with an 11x11 Gaussian window, one value per image."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("expected matching NCHW tensors")
    window = _gaussian_window(prediction.device, prediction.dtype)
    channels = prediction.shape[1]
    weight = window.expand(channels, 1, 11, 11)
    mu_x = F.conv2d(prediction, weight, padding=5, groups=channels)
    mu_y = F.conv2d(target, weight, padding=5, groups=channels)
    sigma_x = F.conv2d(prediction * prediction, weight, padding=5, groups=channels) - mu_x ** 2
    sigma_y = F.conv2d(target * target, weight, padding=5, groups=channels) - mu_y ** 2
    sigma_xy = F.conv2d(prediction * target, weight, padding=5, groups=channels) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    score = numerator / denominator
    return score.mean(dim=(1, 2, 3))


def _gaussian_window(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(11, device=device, dtype=dtype) - 5
    kernel = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
    kernel = kernel / kernel.sum()
    window = kernel[:, None] * kernel[None, :]
    return window.view(1, 1, 11, 11)


_lpips_model = None


def lpips_vgg(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """VGG LPIPS, one value per image. Images are ``N,3,H,W`` in ``[0, 1]``."""
    global _lpips_model
    import lpips

    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net="vgg").to(prediction.device)
        _lpips_model.eval()
    scaled_prediction = prediction * 2 - 1
    scaled_target = target * 2 - 1
    values = []
    with torch.no_grad():
        for index in range(prediction.shape[0]):
            value = _lpips_model(scaled_prediction[index : index + 1], scaled_target[index : index + 1])
            values.append(value.reshape(1))
    return torch.cat(values)
