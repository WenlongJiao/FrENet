import rawpy
import torch
import numpy as np
import torch.nn.functional as F

def pad_image(image):
    # 如果输入是批量数据，则遍历每个样本
    if image.dim() == 4:  # [B, C, H, W]
        b, c, h, w = image.shape
        padded_images = []
        for i in range(b):
            padded_image = _pad_single_image(image[i])  # 单张图像填充
            padded_images.append(padded_image)
        return torch.stack(padded_images, dim=0)
    else:  # 单个图像 [C, H, W]
        return _pad_single_image(image)


def _pad_single_image(image):
    _, h, w = image.shape

    # 计算需要填充的高度和宽度
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    # 按 (left, right, top, bottom) 顺序进行填充
    padding = (0, pad_w, 0, pad_h)

    # 使用 `reflect` 模式进行填充
    padded_image = F.pad(image, padding, mode='reflect')

    return padded_image

def upscale(tensor, scale=4, mode='bicubic'):
    tensor = F.interpolate(tensor, scale_factor=scale, mode=mode)
    return tensor