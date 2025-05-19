import torch
import torch.nn as nn
import os
import glob
import re
import numpy as np

from jinja2.compiler import F


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

def gaussian_kernel(size: int, sigma: float):
    """
    生成一个二维高斯模糊核。

    Args:
        size (int): 高斯核的大小（必须是奇数）。
        sigma (float): 高斯核的标准差。

    Returns:
        torch.Tensor: 形状为 (size, size) 的高斯模糊核。
    """
    # 创建一个一维坐标向量，范围为 [-size//2, size//2]
    x = torch.arange(-size // 2 + 1, size // 2 + 1, dtype=torch.float32)
    # 生成一维高斯分布
    gauss = torch.exp(-x**2 / (2 * sigma**2))
    # 将一维高斯分布归一化
    gauss_1d = gauss / gauss.sum()
    # 利用一维高斯生成二维高斯核 (外积)
    kernel_2d = gauss_1d[:, None] @ gauss_1d[None, :]
    return kernel_2d

def cleanup_old_checkpoints(filename, max_checkpoints=3, logger=None):
    # 获取文件所在目录和文件前缀
    dir_path = os.path.dirname(filename)
    base_name = os.path.basename(filename).split('_epoch')[0]  # 提取文件名前缀（去掉 _epoch 部分）

    # 获取所有符合条件的检查点文件，并按 epoch 排序
    '''checkpoint_files = sorted(
        glob.glob(os.path.join(dir_path, f"{base_name}_epoch*.pth")),
        key=lambda x: int(x.split('_epoch')[-1].split('.')[0])  # 提取 epoch 排序
    )'''

    checkpoint_files = glob.glob(os.path.join(dir_path, f"{base_name}_epoch*.pth"))
    epoch_pattern = re.compile(r'_epoch(\d+).pth')

    keep_files = []
    for file in checkpoint_files:
        match = epoch_pattern.search(file)
        if match:
            epoch = int(match.group(1))
            if epoch >= 500 and epoch % 100 == 0:  # 保留 epoch >= 501 且能被 100 整除的文件
                keep_files.append(file)

    # 将剩下的文件按 epoch 排序
    other_files = list(set(checkpoint_files) - set(keep_files))
    other_files = sorted(other_files, key=lambda x: int(epoch_pattern.search(x).group(1)), reverse=True)  # 从大到小排序

    '''# 如果文件超过最大数量，删除多余的旧文件
    if len(checkpoint_files) > max_checkpoints:
        for old_file in checkpoint_files[:-max_checkpoints]:
            os.remove(old_file)
            if logger:
                logger.info(f"Old checkpoint removed: '{old_file}'")'''

    if len(other_files) > max_checkpoints:
        for old_file in other_files[max_checkpoints:]:
            try:
                os.remove(old_file)
                if logger:
                    logger.info(f"Old checkpoint removed: '{old_file}'")
            except Exception as e:
                if logger:
                    logger.error(f"Failed to remove '{old_file}': {str(e)}")

def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename='checkpoint.pth', max_checkpoints=3, logger=None):
    if filename.endswith('.pth'):
        filename = filename[:-4]  # 去掉 .pth
    filename = f"{filename}_epoch{epoch}.pth"

    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    checkpoint = {
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'loss': loss
    }

    if logger:
        logger.info(f"Checkpoint saved to '{filename}' (epoch {epoch}, loss {loss:.8f})")

    torch.save(checkpoint, filename)

    # clean up
    cleanup_old_checkpoints(filename, max_checkpoints, logger)

def load_checkpoint(model, optimizer, scheduler, filename, device, logger=None):
    if os.path.isfile(filename):
        checkpoint = torch.load(filename, map_location=device)

        # Handle DataParallel wrapping
        if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            # If the current model is a DataParallel model
            new_state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('module.'):
                    new_state_dict[key] = value
                else:
                    new_state_dict['module.' + key] = value
            model.load_state_dict(new_state_dict)
        else:
            # If the current model is not a DataParallel model
            new_state_dict = {}
            for key, value in checkpoint['model_state_dict'].items():
                if key.startswith('module.'):
                    new_state_dict[key[7:]] = value  # Remove 'module.' prefix
                else:
                    new_state_dict[key] = value
            model.load_state_dict(new_state_dict)

        if optimizer != None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler != None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        if logger:
            logger.info(f"Checkpoint loaded from '{filename}' (epoch {epoch}, loss {loss:.8f})")

        return model, optimizer, scheduler, epoch, loss
    else:
        if logger:
            logger.warning(f"No checkpoint found at '{filename}'")

        return model, optimizer, scheduler, 0, None
    
def Packing(images):
    input_dim = images.dim()
    if input_dim == 3:
        images = images.unsqueeze(0)
        was_3d = True
    elif input_dim == 4:
        was_3d = False
    else:
        raise ValueError(f"Input tensor must be 3D or 4D, but got {input_dim}D")

    b, c, H, W = images.shape

    assert c == 1, f"Input image must have 1 channel (Bayer pattern), but got {c}"
    assert H % 2 == 0 and W % 2 == 0, "Height and width of the image must be even."

    R = images[:, 0, 0:H:2, 0:W:2]
    G1 = images[:, 0, 0:H:2, 1:W:2]
    G2 = images[:, 0, 1:H:2, 0:W:2]
    B = images[:, 0, 1:H:2, 1:W:2]
    packed_images = torch.cat((R.unsqueeze(1), G1.unsqueeze(1), G2.unsqueeze(1), B.unsqueeze(1)), dim=1)

    if was_3d:
        packed_images = packed_images.squeeze(0) # [4, H/2, W/2]

    return packed_images


def Unpacking(packed_images):
    input_dim = packed_images.dim()
    if input_dim == 3:
        packed_images = packed_images.unsqueeze(0)
        was_3d = True
    elif input_dim == 4:
        was_3d = False
    else:
        raise ValueError(f"Input tensor must be 3D or 4D, but got {input_dim}D")

    b, c, h, w = packed_images.shape
    assert c == 4, f"The number of channels in packed images must be 4, but got {c}."

    R = packed_images[:, 0, :, :]
    G1 = packed_images[:, 1, :, :]
    G2 = packed_images[:, 2, :, :]
    B = packed_images[:, 3, :, :]
    H_out, W_out = h * 2, w * 2
    unpacked_images = torch.zeros((b, 1, H_out, W_out), device=packed_images.device, dtype=packed_images.dtype)

    unpacked_images[:, 0, 0:H_out:2, 0:W_out:2] = R
    unpacked_images[:, 0, 0:H_out:2, 1:W_out:2] = G1
    unpacked_images[:, 0, 1:H_out:2, 0:W_out:2] = G2
    unpacked_images[:, 0, 1:H_out:2, 1:W_out:2] = B

    if was_3d:
        unpacked_images = unpacked_images.squeeze(0)

    return unpacked_images

# [b, c, h, w] -> [b*n, c, p, p], p=patch_size, n=num_patches
def PatchFunc(x, patch_size):
    batch, channel, height, width = x.shape

    nh = height // patch_size
    nw = width // patch_size

    x_patches = x.view(batch, channel, nh, patch_size, nw, patch_size)  # [b, c, nh, p, nw, p]
    x_patches = x_patches.permute(0, 1, 2, 4, 3, 5).contiguous().view(batch, channel, nh * nw, patch_size, patch_size)  # [b, c, nh*nw, p, p]
    x_patches = x_patches.permute(0, 2, 1, 3, 4).contiguous().view(batch * nh * nw, channel, patch_size, patch_size)  # [b*np, c, p, p]

    return x_patches

# [b*n, c, p, p] -> [b, c, h, w], p=patch_size, n=num_patches
def UnpatchFunc(x_patched, x_size, patch_size):
    batch, channel, height, width = x_size

    nh = height // patch_size
    nw = width // patch_size

    x_unpatch = x_patched.view(batch, nh * nw, channel, patch_size, patch_size).permute(0, 2, 1, 3, 4).contiguous()
    x_unpatch = x_unpatch.view(batch, channel, nh, nw, patch_size, patch_size).permute(0, 1, 2, 4, 3, 5).contiguous()
    x_unpatch = x_unpatch.view(batch, channel, height, width)

    return x_unpatch

# [b, c, h, w] -> [b, h*w, c]
def PatchEmbed(x):
    b, c, h, w = x.shape

    x = x.flatten(2).transpose(1, 2)

    return x


# [b, h*w, c] -> [b, c, h, w]
def PatchUnembed(x, x_size, embed_dim):
    x = x.transpose(1, 2).contiguous().view(x.shape[0], embed_dim, x_size[0], x_size[1])

    return x

def window_partition(x, window_size):
    """
        [b,c,h,w] -> [b*num_windows, c, window_size, window_size]
    """
    b, c, h, w = x.shape
    x = x.view(b, c, h // window_size, window_size, w // window_size, window_size)
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, c, window_size, window_size)

    return windows

def window_reverse(px, window_size, x_size):
    """
         [b*num_windows, c, window_size, window_size] -> [b,c,h,w]
    """
    h, w = x_size
    c = px.shape[1]
    nw_h = h // window_size # num_windows of h
    nw_w = w // window_size # num_windows of w
    num_windows = nw_h * nw_w
    b =px.shape[0] // num_windows

    y = px.view(b, nw_h, nw_w, c, window_size, window_size)
    y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, c, h, w)

    return y


def RelativeMul(img1, img2):
    """
    使用unfold实现的更高效版本
    Args:
        img1: [B, C, H, W]
        img2: [B, C, fH, fW] (H % fH == 0, W % fW == 0)
    """
    B, C, H, W = img1.shape
    fH, fW = img2.shape[2], img2.shape[3]
    wh, ww = H // fH, W // fW

    # 使用unfold划分窗口 [B, C, fH, fW, wh, ww]
    windows = img1.unfold(2, wh, wh).unfold(3, ww, ww)  # [B, C, fH, fW, wh, ww]

    # 应用权重 [B, C, fH, fW, 1, 1]
    weighted = windows * img2.view(B, C, fH, fW, 1, 1)

    # 折叠回原尺寸 [B, C, H, W]
    output = weighted.permute(0, 1, 2, 4, 3, 5).contiguous()  # [B, C, fH, wh, fW, ww]
    return output.view(B, C, H, W)


def image_to_blocks(x: torch.Tensor, k: int) -> torch.Tensor:
    """
    将图像分块为不重叠的k×k小块
    Args:
        x: 输入张量 [B, C, H, W]，要求H和W必须能被k整除
        k: 分块大小
    Returns:
        分块后的张量 [B, (H//k)*(W//k), C, k, k]
    """
    B, C, H, W = x.shape
    assert H % k == 0 and W % k == 0, "输入尺寸必须能被k整除"

    # 展开为块 [B, C, num_blocks_h, k, num_blocks_w, k]
    blocks = x.unfold(2, k, k).unfold(3, k, k)

    # 合并块维度 [B, C, num_blocks_h, num_blocks_w, k, k]
    blocks = blocks.contiguous().view(B, C, -1, k, k)  # [B, C, total_blocks, k, k]

    # 调整维度顺序 [B, total_blocks, C, k, k]
    blocks = blocks.permute(0, 2, 1, 3, 4)
    return blocks


def blocks_to_image(blocks: torch.Tensor, original_shape: tuple) -> torch.Tensor:
    """
    将分块后的张量还原为原始图像
    Args:
        blocks: 分块张量 [B, num_blocks, C, k, k]
        original_shape: 原始图像形状 (B, C, H, W)
    Returns:
        重建后的图像 [B, C, H, W]
    """
    B, C, H, W = original_shape
    k = blocks.shape[-1]
    num_blocks = blocks.shape[1]

    # 调整维度顺序 [B, C, num_blocks, k, k]
    blocks = blocks.permute(0, 2, 1, 3, 4)

    # 计算原始块排列方式
    num_blocks_h = H // k
    num_blocks_w = W // k

    # 重塑为可折叠的形状 [B, C, num_blocks_h, num_blocks_w, k, k]
    blocks = blocks.view(B, C, num_blocks_h, num_blocks_w, k, k)

    # 重建图像
    x = blocks.permute(0, 1, 2, 4, 3, 5).contiguous()  # [B, C, num_blocks_h, k, num_blocks_w, k]
    x = x.view(B, C, num_blocks_h * k, num_blocks_w * k)
    return x

'''class SharedDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size, overlap_ratio=0):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.overlap_ratio = overlap_ratio

        self.overlap_kernel_size = int(kernel_size * (1 + overlap_ratio))
        self.shared_kernel = nn.Parameter(torch.randn(1, self.overlap_kernel_size, self.overlap_kernel_size))

    def forward(self, x):
        kernel = self.shared_kernel.repeat(self.in_channels, 1, 1, 1)  # [C, 1, K, K]
        return nn.functional.conv2d(x,
                                    kernel,
                                    groups=self.in_channels,
                                    stride=self.kernel_size,
                                    padding=(self.overlap_kernel_size - self.kernel_size) // 2)'''


class SharedDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.shared_kernel = nn.Parameter(torch.randn(1, self.kernel_size, self.kernel_size))

    def forward(self, x):
        kernel = self.shared_kernel.repeat(self.in_channels, 1, 1, 1)  # [C, 1, K, K]
        return nn.functional.conv2d(x,
                                    kernel,
                                    groups=self.in_channels,
                                    stride=self.kernel_size,)


# DCT
def get_dctMatrix(m, n):
    N = n
    C_temp = np.zeros([m, n])
    C_temp[0, :] = 1 * np.sqrt(1 / N)

    for i in range(1, m):
        for j in range(n):
            C_temp[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N)) * np.sqrt(2 / N)
    return torch.tensor(C_temp, dtype=torch.float)

def dct1d(feature, dctMat):
    feature = feature @ dctMat.T # dctMat @ feature  #
    return feature.contiguous()  # torch.tensor(x, device=feature.device)

def idct1d(feature, dctMat):
    feature = feature @ dctMat # .T # dctMat.T @ feature  # .T
    return feature.contiguous()  # torch.tensor(x, device=feature.device)

def dct2dx(feature, dctMat1, dctMat2):
    # print(dctMat.shape, feature.shape)
    feature = dct1d(feature, dctMat1) # dctMat1 @ feature
    # print(dctMat.shape, feature.shape)
    feature = dct1d(feature.transpose(-1, -2), dctMat2) # feature @ dctMat2.T
    return feature.transpose(-1, -2).contiguous()  # torch.tensor(x, device=feature.device)

def idct2dx(feature, dctMat1, dctMat2):
    feature = idct1d(feature, dctMat1)  # dctMat.T @ feature # .transpose(-1, -2)
    feature = idct1d(feature.transpose(-1, -2), dctMat2)
    return feature.transpose(-1, -2).contiguous()  # torch.tensor(x, device=feature.device)

class DCT2x(nn.Module):
    def __init__(self, norm='ortho'):
        super(DCT2x, self).__init__()
        self.dctMatH = None
        self.dctMatW = None
        self.norm = norm

    def check_dct_matrix(self, h, w):
        if self.dctMatH is None or self.dctMatW is None:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1] and w != self.dctMatW.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            # self.dctMatH = self.dctMatH.to(x.device)
        elif w != self.dctMatW.shape[-1]:
            self.dctMatW = get_dctMatrix(w, w)

    def forward(self, x):
        h, w = x.shape[-2:]
        self.check_dct_matrix(h, w)
        dctMatH = self.dctMatH.to(x.device)
        dctMatW = self.dctMatW.to(x.device)
        # print(x.shape, self.dctMatH.shape, self.dctMatW.shape)
        x = dct2dx(x, dctMatW, dctMatH)

        return x
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        flops += C * H * W * np.log2(H * W)
        return flops

class IDCT2x(nn.Module):
    def __init__(self, norm='ortho'):
        super(IDCT2x, self).__init__()
        self.dctMatH = None
        self.dctMatW = None
        self.norm = norm

    def check_dct_matrix(self, h, w):
        if self.dctMatH is None or self.dctMatW is None:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1] and w != self.dctMatW.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            self.dctMatW = get_dctMatrix(w, w)
        elif h != self.dctMatH.shape[-1]:
            self.dctMatH = get_dctMatrix(h, h)
            # self.dctMatH = self.dctMatH.to(x.device)
        elif w != self.dctMatW.shape[-1]:
            self.dctMatW = get_dctMatrix(w, w)

    def forward(self, x):
        h, w = x.shape[-2:]
        self.check_dct_matrix(h, w)
        dctMatH = self.dctMatH.to(x.device)
        dctMatW = self.dctMatW.to(x.device)
        x = idct2dx(x, dctMatW, dctMatH)

        return x
    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        flops += C * H * W * np.log2(H * W)
        return flops











