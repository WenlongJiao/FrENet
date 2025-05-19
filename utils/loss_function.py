import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from utils.model_util import Packing
from utils.calculate_util import calculate_ssim

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are 'none', 'mean' and 'sum'.

    Returns:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    else:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean'):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights. Default: None.
        reduction (str): Same as built-in losses of PyTorch. Options are
            'none', 'mean' and 'sum'. Default: 'mean'.

    Returns:
        Tensor: Loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        assert weight.dim() == loss.dim()
        assert weight.size(1) == 1 or weight.size(1) == loss.size(1)
        loss = loss * weight

    # if weight is not specified or reduction is sum, just reduce the loss
    if weight is None or reduction == 'sum':
        loss = reduce_loss(loss, reduction)
    # if reduction is mean, then compute mean over weight region
    elif reduction == 'mean':
        if weight.size(1) > 1:
            weight = weight.sum()
        else:
            weight = weight.sum() * loss.size(1)
        loss = loss.sum() / weight

    return loss


def weighted_loss(loss_func):
    """Create a weighted version of a given loss function.

    To use this decorator, the loss function must have the signature like
    `loss_func(pred, target, **kwargs)`. The function only needs to compute
    element-wise loss without any reduction. This decorator will add weight
    and reduction arguments to the function. The decorated function will have
    the signature like `loss_func(pred, target, weight=None, reduction='mean',
    **kwargs)`.

    :Example:

    >>> import torch
    >>> @weighted_loss
    >>> def l1_loss(pred, target):
    >>>     return (pred - target).abs()

    >>> pred = torch.Tensor([0, 2, 3])
    >>> target = torch.Tensor([1, 1, 1])
    >>> weight = torch.Tensor([1, 0, 1])

    >>> l1_loss(pred, target)
    tensor(1.3333)
    >>> l1_loss(pred, target, weight)
    tensor(1.5000)
    >>> l1_loss(pred, target, reduction='none')
    tensor([1., 1., 2.])
    >>> l1_loss(pred, target, weight, reduction='sum')
    tensor(3.)
    """

    @functools.wraps(loss_func)
    def wrapper(pred, target, weight=None, reduction='mean', **kwargs):
        # get element-wise loss
        loss = loss_func(pred, target, **kwargs)
        loss = weight_reduce_loss(loss, weight, reduction)
        return loss

    return wrapper

@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        if isinstance(pred, list):
            loss = 0.
            for predi in pred:
                loss += l1_loss(
                predi, target, weight, reduction=self.reduction)
            return self.loss_weight * loss
        else:
            return self.loss_weight * l1_loss(
                pred, target, weight, reduction=self.reduction)

class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x.to('cuda:0') - y.to('cuda:0')
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss

class EdgeLoss(nn.Module):
    def __init__(self):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(4,1,1,1)
        if torch.cuda.is_available():
            self.kernel = self.kernel.to('cuda:0')
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x.to('cuda:0')), self.laplacian_kernel(y.to('cuda:0')))
        return loss

class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss

class GradientMSELoss(nn.Module):
    def __init__(self, w1=0.5, w2=0.25, w3=0.25):
        """
        alpha: 控制MSE和梯度损失之间的权重。
        alpha = 0.5 表示MSE损失和梯度损失权重相同。
        """
        super(GradientMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3

    def gradient(self, x):
        """
        计算图像的水平和垂直梯度。
        Sobel滤波器用来计算梯度。
        """
        c = x.shape[1]
        
        
        sobel_x = torch.tensor([[-1, 0, 1], 
                                [-2, 0, 2], 
                                [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).to(x.device)
        
        sobel_y = torch.tensor([[-1, -2, -1], 
                                [0, 0, 0], 
                                [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).to(x.device)
        
        sobel_x = sobel_x.repeat(1, c, 1, 1)
        sobel_y = sobel_y.repeat(1, c, 1, 1)

        grad_x = F.conv2d(x, sobel_x, padding=1)
        grad_y = F.conv2d(x, sobel_y, padding=1)
        
        grad = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)  # 防止梯度为0的情况
        return grad

    def forward(self, pred, target):
        """
        计算MSE损失和梯度损失，并将两者结合。
        """
        # MSE损失
        mse_loss = self.mse_loss(pred, target)

        # 计算预测图像和目标图像的梯度
        pred_grad = self.gradient(pred)
        target_grad = self.gradient(target)

        # 梯度损失
        grad_loss = self.mse_loss(pred_grad, target_grad)
        
        # packed梯度损失
        packed_pred = Packing(pred)
        packed_target = Packing(target)
        packed_pred_grad = self.gradient(packed_pred)
        packed_target_grad = self.gradient(packed_target)
        
        packed_grad_loss = self.mse_loss(packed_pred_grad, packed_target_grad)

        # 总损失：结合MSE损失和梯度损失
        total_loss = self.w1 * mse_loss + self.w2 * grad_loss + self.w3 * packed_grad_loss

        return total_loss


class SSIMMSELoss(nn.Module):
    def __init__(self, w=0.8) -> None:
        super(SSIMMSELoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.w = w
        
    def calculate_ssim_per_image(self, pred, target):
        """
        计算每个图像的 SSIM
        """
        pred = pred.squeeze(0).cpu().detach()
        target = target.squeeze(0).cpu().detach()
        return 1 - calculate_ssim(pred, target, data_range=1.0)  # 1 - SSIM 是损失
    
    def forward(self, pred, target):
        B, C, H, W = pred.shape
        
        # MSE损失
        mse_loss = self.mse_loss(pred, target)
        
        # SSIM 损失
        ssim_loss = 0
        for b in range(B):
            for c in range(C):
                ssim_loss += self.calculate_ssim_per_image(pred[b, c, :, :], target[b, c, :, :])
        
        ssim_loss /= (B * C)  # 计算 SSIM 损失的平均值
        
        # 总损失
        total_loss = self.w * mse_loss + (1 - self.w) * ssim_loss
        
        return total_loss


def FReLU(img):
    x = torch.fft.rfft2(img)
    x_real = torch.relu(x.real)
    x_imag = torch.relu(x.imag)
    x = torch.complex(x_real, x_imag)
    x = torch.fft.irfft2(x) - img / 2.
    return x
class FRLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=0.01, reduction='mean'):
        super(FRLoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target):
        diff = FReLU(pred) - FReLU(target)
        loss = torch.mean(torch.abs(diff))
        # print(loss, self.l1_loss(pred, target))
        return self.loss_weight * loss + self.l1_loss(pred, target)

class FreqLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss of fft.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FreqLoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.l1_loss = L1Loss(loss_weight, reduction)

    def forward(self, pred, target):
        diff = torch.fft.rfft2(pred) - torch.fft.rfft2(target)
        loss = torch.mean(torch.abs(diff))
        # print(loss)
        return self.loss_weight * (loss * 0.01 + self.l1_loss(pred, target))

class CrossEntropyLoss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(reduction=reduction)
        # self.loss = nn.MSELoss(reduction=reduction)

    def forward(self, pred, gt, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        # print(pred.shape, gt.shape)

        return self.loss_weight * self.loss(pred, gt)