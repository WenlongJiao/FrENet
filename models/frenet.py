import torch
import torch.nn as nn
import torch.nn.functional
import time
import functools
import math

from torch.nn import init

from utils.model_util import *


class FrENet(nn.Module):
    def __init__(self,
                 in_channel=4,
                 width=64,
                 middle_blk_num=4,
                 enc_blk_nums=[2, 2, 4],
                 dec_blk_nums=[4, 2, 2],
                 train_size=None,
                 img_size=64,
                 grid_overlap_size=16, ) -> None:
        super(FrENet, self).__init__()

        self.train_size = train_size
        self.grid_overlap_size = (grid_overlap_size, grid_overlap_size)
        self.grid_kernel_size = [self.train_size, self.train_size]
        self.grid = True

        self.intro = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=in_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        image_size = img_size

        for num in enc_blk_nums:
            fft_kernel = image_size//2 if image_size <= 8 else image_size//8
            self.encoders.append(
                nn.ModuleList(
                    [FrEBlock(in_channel=chan, fft_kernel=fft_kernel, img_size=image_size) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(in_channels=chan, out_channels=2 * chan, kernel_size=2, stride=2)
            )
            chan = 2 * chan
            image_size = image_size // 2

        # Middle blocks
        fft_kernel = image_size // 2 if image_size <= 8 else image_size // 8
        self.middle_blks = nn.ModuleList(
            [FrEBlock(in_channel=chan, fft_kernel=fft_kernel, img_size=image_size) for _ in range(middle_blk_num)]
        )

        # Decoder stages
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            image_size = image_size * 2
            fft_kernel = image_size // 2 if image_size <= 8 else image_size // 8
            self.decoders.append(
                nn.ModuleList(
                    [FrEBlock(in_channel=chan, fft_kernel=fft_kernel, img_size=image_size) for _ in range(num)]
                )
            )

    def forward(self, x):
        BATCH, CHANNEL, HEIGHT, WIDTH = x.shape
        inp = x

        if self.grid and self.train_size:
            x = self.grids(x)
        x = self.intro(x)

        encs = []
        x_fft_skips = []

        current_x = x
        for stage_idx, (encoder_stage, down_layer) in enumerate(zip(self.encoders, self.downs)):
            stage_fft_skip = None
            for freblock in encoder_stage:
                current_x, stage_fft_skip = freblock(current_x, None, is_encoder=True)

            encs.append(current_x)
            x_fft_skips.append(stage_fft_skip)

            current_x = down_layer(current_x)

        middle_block_skip = 0
        for freblock in self.middle_blks:
            current_x = freblock(current_x, middle_block_skip, is_encoder=False)

        for stage_idx, (decoder_stage, up_layer, enc_skip, stage_input_fft_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1],
                                                                                                  x_fft_skips[::-1])):
            current_x = up_layer(current_x)
            current_x = current_x + enc_skip

            current_stage_fft_skip = stage_input_fft_skip
            for block_idx, freblock in enumerate(decoder_stage):
                current_x = freblock(current_x, current_stage_fft_skip, is_encoder=False,)

        x = self.ending(current_x)

        if self.grid and self.train_size:
            x = self.grids_inverse(x)

        x = x[:, :, :HEIGHT, :WIDTH].contiguous() + inp

        return x

    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        assert b == 1
        k1, k2 = self.grid_kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        grid_overlap_size = self.grid_overlap_size  # (64, 64)

        stride = (k1 - grid_overlap_size[0], k2 - grid_overlap_size[1])
        self.stride = stride
        num_row = (h - grid_overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - grid_overlap_size[1] - 1) // stride[1] + 1
        self.nr = num_row
        self.nc = num_col

        # import math
        step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                # if not self.ek1:
                #     # print(step_i, i, k1, h)
                #     self.ek1 = i + k1 - h # - self.grid_overlap_size[0]
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    # if not self.ek2:
                    #     self.ek2 = j + k2 - w # + self.grid_overlap_size[1]
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def get_overlap_matrix(self, h, w):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        self.h = h
        self.w = w
        self.ek1 = self.nr * self.stride[0] + self.grid_overlap_size[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.grid_overlap_size[1] * 2 - w
        # self.ek1, self.ek2 = 48, 224
        # print(self.ek1, self.ek2, self.nr)
        # print(self.grid_overlap_size)
        # self.grid_overlap_size = [8, 8]
        # self.grid_overlap_size = [self.grid_overlap_size[0] * 2, self.grid_overlap_size[1] * 2]
        self.fuse_matrix_w1 = torch.linspace(1., 0., self.grid_overlap_size[1]).view(1, 1, self.grid_overlap_size[1])
        self.fuse_matrix_w2 = torch.linspace(0., 1., self.grid_overlap_size[1]).view(1, 1, self.grid_overlap_size[1])
        self.fuse_matrix_h1 = torch.linspace(1., 0., self.grid_overlap_size[0]).view(1, self.grid_overlap_size[0], 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., self.grid_overlap_size[0]).view(1, self.grid_overlap_size[0], 1)
        self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1)
        self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1)

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.grid_kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            if i != 0 and i + k1 != h:
                outs[cnt, :, :self.grid_overlap_size[0], :] *= self.fuse_matrix_h2.to(outs.device)
            if i + k1 * 2 - self.ek1 < h:
                # print(outs[cnt, :,  i + k1 - self.grid_overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
                outs[cnt, :, -self.grid_overlap_size[0]:, :] *= self.fuse_matrix_h1.to(outs.device)
            if i + k1 == h:
                outs[cnt, :, :self.ek1, :] *= self.fuse_matrix_eh2.to(outs.device)
            if i + k1 * 2 - self.ek1 == h:
                outs[cnt, :, -self.ek1:, :] *= self.fuse_matrix_eh1.to(outs.device)

            if j != 0 and j + k2 != w:
                outs[cnt, :, :, :self.grid_overlap_size[1]] *= self.fuse_matrix_w2.to(outs.device)
            if j + k2 * 2 - self.ek2 < w:
                # print(j, j + k2 - self.grid_overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, -self.grid_overlap_size[1]:] *= self.fuse_matrix_w1.to(outs.device)
            if j + k2 == w:
                # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                outs[cnt, :, :, :self.ek2] *= self.fuse_matrix_ew2.to(outs.device)
            if j + k2 * 2 - self.ek2 == w:
                # print('j + k2*2 - self.ek2 == w: ')
                outs[cnt, :, :, -self.ek2:] *= self.fuse_matrix_ew1.to(outs.device)
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds  # / count_mt

# -----------------------------------------------------------------------

# for unet
class FrEBlock(nn.Module):
    def __init__(self, in_channel, fft_kernel=4, img_size=64, c_expand=2, ffn_expand=2):
        super(FrEBlock, self).__init__()

        self.c_expand = c_expand
        self.ffn_expand = ffn_expand
        channel_expand = self.c_expand * in_channel
        channel_ffn = self.ffn_expand * in_channel

        self.fft_kernel = fft_kernel

        # fft conv
        self.conv_fft1 = nn.Conv2d(in_channels=in_channel*2, out_channels=channel_expand*2, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_fft2 = nn.Conv2d(in_channels=channel_expand*2, out_channels=channel_expand*2, kernel_size=3, stride=1, padding=1, groups=channel_expand*2, bias=True)
        self.conv_fft3 = nn.Conv2d(in_channels=channel_expand, out_channels=in_channel*2, kernel_size=1, stride=1, padding=0, bias=True)

        self.ffn = FeedForward(dim=in_channel, ffn_expansion_factor=2.66, bias=True)

        # oca
        self.afpm_fft = AFPM(in_channel=channel_expand, kernel_size=fft_kernel, H=img_size, W=img_size)
        self.sca = SCA(in_channel=channel_expand)

        self.norm2 = LayerNorm2d(in_channel)
        self.norm_fft1 = LayerNorm2d(in_channel*2)

        self.beta = nn.Parameter(torch.zeros((1, in_channel, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channel, 1, 1)), requires_grad=True)

        self.gelu = nn.GELU()
        self.sg = SimpleGate()

    def forward(self, x, x_fft_skip=None, is_encoder=False):
        batch, channel, height, width = x.shape
        x_size = [batch, channel, height, width]
        inp = x

        # FFT
        x_fft = torch.fft.fft2(x)
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))
        if is_encoder==False:
            x_fft = x_fft + x_fft_skip
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_fft = self.norm_fft1(x_fft)
        x_fft = self.conv_fft1(x_fft)
        x_fft = self.conv_fft2(x_fft)
        x_fft = self.sg(x_fft)
        local_fea = self.afpm_fft(x_fft)
        global_fea = self.sca(x_fft)
        x_fft = local_fea + global_fea
        x_fft_real, x_fft_imag = self.conv_fft3(x_fft).chunk(2, dim=1)
        x_fft = torch.complex(x_fft_real, x_fft_imag)
        x_fft = torch.fft.ifftshift(x_fft, dim=(-2, -1))
        x_fft = torch.fft.ifft2(x_fft).real

        x_out = x_fft
        x_out = x_out * self.beta + inp

        inp = x_out

        # LayerNorm
        x_out_norm = self.norm2(x_out)
        x_ffn = self.ffn(x_out_norm)
        x_out = x_ffn
        x_out = x_out * self.gamma + inp

        if is_encoder:
            return x_out, x_fft_skip
        else:
            return x_out


#--------------------------------------------------------------------------

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

class SCA(nn.Module):
    def __init__(self, in_channel):
        super(SCA, self).__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
        )

    def forward(self, x):
        y = self.sca(x)
        return x * y


class AFPM(nn.Module):
    def __init__(self, in_channel, kernel_size, H=64, W=64, mlp_hidden_dim=None):
        super().__init__()
        self.in_channel = in_channel
        self.K = kernel_size
        self.H, self.W = H, W

        if H % self.K != 0 or W % self.K != 0:
            raise ValueError(f"输入尺寸 H={H}, W={W} 必须能被 kernel_size={self.K} 整除。")

        self.num_h = H // self.K
        self.num_w = W // self.K
        self.L = self.num_h * self.num_w

        # Positional Encoding
        center_y = H / 2.0
        center_x = W / 2.0
        max_dist = math.sqrt(center_y**2 + center_x**2)
        if max_dist == 0: max_dist = 1.0 # Avoid division by zero if H=W=1

        patch_positions = torch.zeros(self.L)
        for l in range(self.L):
            i = l // self.num_w
            j = l % self.num_w
            patch_center_y = i * self.K + self.K / 2.0
            patch_center_x = j * self.K + self.K / 2.0
            dist = math.sqrt((patch_center_y - center_y)**2 + (patch_center_x - center_x)**2)
            patch_positions[l] = dist / max_dist # Normalized distance

        self.register_buffer('patch_pos_dist', patch_positions)

        if mlp_hidden_dim is None:
            mlp_hidden_dim = in_channel

        self.kernel_generator_mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, self.K * self.K)
        )

        self.pos_bias_mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, 1)  # Output: additive embedding
        )

        self.global_conv = nn.Conv2d(
                in_channels=in_channel,
                out_channels=in_channel,
                kernel_size=1,
                groups=1
            )


    def forward(self, x):
        B, C, H, W = x.shape
        L = self.L
        K = self.K

        unfolded_x = torch.nn.functional.unfold(x, K, stride=K) # [B, C*K*K, L]
        unfolded_x = unfolded_x.view(B, C, K * K, L) # [B, C, K*K, L]
        unfolded_x = unfolded_x.permute(0, 3, 1, 2).contiguous() # [B, L, C, K*K]
        unfolded_x = unfolded_x.view(B*L, C, K, K) # [B*L, C, K, K]

        pos_dist_input = self.patch_pos_dist.unsqueeze(-1)  # [L, 1]
        pos_kernels = self.kernel_generator_mlp(pos_dist_input)  # [L, K*K]
        pos_bias = self.pos_bias_mlp(pos_dist_input)  # [L, 1]

        unfolded_x = unfolded_x.view(B, L, C, K*K) # [B, L, C, K^2]
        unfolded_x = unfolded_x.permute(0, 2, 3, 1).contiguous() # [B, C, K^2, L]

        features = torch.einsum('bckl,lk->bcl', unfolded_x, pos_kernels)  # [B, C, L]
        features = features + pos_bias.view(1, 1, L)

        features = features.permute(0, 2, 1).contiguous()  # [B, L, C]
        features = features.view(B * L, C, 1, 1)  # [B*L, C, 1, 1]
        features = self.global_conv(features)  # [B*L, C, 1, 1]
        features = features.view(B, L, C).permute(0, 2, 1)  # [B, C, L]
        features = features.unsqueeze(2)  # [B, C, 1, L]

        modulated = unfolded_x * features # [B, C, K^2, L]

        modulated = modulated.permute(0, 3, 1, 2).contiguous() # [B, L, C, K^2]
        modulated = modulated.view(B * L, C, K, K) # [B*L, C, K, K]

        output = modulated.view(B, L, C*K*K) # [B, L, C*K^2]
        output = output.permute(0, 2, 1).contiguous() # [B, C*K^2, L]
        output = torch.nn.functional.fold(
            output,
            (H, W), kernel_size=K, stride=K
        ) # [B, C, H, W]

        return output


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, ffn='ffn', window_size=None):
        super(FeedForward, self).__init__()

        self.ffn_expansion_factor = ffn_expansion_factor

        self.ffn = ffn
        if self.ffn_expansion_factor == 0:
            hidden_features = dim
            self.project_in = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

            self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                    groups=dim, bias=bias)
        else:
            hidden_features = int(dim*ffn_expansion_factor)
            self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
            self.act = nn.GELU()
            self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.dim = dim
        self.hidden_dim = hidden_features
    def forward(self, inp):
        x = self.project_in(inp)
        if self.ffn_expansion_factor == 0:
            x = self.act(self.dwconv(x))
        else:
            x1, x2 = self.dwconv(x).chunk(2, dim=1)
            x = self.act(x1) * x2
        x = self.project_out(x)
        return x

