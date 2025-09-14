import math
import selective_scan_cuda
import torch

import torch.nn.functional as F

from einops import repeat
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SFFM(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SFFM, self).__init__()

        inter_channels = max(channels // reduction, 4)

        self.channel_cascade = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=2 * channels, out_channels=inter_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inter_channels, out_channels=2 * channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.spatial_cascade = nn.Sequential(
            nn.Conv2d(in_channels=2 * channels, out_channels=inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inter_channels, out_channels=1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )

        self.channel_parallel = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=2 * channels, out_channels=inter_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inter_channels, out_channels=1, kernel_size=1, bias=False)
        )
        self.spatial_parallel = nn.Sequential(
            nn.Conv2d(in_channels=2 * channels, out_channels=inter_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=inter_channels, out_channels=1, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, ir, vi):
        fuse = torch.cat([ir, vi], dim=1)

        channel_w = self.channel_cascade(fuse) * fuse

        spatial_w = self.spatial_cascade(channel_w) * channel_w

        spatial_w = spatial_w + fuse

        channel_w = self.channel_parallel(spatial_w)
        spatial_w = self.spatial_parallel(spatial_w)

        w = torch.sigmoid(channel_w * spatial_w)

        fuse = ir * w + vi * (1 - w)

        return fuse


class PatchEmbed(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size, norm_layer):
        super().__init__()

        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x


class EfficientScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, step_size=2):  # noqa
        B, C, org_h, org_w = x.shape
        ctx.shape = (B, C, org_h, org_w)
        ctx.step_size = step_size

        if org_w % step_size != 0:
            pad_w = step_size - org_w % step_size
            x = F.pad(x, (0, pad_w, 0, 0))
        W = x.shape[3]

        if org_h % step_size != 0:
            pad_h = step_size - org_h % step_size
            x = F.pad(x, (0, 0, 0, pad_h))
        H = x.shape[2]

        H = H // step_size
        W = W // step_size

        xs = x.new_empty((B, 4, C, H * W))

        xs[:, 0] = x[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)
        xs[:, 1] = x.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 2] = x[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1)
        xs[:, 3] = x.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)

        xs = xs.view(B, 4, C, -1)
        return xs

    @staticmethod
    def backward(ctx, grad_xs: torch.Tensor):  # noqa

        B, C, org_h, org_w = ctx.shape
        step_size = ctx.step_size

        newH, newW = math.ceil(org_h / step_size), math.ceil(org_w / step_size)
        grad_x = grad_xs.new_empty((B, C, newH * step_size, newW * step_size))

        grad_xs = grad_xs.view(B, 4, C, newH, newW)

        grad_x[:, :, ::step_size, ::step_size] = grad_xs[:, 0].reshape(B, C, newH, newW)
        grad_x[:, :, 1::step_size, ::step_size] = grad_xs[:, 1].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)
        grad_x[:, :, ::step_size, 1::step_size] = grad_xs[:, 2].reshape(B, C, newH, newW)
        grad_x[:, :, 1::step_size, 1::step_size] = grad_xs[:, 3].reshape(B, C, newW, newH).transpose(dim0=2, dim1=3)

        if org_h != grad_x.shape[-2] or org_w != grad_x.shape[-1]:
            grad_x = grad_x[:, :, :org_h, :org_w]

        return grad_x, None


class SelectiveScan(torch.autograd.Function):
    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
        assert nrows in [1, 2, 3, 4], f"{nrows}"
        assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
        ctx.delta_softplus = delta_softplus
        ctx.nrows = nrows
        if u.stride(-1) != 1:
            u = u.contiguous()
        if delta.stride(-1) != 1:
            delta = delta.contiguous()
        if D is not None:
            D = D.contiguous()
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if B.dim() == 3:
            B = B.unsqueeze(dim=1)
            ctx.squeeze_B = True
        if C.dim() == 3:
            C = C.unsqueeze(dim=1)
            ctx.squeeze_C = True

        out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, None, delta_bias, delta_softplus)
        ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
        return out

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dout, *args):
        u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
        if dout.stride(-1) != 1:
            dout = dout.contiguous()

        du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
            u, delta, A, B, C, D, None, delta_bias, dout, x, None, None, ctx.delta_softplus,
            False
        )

        dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
        dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC

        return du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None


class EfficientMerge(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ys: torch.Tensor, ori_h: int, ori_w: int, step_size=2):  # noqa
        B, K, C, L = ys.shape
        H, W = math.ceil(ori_h / step_size), math.ceil(ori_w / step_size)
        ctx.shape = (H, W)
        ctx.ori_h = ori_h
        ctx.ori_w = ori_w
        ctx.step_size = step_size

        new_h = H * step_size
        new_w = W * step_size

        y = ys.new_empty((B, C, new_h, new_w))

        y[:, :, ::step_size, ::step_size] = ys[:, 0].reshape(B, C, H, W)
        y[:, :, 1::step_size, ::step_size] = ys[:, 1].reshape(B, C, W, H).transpose(dim0=2, dim1=3)
        y[:, :, ::step_size, 1::step_size] = ys[:, 2].reshape(B, C, H, W)
        y[:, :, 1::step_size, 1::step_size] = ys[:, 3].reshape(B, C, W, H).transpose(dim0=2, dim1=3)

        if ori_h != new_h or ori_w != new_w:
            y = y[:, :, :ori_h, :ori_w].contiguous()

        y = y.view(B, C, -1)
        return y

    @staticmethod
    def backward(ctx, grad_x: torch.Tensor):  # noqa
        B, C, L = grad_x.shape
        step_size = ctx.step_size

        grad_x = grad_x.view(B, C, ctx.ori_h, ctx.ori_w)

        if ctx.ori_w % step_size != 0:
            pad_w = step_size - ctx.ori_w % step_size
            grad_x = F.pad(grad_x, (0, pad_w, 0, 0))

        if ctx.ori_h % step_size != 0:
            pad_h = step_size - ctx.ori_h % step_size
            grad_x = F.pad(grad_x, (0, 0, 0, pad_h))
        B, C, H, W = grad_x.shape
        H = H // step_size
        W = W // step_size
        grad_xs = grad_x.new_empty((B, 4, C, H * W))

        grad_xs[:, 0] = grad_x[:, :, ::step_size, ::step_size].reshape(B, C, -1)
        grad_xs[:, 1] = grad_x.transpose(dim0=2, dim1=3)[:, :, ::step_size, 1::step_size].reshape(B, C, -1)
        grad_xs[:, 2] = grad_x[:, :, ::step_size, 1::step_size].reshape(B, C, -1)
        grad_xs[:, 3] = grad_x.transpose(dim0=2, dim1=3)[:, :, 1::step_size, 1::step_size].reshape(B, C, -1)

        return grad_xs, None, None, None


class SS2D(nn.Module):
    def __init__(self, d_conv, d_model, d_state, expand):
        super().__init__()

        d_inner = int(expand * d_model)
        dt_rank = math.ceil(d_model / 16)

        self.d_state = d_state
        self.dt_rank = dt_rank

        self.act = nn.SiLU()
        self.dropout = nn.Identity()

        self.conv2d = nn.Conv2d(
            in_channels=d_inner,
            out_channels=d_inner,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            groups=d_inner,
            bias=True
        )

        self.in_proj = nn.Linear(in_features=d_model, out_features=d_inner * 2, bias=False)
        self.out_norm = nn.LayerNorm(d_inner)
        self.out_proj = nn.Linear(in_features=d_inner, out_features=d_model, bias=False)

        self.x_proj = nn.ModuleList([
            nn.Linear(in_features=d_inner, out_features=(dt_rank + d_state * 2), bias=False)
            for _ in range(4)
        ])
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_proj = nn.ModuleList([
            self.dt_init_proj(d_inner=d_inner, dt_rank=dt_rank)
            for _ in range(4)
        ])
        self.dt_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_proj], dim=0))
        self.dt_proj_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_proj], dim=0))

        self.A_logs = self.A_log_init(d_inner)
        self.Ds = self.D_init(d_inner)

    @staticmethod
    def dt_init_proj(d_inner, dt_rank, dt_init_floor=1e-4, dt_scale=1.0, dt_min=0.001, dt_max=0.1):
        dt_proj = nn.Linear(in_features=dt_rank, out_features=d_inner, bias=True)

        dt_init_std = dt_rank ** -0.5 * dt_scale

        nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))

        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_inner):
        A = repeat(
            torch.arange(1, 17, dtype=torch.float32),
            "n -> d n",
            d=d_inner,
        ).contiguous()

        A_log = torch.log(A)
        A_log = repeat(A_log, "d n -> r d n", r=4)
        A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True

        return A_log

    @staticmethod
    def D_init(d_inner):
        D = torch.ones(d_inner)
        D = repeat(D, "n1 -> r n1", r=4)
        D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True

        return D

    def forward_core_v2(self, x: torch.Tensor, step_size=2):
        B, _, H, W = x.shape
        _, N = self.A_logs.shape
        K, _, R = self.dt_proj_weight.shape

        ori_h, ori_w = H, W

        xs = EfficientScan.apply(x, step_size)

        H = math.ceil(H / step_size)
        W = math.ceil(W / step_size)

        L = H * W

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, self.dt_proj_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(self.A_logs.to(torch.float))
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = self.Ds.to(torch.float)
        delta_bias = self.dt_proj_bias.view(-1).to(torch.float)

        ys: torch.Tensor = SelectiveScan.apply(xs, dts, As, Bs, Cs, Ds, delta_bias, True, 1).view(B, K, -1, L)

        ori_h, ori_w = int(ori_h), int(ori_w)

        y = EfficientMerge.apply(ys, ori_h, ori_w, step_size)

        H = ori_h
        W = ori_w

        y = y.transpose(dim0=1, dim1=2).contiguous()
        y = self.out_norm(y).view(B, H, W, -1)

        return y.to(x.dtype)

    def forward(self, x: torch.Tensor):
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        z = self.act(z)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y = self.forward_core_v2(x)
        y = y * z

        out = self.dropout(self.out_proj(y))

        return out


class MambaBlock(nn.Module):
    def __init__(self, hidden_dim, d_conv=3, d_state=16, expand=2):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.ln = nn.LayerNorm(hidden_dim)

        self.ss2d = SS2D(
            d_conv=d_conv,
            d_model=hidden_dim,
            d_state=d_state,
            expand=expand
        )

        self.linear1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim * 4)
        self.linear2 = nn.Linear(in_features=hidden_dim * 4, out_features=hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        residual = x

        x = self.ln(x)
        x = self.ss2d(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        x = x + residual

        return x


class LDC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(LDC, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.kernel_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels // 2, out_channels=out_channels * in_channels * kernel_size * kernel_size,
                      kernel_size=1)
        )

        self.bias_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=in_channels // 4, out_channels=out_channels, kernel_size=1)
        )

        self.enhance = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 2, out_channels=in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, ir_feat, vi_feat):
        B, C, H, W = ir_feat.shape

        concat_feat = torch.cat([ir_feat, vi_feat], dim=1)

        dynamic_kernel = self.kernel_gen(concat_feat)
        dynamic_kernel = dynamic_kernel.view(B, self.out_channels, self.in_channels,
                                             self.kernel_size, self.kernel_size)

        dynamic_bias = self.bias_gen(concat_feat).view(B, self.out_channels)

        enhanced_feat = self.enhance(concat_feat)

        fuse_feat = []
        for i in range(B):
            conv_out = F.conv2d(enhanced_feat[i:i + 1], dynamic_kernel[i],
                                bias=dynamic_bias[i], stride=self.stride, padding=self.padding)
            fuse_feat.append(conv_out)

        fuse_feat = torch.cat(fuse_feat, dim=0)

        return fuse_feat


class GFFM(nn.Module):
    def __init__(self, in_channels, embed_dim, num_layers=2, patch_size=4):
        super(GFFM, self).__init__()

        self.embed_dim = embed_dim
        self.patch_size = patch_size

        self.ir_patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            norm_layer=nn.LayerNorm
        )
        self.vi_patch_embed = PatchEmbed(
            in_channels=in_channels,
            embed_dim=embed_dim,
            patch_size=patch_size,
            norm_layer=nn.LayerNorm
        )

        self.ssm = nn.ModuleList([
            MambaBlock(hidden_dim=embed_dim) for _ in range(num_layers)
        ])

        self.ldc = LDC(in_channels=embed_dim, out_channels=embed_dim)

        self.fusion = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim // 2),
            nn.GELU(),
            nn.Linear(in_features=embed_dim // 2, out_features=embed_dim)
        )

        self.unpatch_embed = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=in_channels * patch_size * patch_size),
            nn.GELU()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, ir, vi):
        B, C, H, W = ir.shape

        ir = self.ir_patch_embed(ir)
        vi = self.vi_patch_embed(vi)

        H_patch, W_patch = H // self.patch_size, W // self.patch_size
        ir = ir.view(B, H_patch, W_patch, self.embed_dim)
        vi = vi.view(B, H_patch, W_patch, self.embed_dim)

        for mamba_block in self.ssm:
            ir = mamba_block(ir)
            vi = mamba_block(vi)

        ir_feat = ir.permute(0, 3, 1, 2)
        vi_feat = vi.permute(0, 3, 1, 2)

        fuse_feat = self.ldc(ir_feat, vi_feat)
        fuse_feat = fuse_feat.permute(0, 2, 3, 1)

        ir_fuse = ir + fuse_feat
        vi_fuse = vi + fuse_feat

        fuse = self.fusion(ir_fuse + vi_fuse)
        fuse = fuse.view(B, -1, self.embed_dim)

        fuse = self.unpatch_embed(fuse)

        fuse = fuse.view(B, H, W, C).permute(0, 3, 1, 2)
        fuse = self.refine(fuse)

        return fuse


class FRM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FRM, self).__init__()

        self.shallow_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.IN = nn.InstanceNorm2d(num_features=in_channels, affine=False, track_running_stats=False)

        self.alpha_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels // 2, out_channels=in_channels, kernel_size=3, padding=1),
        )
        self.beta_conv = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels // 2, out_channels=in_channels, kernel_size=3, padding=1),
        )

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.recon_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, fuse_s, fuse_g):
        alpha = self.alpha_conv(fuse_g)
        beta = self.beta_conv(fuse_g)

        fuse_s = self.IN(self.shallow_conv(fuse_s))
        fuse = alpha * fuse_s + beta

        fuse_d1 = self.block1(fuse)
        fuse_d2 = self.block2(fuse + fuse_d1)
        fuse_d3 = self.block3(fuse_d2 + fuse_d1)

        fuse = fuse + fuse_d1 + fuse_d2 + fuse_d3

        fuse = self.recon_conv(fuse)

        return fuse


class DAMFusion(nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, embed_dim=128):
        super(DAMFusion, self).__init__()

        self.ir_conv = ConvBlock(in_channels=in_channels, out_channels=hidden_channels)
        self.vi_conv = ConvBlock(in_channels=in_channels, out_channels=hidden_channels)

        self.SFFM = SFFM(channels=hidden_channels)
        self.GFFM = GFFM(in_channels=hidden_channels, embed_dim=embed_dim)
        self.FRM = FRM(in_channels=hidden_channels, out_channels=hidden_channels)

    def forward(self, ir, vi):
        ir = self.ir_conv(ir)
        vi = self.vi_conv(vi)

        fuse_s = self.SFFM(ir, vi)
        fuse_g = self.GFFM(ir, vi)

        fuse = self.FRM(fuse_s, fuse_g)

        return fuse
