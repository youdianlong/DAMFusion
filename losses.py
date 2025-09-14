import torch

import torch.nn.functional as F

from math import exp
from torch import nn
from torch.autograd import Variable


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])

    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())

    return window


def ssim(img1, img2, window_size=11):
    (_, channel, _, _) = img1.size()

    window = create_window(window_size, channel)
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()


class Sobel_xy(nn.Module):
    def __init__(self):
        super(Sobel_xy, self).__init__()

        kernel_x = [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]
        kernel_y = [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1]
        ]

        kernel_x = torch.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)

        self.weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
        self.weight_y = nn.Parameter(data=kernel_y, requires_grad=False)

    def forward(self, x):
        sobel_x = F.conv2d(x, self.weight_x.to(x.device), padding=1)
        sobel_y = F.conv2d(x, self.weight_y.to(x.device), padding=1)

        return torch.abs(sobel_x) + torch.abs(sobel_y)


class Loss_Intensity(nn.Module):
    def __init__(self):
        super(Loss_Intensity, self).__init__()

    def forward(self, ir, vi, fuse):
        intensity_joint = torch.max(ir, vi)
        loss = F.l1_loss(fuse, intensity_joint)

        return loss


class Loss_Gradient(nn.Module):
    def __init__(self):
        super(Loss_Gradient, self).__init__()

        self.sobel = Sobel_xy()

    def forward(self, ir, vi, fuse):
        ir_grad = self.sobel(ir)
        vi_grad = self.sobel(vi)
        fuse_grad = self.sobel(fuse)

        joint_grad = torch.max(ir_grad, vi_grad)
        loss = F.l1_loss(fuse_grad, joint_grad)

        return loss


class Loss_SSIM(nn.Module):
    def __init__(self):
        super(Loss_SSIM, self).__init__()

        self.sobel = Sobel_xy()

    def forward(self, ir, vi, fuse):
        ir_grad = self.sobel(ir)
        vi_grad = self.sobel(vi)

        ir_weight = torch.mean(ir_grad) / (torch.mean(ir_grad) + torch.mean(vi_grad))
        vi_weight = torch.mean(vi_grad) / (torch.mean(ir_grad) + torch.mean(vi_grad))

        loss = ir_weight * ssim(ir, fuse) + vi_weight * ssim(vi, fuse)

        return loss


class FusionLoss(nn.Module):
    def __init__(self, task='IVF'):
        super(FusionLoss, self).__init__()

        self.task = task

        self.intensity = Loss_Intensity()
        self.gradient = Loss_Gradient()
        self.ssim = Loss_SSIM()

    def forward(self, ir, vi, fuse):
        int_loss = self.intensity(ir, vi, fuse)

        grad_loss = self.gradient(ir, vi, fuse)
        ssim_loss = 1 - self.ssim(ir, vi, fuse)

        if self.task == 'IVF':
            total_loss = 8.0 * int_loss + 10.0 * grad_loss + 6.0 * ssim_loss
        else:
            total_loss = 12.0 * int_loss + 10.0 * grad_loss + 5.0 * ssim_loss

        return total_loss, grad_loss, int_loss, ssim_loss
