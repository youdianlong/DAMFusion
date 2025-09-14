import math

import numpy as np

from scipy.signal import convolve2d


def get_VIFF(img1, img2):
    sigma_nsq = 2
    eps = 1e-10

    num = 0.0
    den = 0.0

    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0

        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()

        assert sumh != 0, "sumh is None"

        win = h / sumh

        if scale > 1:
            img1 = convolve2d(img1, np.rot90(win, 2), mode='valid')
            img2 = convolve2d(img2, np.rot90(win, 2), mode='valid')
            img1 = img1[::2, ::2]
            img2 = img2[::2, ::2]

        mu1 = convolve2d(img1, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(img2, np.rot90(win, 2), mode='valid')

        mu1_sq = mu1 * mu1
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = convolve2d(img1 * img1, np.rot90(win, 2), mode='valid') - mu1_sq
        sigma2_sq = convolve2d(img2 * img2, np.rot90(win, 2), mode='valid') - mu2_sq
        sigma12 = convolve2d(img1 * img2, np.rot90(win, 2), mode='valid') - mu1_mu2

        sigma1_sq[sigma1_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        g = sigma12 / (sigma1_sq + eps)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < eps] = 0
        sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
        sigma1_sq[sigma1_sq < eps] = 0

        g[sigma2_sq < eps] = 0
        sv_sq[sigma2_sq < eps] = 0

        sv_sq[g < 0] = sigma2_sq[g < 0]
        g[g < 0] = 0
        sv_sq[sv_sq <= eps] = eps

        num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

    vifp = num / den

    if np.isnan(vifp):
        return 1.0
    else:
        return vifp


def get_Qabf_Array(img):
    h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
    h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

    Sx = convolve2d(img, h3, mode='same')
    Sy = convolve2d(img, h1, mode='same')

    g = np.sqrt(np.multiply(Sx, Sx) + np.multiply(Sy, Sy))

    a = np.zeros_like(img)
    a[Sx == 0] = math.pi / 2
    a[Sx != 0] = np.arctan(Sy[Sx != 0] / Sx[Sx != 0])

    return g, a


def get_Qabf(aA, gA, aF, gF):
    Tg = 0.9994
    kg = -15
    Dg = 0.5
    Ta = 0.9879
    ka = -22
    Da = 0.8

    GAF, AAF, QgAF, QaAF, QAF = np.zeros_like(aA), np.zeros_like(aA), np.zeros_like(aA), np.zeros_like(
        aA), np.zeros_like(aA)

    GAF[gA > gF] = gF[gA > gF] / gA[gA > gF]
    GAF[gA == gF] = gF[gA == gF]
    GAF[gA < gF] = gA[gA < gF] / gF[gA < gF]
    AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
    QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
    QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
    QAF = QgAF * QaAF

    return QAF
