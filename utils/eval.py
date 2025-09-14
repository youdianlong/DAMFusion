import numpy as np
import sklearn.metrics as skm

from skimage.metrics import structural_similarity as ssim

from utils.metric_utils import get_VIFF, get_Qabf_Array, get_Qabf


def EN(img):
    a = np.uint8(np.round(img)).flatten()
    h = np.bincount(a) / a.shape[0]

    return -sum(h * np.log2(h + (h == 0)))


def SD(img):
    return np.std(img)


def SF(img):
    return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2))


def MI(ir_img, vi_img, fuse_img):
    return skm.mutual_info_score(fuse_img.flatten(), ir_img.flatten()) + skm.mutual_info_score(fuse_img.flatten(),
                                                                                               vi_img.flatten())


def SCD(ir_img, vi_img, fuse_img):
    fuse_ir_img = fuse_img - ir_img
    fuse_vi_img = fuse_img - vi_img

    corr1 = np.sum((ir_img - np.mean(ir_img)) * (fuse_vi_img - np.mean(fuse_vi_img))) / np.sqrt(
        (np.sum((ir_img - np.mean(ir_img)) ** 2)) * (np.sum((fuse_vi_img - np.mean(fuse_vi_img)) ** 2)))
    corr2 = np.sum((vi_img - np.mean(vi_img)) * (fuse_ir_img - np.mean(fuse_ir_img))) / np.sqrt(
        (np.sum((vi_img - np.mean(vi_img)) ** 2)) * (np.sum((fuse_ir_img - np.mean(fuse_ir_img)) ** 2)))

    return corr1 + corr2


def VIFF(ir_img, vi_img, fuse_img):
    return get_VIFF(ir_img, fuse_img) + get_VIFF(vi_img, fuse_img)


def Qabf(ir_img, vi_img, fuse_img):
    ir_g, ir_a = get_Qabf_Array(ir_img)
    vi_g, vi_a = get_Qabf_Array(vi_img)
    fuse_g, fuse_a = get_Qabf_Array(fuse_img)

    QAF = get_Qabf(ir_a, ir_g, fuse_a, fuse_g)
    QBF = get_Qabf(vi_a, vi_g, fuse_a, fuse_g)

    deno = np.sum(ir_g + vi_g)
    nume = np.sum(np.multiply(QAF, ir_g) + np.multiply(QBF, vi_g))

    return nume / deno


def SSIM(ir_img, vi_img, fuse_img):
    return ssim(fuse_img, ir_img) + ssim(fuse_img, vi_img)


def eval_one(ir_img, vi_img, fuse_img):
    EN_score = EN(fuse_img)
    SD_score = SD(fuse_img)
    SF_score = SF(fuse_img)
    MI_score = MI(ir_img, vi_img, fuse_img)
    SCD_score = SCD(ir_img, vi_img, fuse_img)
    VIFF_score = VIFF(ir_img, vi_img, fuse_img)
    QABF_score = Qabf(ir_img, vi_img, fuse_img)
    SSIM_score = SSIM(ir_img, vi_img, fuse_img)

    return EN_score, SD_score, SF_score, MI_score, SCD_score, VIFF_score, QABF_score, SSIM_score
