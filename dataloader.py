import cv2
import os
import random
import torch

import numpy as np

from natsort import natsorted
from torch.utils.data import Dataset


class FusionData(Dataset):
    def __init__(self, ir_path, vi_path, patch_size=256):
        super(FusionData, self).__init__()

        self.ir_path = ir_path
        self.vi_path = vi_path
        self.patch_size = patch_size

        self.name_list = natsorted(os.listdir(self.ir_path))

    def _random_crop(self, ir_img, vi_img):
        h, w = ir_img.shape
        if h <= self.patch_size and w <= self.patch_size:
            pad_h = max(0, self.patch_size - h)
            pad_w = max(0, self.patch_size - w)
            ir_img = np.pad(ir_img, ((0, pad_h), (0, pad_w)), mode='reflect')
            vi_img = np.pad(vi_img, ((0, pad_h), (0, pad_w)), mode='reflect')
        else:
            top = random.randint(0, max(0, h - self.patch_size))
            left = random.randint(0, max(0, w - self.patch_size))
            ir_img = ir_img[top:top + self.patch_size, left:left + self.patch_size]
            vi_img = vi_img[top:top + self.patch_size, left:left + self.patch_size]

        return ir_img, vi_img

    def _augment_data(self, ir_img, vi_img):
        if random.random() < 0.5:
            ir_img = np.fliplr(ir_img).copy()
            vi_img = np.fliplr(vi_img).copy()

        if random.random() < 0.3:
            ir_img = np.flipud(ir_img).copy()
            vi_img = np.flipud(vi_img).copy()

        if random.random() < 0.3:
            k = random.randint(1, 3)
            ir_img = np.rot90(ir_img, k).copy()
            vi_img = np.rot90(vi_img, k).copy()

        return ir_img, vi_img

    def __getitem__(self, index):
        img_name = self.name_list[index]

        ir_img = cv2.imread(os.path.join(self.ir_path, img_name), cv2.IMREAD_GRAYSCALE)
        vi_img = cv2.imread(os.path.join(self.vi_path, img_name))

        if len(vi_img.shape) == 3:
            vi_img = cv2.cvtColor(vi_img, cv2.COLOR_BGR2GRAY)

        ir_img, vi_img = self._random_crop(ir_img, vi_img)
        ir_img, vi_img = self._augment_data(ir_img, vi_img)

        ir_img = ir_img.astype(np.float32) / 255.0
        vi_img = vi_img.astype(np.float32) / 255.0

        ir_img = torch.from_numpy(ir_img).unsqueeze(0)
        vi_img = torch.from_numpy(vi_img).unsqueeze(0)

        return vi_img, ir_img

    def __len__(self):
        return len(self.name_list)
