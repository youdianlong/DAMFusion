import cv2
import os
import torch

import numpy as np

from tqdm import tqdm

from network import DAMFusion

device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')


def load_img(ir_path, vi_path):
    ir_img = cv2.imread(ir_path, cv2.IMREAD_GRAYSCALE)
    ir_img = ir_img.astype(np.float32) / 255.0
    ir_img = torch.from_numpy(ir_img).unsqueeze(0).unsqueeze(0)

    vi_img = cv2.imread(vi_path, cv2.IMREAD_COLOR)
    vi_img = cv2.cvtColor(vi_img, cv2.COLOR_BGR2RGB)
    vi_img_YCbCr = cv2.cvtColor(vi_img, cv2.COLOR_RGB2YCrCb)

    vi_img = vi_img_YCbCr[:, :, 0]
    vi_img_CbCr = vi_img_YCbCr[:, :, 1:3]

    vi_img = vi_img.astype(np.float32) / 255.0
    vi_img = torch.from_numpy(vi_img).unsqueeze(0).unsqueeze(0)

    return ir_img, vi_img, vi_img_CbCr, vi_img_YCbCr.shape[:2]


def main():
    ir_path = '...'
    vi_path = '...'
    fuse_path = '...'
    model_path = '...'

    model = DAMFusion().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for img_name in tqdm(os.listdir(ir_path)):
        ir_img, vi_img, vi_img_CbCr, original_shape = load_img(
            ir_path=os.path.join(ir_path, img_name),
            vi_path=os.path.join(vi_path, img_name)
        )

        ir_img = ir_img.to(device)
        vi_img = vi_img.to(device)

        with torch.no_grad():
            fuse_img = model(ir_img, vi_img)

            fuse_img = fuse_img.squeeze().cpu().numpy()
            fuse_img = (fuse_img * 255.0).astype(np.uint8)

            if fuse_img.shape != original_shape:
                fuse_img = cv2.resize(fuse_img, (original_shape[1], original_shape[0]))

            fuse_img_YCbCr = np.zeros((original_shape[0], original_shape[1], 3), dtype=np.uint8)
            fuse_img_YCbCr[:, :, 0] = fuse_img
            fuse_img_YCbCr[:, :, 1:3] = vi_img_CbCr

            fuse_img = cv2.cvtColor(fuse_img_YCbCr, cv2.COLOR_YCrCb2RGB)
            fuse_img = cv2.cvtColor(fuse_img, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(fuse_path, img_name), fuse_img)


if __name__ == "__main__":
    main()
