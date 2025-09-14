# DAMFusion

## Description
This project includes the source code of the DAMFusion(DAMFusion: Cross-modal Image Fusion via Dual Attention and Mamba).

## Environment

For key packages

```
torch==1.13.1
torchvision==0.14.1

mamba-ssm==1.0.1
casual-conv1d==1.0.0
```

## Train

For train, you should first place your path in `train.py`, then run `python train.py`


## Test

For test, you should first place your path in `test.py`, then run `python test.py` to get fused images.

For metrics, you can find eval scripts in `./utils/eval.py` and `./utils/metrics_utils.py`

## Notes
This package is free for academic usage. For other purposes, please contact Prof. Dian-Long You (youdianlong@sina.com). This package was developed by Yu-Long Wang (yulongwang@stumail.ysu.edu.cn). For any problem concerning the code, please feel free to contact.
