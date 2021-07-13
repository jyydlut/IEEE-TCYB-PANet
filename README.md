# PANet
The code for "PANet: Patch-Aware Network for Light Field Salient Object Detection"

# Requirements

Windows 10

PyTorch 1.4.0

CUDA 10.0

Cudnn 7.6.0

Python 3.6.5

Numpy 1.16.4

# Training 

Modify your path of training dataset in config.py
Run train.py for training the saliency model
Run train_mslm.py for training the MSLM model
Run train_srm.py for training the SRM model
Run train_second_decoder.py for training the second decoder

# Testing

Download pretrained models from here. Code:
Modify your path of testing dataset in config.py
Run test to inference saliency maps

# Saliency Maps

DUTS-LFSD. Code:  HFUT-LFSD. Code:  LFSD. Code:

# Contact and Questions

Contact:Yongyao Jiang. Email:jiangyy@mail.dlut.edu.cn
