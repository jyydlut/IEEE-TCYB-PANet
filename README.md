# PANet: Patch-Aware Network for Light Field Salient Object Detection

# Introduction

Accepted paper in IEEE Trans on Cybernetics, 'PANet: Patch-Aware Network for Light Field Salient Object Detection', Yongri Piao, Yongyao Jiang, Miao Zhang, Jian Wang and Huchuan Lu.
![Network architecture](https://github.com/jyydlut/IEEE-TCYB-PANet/blob/main/img/network.jpg)


# Requirements

Windows 10

PyTorch 1.4.0

CUDA 10.0

Cudnn 7.6.0

Python 3.6.5

Numpy 1.16.4

# Training 

Modify your path of training dataset in config.py

Run train.py for training the saliency model, the maximum of training iterations is 500000.

Run train_mslm.py for training the MSLM model, the maximum of training iterations is 5000.

Run train_srm.py for training the SRM model, the maximum of training iterations is 5000.

Run train_second_decoder.py for training the second decoder, the maximum of training iterations is 500000.

# Testing

Download pretrained models from [here](https://pan.baidu.com/s/1zUtCIHJsZhfRP_ldkmzozg). Code: qwer

Modify your path of testing dataset in config.py

Run test to inference saliency maps

# Saliency Maps

DUTS-LFSD&HFUT-LFSD&LFSD, [Download link](https://pan.baidu.com/s/1tf1GNfxEAO456qsUySL72A). Code: qwer

# Contact and Questions

Contact:Yongyao Jiang. Email:jiangyy@mail.dlut.edu.cn
