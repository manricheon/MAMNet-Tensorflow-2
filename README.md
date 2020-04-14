# MAMNet-Tensorflow-2
This repository is an unofficial Tensorflow 2.0 implementation of the paper "MAMNet: Multi-path Adaptive Modulation Network for Image Super-Resolution". Official Tensorflow implementation can be found here >> [[Official MAMNet](https://github.com/junhyukk/MAMNet-Tensorflow)]. If you have any interests or questions about this work, do not hesitate to contact us.  [[arXiv](https://arxiv.org/abs/1811.12043)]

## Introduction
![teaser_image](figures/teaser_image.png)
In recent years, single image super-resolution (SR) methods based on deep convolutional neural networks (CNNs) have made significant progress. However, due to the non-adaptive nature of the convolution operation, they cannot adapt to various characteristics of images, which limits their representational capability and, consequently, results in unnecessarily large model sizes.
To address this issue, we propose a novel multi-path adaptive modulation network (MAMNet).
Specifically, we propose a multi-path adaptive modulation block (MAMB), which is a lightweight yet effective residual block that adaptively modulates residual feature responses by fully exploiting their information via three paths.
The three paths model three types of information suitable for SR: 1) channel-specific information (CSI) using global variance pooling, inter-channel dependencies (ICD) based on the CSI, and channel-specific spatial dependencies (CSD) via depth-wise convolution.

The overall architecture of MAMNet is illustrated as follows:
<br/><br/><br/><br/>
![MAMNet](figures/MAMNet.png)
<br/><br/>
The structure of MAMB is illustrated as follows:
<br/><br/>
![MAMB](figures/MAMB.png)

## Training

```shell
python main.py
  --gpu_id 0
  --model_name MAMNet
  --dataset_dir <path of the DIV2K dataset>
  --dataset_name DIV2K
  --exp_dir <path of experiments>
  --exp_name <name of experiment> 
  --num_res 64 --num_feats 64 
  --is_MAM --is_CSI --is_ICD --is_CSD 
  --scale <scaling factor> 
  --is_init_res 
  --is_train 
```

## Inference (Test)

``` shell
python inference.py 
  --gpu_id 0 
  --model_name MAMNet 
  --test_input <path of input dir>
  --test_output <path of output dir>
  --exp_dir <path of experiments> 
  --exp_name <name of experiment>  
  --num_res 64 --num_feats 64 
  --is_MAM --is_CSI --is_ICD --is_CSD 
  --scale <scaling factor>
```

## Acknowledgement
Many parts are learned and borrowed from useful repositories below. Thanks.
- https://github.com/krasserm/super-resolution
- https://github.com/junhyukk/MAMNet-Tensorflow
