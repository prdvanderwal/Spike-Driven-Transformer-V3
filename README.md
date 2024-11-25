# Spike-Driven-Transformer-V3
# Gated Attention Coding for Training High-performance and Efficient Spiking Neural Networks ([AAAI24](https://arxiv.org/abs/2308.06582))
Man Yao*, Xuerui Qiu*
[Man Yao*](https://scholar.google.com/citations?user=eE4vvp0AAAAJ),[Xuerui Qiu*](https://scholar.google.com/citations?user=bMwW4e8AAAAJ&hl=zh-CN), [Jiakui Hu](https://github.com/jkhu29), [Tianxiang Hu](), [Guoqi Li](https://scholar.google.com/citations?user=qCfE--MAAAAJ&).

* Equal contribution.
BICLab, Institute of Automation, Chinese Academy of Sciences



:rocket:  :rocket:  :rocket: **News**:

- **Dec. 19, 2023**: Release the code for training and testing.

## Abstract
The ambition of brain-inspired Spiking Neural Networks (SNNs) is to become a low-power alternative to traditional Artificial Neural Networks (ANNs). This work addresses two major challenges in realizing this vision: the performance gap between SNNs and ANNs, and the high training costs of SNNs. We identify intrinsic flaws in spiking neurons caused by binary firing mechanisms and propose a Spike Firing Approximation (SFA) method using integer training and spike-driven inference. This optimizes the spike firing pattern of spiking neurons, enhancing efficient training, reducing power consumption, improving performance, enabling easier scaling, and better utilizing neuromorphic chips. We also develop an efficient spike-driven Transformer architecture and a spike-masked autoencoder to prevent performance degradation during SNN scaling. On ImageNet-1k, we achieve state-of-the-art top-1 accuracy of 78.5\%, 79.8\%, 84.0\%, and 86.2\% with models containing 10M, 19M, 83M, and 173M parameters, respectively. For instance, the 10M model outperforms the best existing SNN by 7.2\% on ImageNet, with training time acceleration and inference energy efficiency improved by 4.5$\times$ and 3.9$\times$, respectively. We validate the effectiveness and efficiency of the proposed method across various tasks, including object detection, semantic segmentation, and neuromorphic vision tasks. This work enables SNNs to match ANN performance while maintaining the low-power advantage, marking a significant step towards SNNs as a general visual backbone.



## Prerequisites
The Following Setup is tested and it is working:
 * Python 3.7
 * Pytorch 1.8.0
 * Cuda 10.2






## Data Prepare

- use `PyTorch` to load the CIFAR10 and CIFAR100 dataset.
Tree in `./data/`.



ImageNet with the following folder structure, you can extract imagenet by this [script](https://gist.github.com/BIGBALLON/8a71d225eff18d88e469e6ea9b39cef4).

```shell
│imagenet/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```

## Contact Information

```

```

For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact `qiuxuerui2024@ia.ac.cn`.
