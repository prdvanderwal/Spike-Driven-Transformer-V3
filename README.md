# Scaling Spike-driven Transformer with Efficient Spike Firing Approximation Training ()

[Man Yao*](https://scholar.google.com/citations?user=eE4vvp0AAAAJ), [Xuerui Qiu*](https://scholar.google.com/citations?user=bMwW4e8AAAAJ&hl=zh-CN), [Tianxiang Hu](), [Jiakui Hu](https://github.com/jkhu29), [Yuhong Chou](https://scholar.google.com/citations?user=8CpWM4cAAAAJ&hl=zh-CN&oi=ao), [Keyu Tian](https://scholar.google.com/citations?user=6FdkbygAAAAJ&hl=zh-CN&oi=ao), [Jianxing Liao](), [Luziwei Leng](), [Bo Xu](), [Guoqi Li](https://scholar.google.com/citations?user=qCfE--MAAAAJ&)


*Equal contribution.

BICLab, Institute of Automation, Chinese Academy of Sciences



:rocket:  :rocket:  :rocket: **News**:

- **Dec. 19, 2023**: Release the code for training and testing.

## Abstract
The ambition of brain-inspired Spiking Neural Networks (SNNs) is to become a low-power alternative to traditional Artificial Neural Networks (ANNs). This work addresses two major challenges in realizing this vision: the performance gap between SNNs and ANNs, and the high training costs of SNNs. We identify intrinsic flaws in spiking neurons caused by binary firing mechanisms and propose a Spike Firing Approximation (SFA) method using integer training and spike-driven inference. This optimizes the spike firing pattern of spiking neurons, enhancing efficient training, reducing power consumption, improving performance, enabling easier scaling, and better utilizing neuromorphic chips. We also develop an efficient spike-driven Transformer architecture and a spike-masked autoencoder to prevent performance degradation during SNN scaling. On ImageNet-1k, we achieve state-of-the-art top-1 accuracy of 78.5\%, 79.8\%, 84.0\%, and 86.2\% with models containing 10M, 19M, 83M, and 173M parameters, respectively. For instance, the 10M model outperforms the best existing SNN by 7.2\% on ImageNet, with training time acceleration and inference energy efficiency improved by 4.5$\times$ and 3.9$\times$, respectively. We validate the effectiveness and efficiency of the proposed method across various tasks, including object detection, semantic segmentation, and neuromorphic vision tasks. This work enables SNNs to match ANN performance while maintaining the low-power advantage, marking a significant step towards SNNs as a general visual backbone.


## Classification

### Requirements

```python3
pytorch >= 2.0.0
cupy
spikingjelly == 0.0.0.0.12
```

### Results on Imagenet-1K

Pre-trained ckpts and training logs of 84M: [here](https://drive.google.com/drive/folders/12JcIRG8BF6JcgPsXIetSS14udtHXeSSx?usp=sharing).

### Train & Test

The hyper-parameters are in `./conf/`.

Train:

```shell
torchrun --standalone --nproc_per_node=8 \
  main_finetune.py \
  --batch_size 128 \
  --blr 6e-4 \
  --warmup_epochs 10 \
  --epochs 200 \
  --model metaspikformer_8_512 \
  --data_path /your/data/path \
  --output_dir outputs/T1 \
  --log_dir outputs/T1 \
  --model_mode ms \
  --dist_eval
```

Distillation:

> Please download caformer_b36_in21_ft1k.pth first following [PoolFormer](https://github.com/sail-sg/poolformer).

```shell
torchrun --standalone --nproc_per_node=8 \
  main_finetune.py \
  --batch_size 24 \
  --blr 2e-5 \
  --warmup_epochs 5 \
  --epochs 50 \
  --model metaspikformer_8_512 \
  --data_path /your/data/path \
  --output_dir outputs/T4 \
  --log_dir outputs/T4 \
  --model_mode ms \
  --dist_eval \
  --finetune /your/ckpt/path \
  --time_steps 4 \
  --kd \
  --teacher_model caformer_b36_in21ft1k \
  --distillation_type hard
```

Test:

```shell
python main_finetune.py --batch_size 128 --model metaspikformer_8_512 --data_path /your/data/path --eval --resume /your/ckpt/path
```

### Data Prepare

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

### Results on ADE20K
First download
Pre-trained ckpts 10M: [here](https://drive.google.com/drive/folders/12JcIRG8BF6JcgPsXIetSS14udtHXeSSx?usp=sharing).
Pre-trained ckpts 19M: [here](https://drive.google.com/drive/folders/12JcIRG8BF6JcgPsXIetSS14udtHXeSSx?usp=sharing).

Train 10M on 4 GPUs:

```shell
bash ./dist_train_2.sh ../configs/EFSDTv2/fpn_sdtv3_512x512_10M_ade20k.py 4
```


### Results on COCO2017:
First download
Pre-trained ckpts 10M: [here](https://drive.google.com/drive/folders/12JcIRG8BF6JcgPsXIetSS14udtHXeSSx?usp=sharing).
Pre-trained ckpts 19M: [here](https://drive.google.com/drive/folders/12JcIRG8BF6JcgPsXIetSS14udtHXeSSx?usp=sharing).

Train 10M on 4 GPUs:

```shell
bash ./dist_train_2.sh ../configs/EFSDTv2/fpn_sdtv3_512x512_10M_ade20k.py 4
```

## Contact Information

```

```

For help or issues using this git, please submit a GitHub issue.

For other communications related to this git, please contact `manyao@ia.ac.cn` and `qiuxuerui2024@ia.ac.cn`.

## Thanks

Our implementation is mainly based on the following codebases. We gratefully thank the authors for their wonderful works.

[Deit](https://github.com/facebookresearch/deit), [MCMAE](https://github.com/Alpha-VL/ConvMAE), [Spark](https://github.com/keyu-tian/SparK).

