
### Results on ADE20K
First download
Pre-trained ckpts 10M: [here](https://drive.google.com/file/d/1pHrampLjyE1kLr-4DS1WgSdnCVPzL6Tq/view?usp=sharing).
Pre-trained ckpts 19M: [here](https://drive.google.com/file/d/1pSGCOzrZNgHDxQXAp-Uelx61snIbQC1H/view?usp=drive_link).

Train 10M on 4 GPUs:

```shell
bash ./dist_train_2.sh ../configs/EFSDTv2/fpn_sdtv3_512x512_10M_ade20k.py 4
```


### Results on COCO2017:
First download
Pre-trained ckpts 10M: [here](https://drive.google.com/file/d/1pHrampLjyE1kLr-4DS1WgSdnCVPzL6Tq/view?usp=sharing).
Pre-trained ckpts 19M: [here](https://drive.google.com/file/d/1pSGCOzrZNgHDxQXAp-Uelx61snIbQC1H/view?usp=drive_link).

Train 10M on 4 GPUs:

```shell
bash ./dist_train_2.sh ../configs/EFSDTv2/fpn_sdtv3_512x512_10M_ade20k.py 4
```

## Segmentation
Train Spike-driven transformer with FPN as pixel decoder in ADE20k
- `cd tools`
- `CUDA_VISIBLE_DEVICES=0,1 ./dist_train.sh ../configs/EFSDTv2/fpn_sdtv3_512x512_19M_ade20k.py 1`
