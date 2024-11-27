## Segmentation
Train Spike-driven transformer with FPN as pixel decoder in ADE20k
- `cd tools`
- `CUDA_VISIBLE_DEVICES=0,1 ./dist_train.sh ../configs/EFSDTv2/fpn_sdtv3_512x512_19M_ade20k.py 1`