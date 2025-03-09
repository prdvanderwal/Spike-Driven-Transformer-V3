#!/bin/bash
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
  --standalone \
  --nproc_per_node=4 \
  --master_port=12345 \
  firing_num_dvs.py \
  --batch_size 4 \
  --model Efficient_Spiking_Transformer_scaling_8_19M \
  --data_path /dev/shm \
  --output_dir /raid/ligq/wkm/sdsa_v2_hardvs/output_dir/hardvs_8bit_firing \
  --log_dir /raid/ligq/wkm/sdsa_v2_hardvs/output_dir/hardvs_8bit_firing \
  --blr 4e-5  \
  --weight_decay 1e-2 \
  --aa rand-m7-mstd0.5-inc1 \
  --num_workers 16 \
  --seed 42 \
  --dist_eval