#!/bin/bash

python firing_num_dvs.py \
--batch_size \
24 \
--model \
Efficient_Spiking_Transformer_scaling_8_19 \
--data_path \
/lxh/lxh_data/HARDVS/rawframe \
--output_dir \
./output_dir/test \
--log_dir \
./output_dir/test \
--aa \
rand-m7-mstd0.5-inc1 \
--dist_eval \
--finetune \
./output_dir/train1/best.pth

#CUDA_VISIBLE_DEVICES=0,3,5,6 torchrun \
#--standalone \
#--nproc_per_node=4 \
#firing_num_dvs.py \
#--batch_size \
#16 \
#--model \
#Efficient_Spiking_Transformer_scaling_8_19 \
#--data_path \
#/lxh/lxh_data/HARDVS/rawframe \
#--output_dir \
#./output_dir/test2 \
#--log_dir \
#./output_dir/test2 \
#--aa \
#rand-m7-mstd0.5-inc1 \
#--dist_eval \
#--finetune \
#./output_dir/train8/best.pth