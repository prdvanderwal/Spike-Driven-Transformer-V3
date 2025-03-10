


### Train

```shell
# Set the CUDA devices to be used for the process
export CUDA_VISIBLE_DEVICES=5,6,7

# Run the torch distributed training script with specified parameters
torchrun \
  --standalone \
  --nproc_per_node=3 \
  main_finetune.py \
  --batch_size 24 \
  --epochs 100 \
  --model Efficient_Spiking_Transformer_scaling_8_19M \
  --data_path ${data_pth} \
  --output_dir  ${output_dir}\
  --log_dir ${log_dir} \
  --blr 4e-5 \
  --weight_decay 1e-2 \
  --aa rand-m7-mstd0.5-inc1 \
  --num_workers 16 \
  --seed 42 \
  --dist_eval
```
