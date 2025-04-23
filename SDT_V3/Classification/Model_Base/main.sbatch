#!/bin/bash
#SBATCH -J LISNN_IN200_li
#SBATCH --output=slurm-%j_x.out
#SBATCH --array=0
#SBATCH -c 8
#SBATCH --time=00-04:00:00
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=48GB
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16

set -e
set -x

# Activate virtual environment
source $HOME/venvs/lisnn/bin/activate

# Set distributed training variables
BASE_PORT=12346
export MASTER_PORT=$((BASE_PORT + SLURM_ARRAY_TASK_ID))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export OMP_NUM_THREADS=8

echo "MASTER_ADDR=$MASTER_ADDR"
echo "WORLD_SIZE=$WORLD_SIZE"

# Runtime variables
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SEED=32
SESSION_ID=$(openssl rand -hex 4)

# Paths
TRAIN_SCRIPT="$HOME/Spike-Driven-Transformer-V3/SDT_V3/Classification/Model_Base/main_finetune.py"
DATASET_NAME="ImageNet-200"
DATASET_DIR="$TMPDIR/dataset/${DATASET_NAME}"

mkdir -p "$TMPDIR/dataset"

# Extract dataset
SECONDS=0
echo "Extracting ${DATASET_NAME}.tar"
tar -xf "/scratch/p315895/datasets/${DATASET_NAME}.tar" -C "$TMPDIR/dataset"
echo "Time taken to extract dataset: ${SECONDS} seconds"

# Training configuration
COMMON_ARGS="--batch_size 256 \
  --blr 6e-4 \
  --warmup_epochs 5 \
  --epochs 20 \
  --model Efficient_Spiking_Transformer_s \
  --data_path ${DATASET_DIR} \
  --output_dir outputs/T1 \
  --log_dir outputs/T1 \
  --model_mode ms \
  --dist_eval "
EXPERIMENT_NAME="${DATASET_NAME}_${TIMESTAMP}_s${SEED}"

# Initialize job array
case ${SLURM_ARRAY_TASK_ID} in
0) torchrun --standalone --nproc_per_node=2 ${TRAIN_SCRIPT} ${COMMON_ARGS} --name ${EXPERIMENT_NAME} --wandb_tags small li ${DATASET_NAME} --lateral_inhibition;;
1) torchrun --standalone --nproc_per_node=2 ${TRAIN_SCRIPT} ${COMMON_ARGS} --name ${EXPERIMENT_NAME} --wandb_tags small ${DATASET_NAME};;
esac

echo "script finished"

<<'END'



