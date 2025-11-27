#!/bin/bash
#SBATCH --job-name=train_model
#SBATCH --output=outputs/train_model_output_%j.log
#SBATCH --error=outputs/train_model_error_%j.log
#SBATCH --partition=main
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

export HF_HOME="/nfs/abush/.cache/huggingface/"
export HF_DATASETS_CACHE="/nfs/abush/.cache/huggingface/datasets"
export HF_MODULES_CACHE="/nfs/abush/.cache/huggingface/modules"

source ./venv/bin/activate

CMD="python train_model.py \
    --pick_technique next \
    --num_pairs 100000 \
    --lr 2e-5 \
    --batch_size 192 \
    --optimizer AdamW \
    --warmup_steps 100 \
    --train_data data/train_flattened_500000.csv \
    --val_data data/val_flattened_10000.csv \
    --test_data data/test_flattened_10000.csv \
    --epochs 1 \
    --output_dir models"
echo "Running: $CMD"
srun -u $CMD

