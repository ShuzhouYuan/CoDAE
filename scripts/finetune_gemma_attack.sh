#!/bin/bash
# Slurm directives
#SBATCH --job-name=finetune_gemma_attack
#SBATCH --output=./logs_gemma_attack/%x_%j.out
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=24G
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00

MODEL_NAME="google/gemma-2-9b-it"
MODEL_FAMILY="gemma"
DATASET_TYPE="attack"

deepspeed --num_gpus=4 ../finetune/src/finetune_model.py \
    --dataset_type $DATASET_TYPE \
    --model_family $MODEL_FAMILY \
    --model_name $MODEL_NAME