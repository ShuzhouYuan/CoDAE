#!/bin/bash
# Slurm directives
#SBATCH --job-name=finetune_qwen_idk
#SBATCH --output=./logs_qwen_new/%x_%j.out
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=24G
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
MODEL_FAMILY="qwen"
DATASET_TYPE="idk"

deepspeed --num_gpus=4 ../finetune/src/finetune_model.py \
    --dataset_type $DATASET_TYPE \
    --model_family $MODEL_FAMILY \
    --model_name $MODEL_NAME