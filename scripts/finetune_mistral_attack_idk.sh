#!/bin/bash
# Slurm directives
#SBATCH --job-name=finetune_mistral_attack_idk
#SBATCH --output=./logs_mistral_attack_idk/%x_%j.out
#SBATCH --gres=gpu:4
#SBATCH --mail-type=ALL
#SBATCH --mem-per-cpu=24G
#SBATCH --cpus-per-task=16
#SBATCH --time=20:00:00

MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
MODEL_FAMILY="mistral"
DATASET_TYPE="attack_idk"

deepspeed --num_gpus=4 ../finetune/src/finetune_model.py \
    --dataset_type $DATASET_TYPE \
    --model_family $MODEL_FAMILY \
    --model_name $MODEL_NAME