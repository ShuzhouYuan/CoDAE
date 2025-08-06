#!/bin/bash

# Main Experiment Script
echo "Starting Main Experiment Workflow!"

echo "Begin judgement llama..."
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct" --save_path "$PATH$/inferences/judgement/" --template llama3 --temperature 0 --generation-config vllm --dataset attack_idk_A_llama --save_name "attack_idk_llama_cot-A" \
> $PATH$/logs/judgement_A_attack_idk.log 2>&1

echo "Begin judgement llama con..."
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct" --save_path "$PATH$/inferences/judgement/" --template llama3 --temperature 0 --generation-config vllm --dataset attack_idk_A_llama_con --save_name "attack_idk_llama_cot-A_con" \
> $PATH$/logs/judgement_A_attack_idk.log 2>&1

echo "Begin judgement mistral..."
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct" --save_path "$PATH$/inferences/judgement/" --template llama3 --temperature 0 --generation-config vllm --dataset attack_idk_A_mistral --save_name "attack_idk_mistral_cot-A" \
> $PATH$/logs/judgement_A_attack_idk.log 2>&1

echo "Begin judgement mistral con..."
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct" --save_path "$PATH$/inferences/judgement/" --template llama3 --temperature 0 --generation-config vllm --dataset attack_idk_A_mistral_con --save_name "attack_idk_mistral_cot-A_con" \
> $PATH$/logs/judgement_A_attack_idk.log 2>&1

echo "Begin judgement qwen..."
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct" --save_path "$PATH$/inferences/judgement/" --template llama3 --temperature 0 --generation-config vllm --dataset attack_idk_A_qwen --save_name "attack_idk_qwen_cot-A" \
> $PATH$/logs/judgement_A_attack_idk.log 2>&1

echo "Begin judgement qwen con..."
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct" --save_path "$PATH$/inferences/judgement/" --template llama3 --temperature 0 --generation-config vllm --dataset attack_idk_A_qwen_con --save_name "attack_idk_qwen_cot-A_con" \
> $PATH$/logs/judgement_A_attack_idk.log 2>&1

echo "Begin judgement internlm..."
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct" --save_path "$PATH$/inferences/judgement/" --template llama3 --temperature 0 --generation-config vllm --dataset attack_idk_A_internlm --save_name "attack_idk_internlm_cot-A" \
> $PATH$/logs/judgement_A_attack_idk.log 2>&1

echo "Begin judgement internlm con..."
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct" --save_path "$PATH$/inferences/judgement/" --template llama3 --temperature 0 --generation-config vllm --dataset attack_idk_A_internlm_con --save_name "attack_idk_internlm_cot-A_con" \
> $PATH$/logs/judgement_A_attack_idk.log 2>&1

echo "Begin judgement gemma..."
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct" --save_path "$PATH$/inferences/judgement/" --template llama3 --temperature 0 --generation-config vllm --dataset attack_idk_A_gemma --save_name "attack_idk_gemma_cot-A" \
> $PATH$/logs/judgement_A_attack_idk.log 2>&1

echo "Begin judgement gemma con..."
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct" --save_path "$PATH$/inferences/judgement/" --template llama3 --temperature 0 --generation-config vllm --dataset attack_idk_A_gemma_con --save_name "attack_idk_gemma_cot-A_con" \
> $PATH$/logs/judgement_A_attack_idk.log 2>&1

echo "Main Experiment Workflow Completed!"
