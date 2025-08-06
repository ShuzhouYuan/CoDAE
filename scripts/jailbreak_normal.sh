#!/bin/bash

# Main Experiment Script
echo "Starting Main Experiment Workflow!"

echo "Begin inference llama_normal..."
python3 scripts/vllm_infer.py --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" --adapter_name_or_path "$REPO$/final_llama_normal" --save_path "$PATH$/inferences/jailbreak_logs/" --template llama3 --temperature 0 --generation-config vllm --dataset jailbreak --save_name "jailbreak_normal_llama" \
> $PATH$/logs/jailbreak_normal_llama.log 2>&1

echo "Begin inference mistral_normal..."
python3 scripts/vllm_infer.py --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" --adapter_name_or_path "$REPO$/final_mistral_normal" --save_path "$PATH$/inferences/jailbreak_logs/" --template mistral --temperature 0 --generation-config vllm --dataset jailbreak --save_name "jailbreak_normal_mistral" \
> $PATH$/logs/jailbreak_normal_mistral.log 2>&1

echo "Begin inference qwen_normal..."
python3 scripts/vllm_infer.py --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" --adapter_name_or_path "$REPO$/final_qwen_normal" --save_path "$PATH$/inferences/jailbreak_logs/" --template qwen --temperature 0 --generation-config vllm --dataset jailbreak --save_name "jailbreak_normal_qwen" \
> $PATH$/logs/jailbreak_normal_qwen.log 2>&1

echo "Begin inference internlm_normal..."
python3 scripts/vllm_infer.py --model_name_or_path "internlm/internlm3-8b-instruct" --adapter_name_or_path "$REPO$/final_internlm_normal" --save_path "$PATH$/inferences/jailbreak_logs/" --temperature 0 --generation-config vllm --dataset jailbreak --save_name "jailbreak_normal_internlm" \
> $PATH$/logs/jailbreak_normal_internlm.log 2>&1

echo "Begin inference gemma_normal..."
python3 scripts/vllm_infer.py --model_name_or_path "google/gemma-2-9b-it" --adapter_name_or_path "$REPO$/final_gemma_normal" --save_path "$PATH$/inferences/jailbreak_logs/" --template gemma --temperature 0 --generation-config vllm --dataset jailbreak --save_name "jailbreak_normal_gemma" \
> $PATH$/logs/jailbreak_normal_gemma.log 2>&1

echo "Main Experiment Workflow Completed!"
