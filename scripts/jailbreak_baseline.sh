#!/bin/bash

# Main Experiment Script
echo "Starting Main Experiment Workflow!"

echo "Begin inference llama_baseline..."
python3 scripts/vllm_infer.py --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" --save_path "$PATH$/inferences/jailbreak_logs/" --template llama3 --temperature 0 --generation-config vllm --dataset jailbreak --save_name "jailbreak_baseline_llama" \
> $PATH$/logs/jailbreak_baseline_llama.log 2>&1

echo "Begin inference mistral_baseline..."
python3 scripts/vllm_infer.py --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" --save_path "$PATH$/inferences/jailbreak_logs/" --template mistral --temperature 0 --generation-config vllm --dataset jailbreak --save_name "jailbreak_baseline_mistral" \
> $PATH$/logs/jailbreak_baseline_mistral.log 2>&1

echo "Begin inference qwen_baseline..."
python3 scripts/vllm_infer.py --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" --save_path "$PATH$/inferences/jailbreak_logs/" --template qwen --temperature 0 --generation-config vllm --dataset jailbreak --save_name "jailbreak_baseline_qwen" \
> $PATH$/logs/jailbreak_baseline_qwen.log 2>&1

echo "Begin inference internlm_baseline..."
python3 scripts/vllm_infer.py --model_name_or_path "internlm/internlm3-8b-instruct" --save_path "$PATH$/inferences/jailbreak_logs/" --temperature 0 --generation-config vllm --dataset jailbreak --save_name "jailbreak_baseline_internlm" \
> $PATH$/logs/jailbreak_baseline_internlm.log 2>&1

echo "Begin inference gemma_baseline..."
python3 scripts/vllm_infer.py --model_name_or_path "google/gemma-2-9b-it" --save_path "$PATH$/inferences/jailbreak_logs/" --template gemma --temperature 0 --generation-config vllm --dataset jailbreak --save_name "jailbreak_baseline_gemma" \
> $PATH$/logs/jailbreak_baseline_gemma.log 2>&1

echo "Main Experiment Workflow Completed!"
