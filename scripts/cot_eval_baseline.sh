#!/bin/bash

# Main Experiment Script
echo "Starting Main Experiment Workflow!"

echo "Begin inference baseline llama A..."
python3 scripts/vllm_infer.py --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" --save_path "$PATH$/inferences/inference_baseline/" --template llama3 --temperature 0 --generation-config vllm --dataset evaluation.jsonl --save_name "baseline_llama_cot-A" \
> $PATH$/logs/inference_baseline_llama_A.log 2>&1

echo "Begin inference baseline llama A con..."
python3 scripts/vllm_infer.py --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" --save_path "$PATH$/inferences/inference_baseline/" --template llama3 --temperature 0 --generation-config vllm --dataset evaluation_constrained.jsonl --save_name "baseline_llama_cot-A_con" \
> $PATH$/logs/inference_baseline_llama_A_con.log 2>&1

echo "Begin inference baseline mistral A..."
python3 scripts/vllm_infer.py --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" --save_path "$PATH$/inferences/inference_baseline/" --template mistral --temperature 0 --generation-config vllm --dataset evaluation.jsonl --save_name "baseline_mistral_cot-A" \
> $PATH$/logs/inference_baseline_mistral_A.log 2>&1

echo "Begin inference baseline mistral A Con..."
python3 scripts/vllm_infer.py --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" --save_path "$PATH$/inferences/inference_baseline/" --template mistral --temperature 0 --generation-config vllm --dataset evaluation_constrained.jsonl --save_name "baseline_mistral_cot-A_con" \
> $PATH$/logs/inference_baseline_mistral_A_con.log 2>&1

echo "Begin inference baseline qwen A..."
python3 scripts/vllm_infer.py --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" --save_path "$PATH$/inferences/inference_baseline/" --template qwen --temperature 0 --generation-config vllm --dataset evaluation.jsonl --save_name "baseline_qwen_cot-A" \
> $PATH$/logs/inference_baseline_qwen_A.log 2>&1

echo "Begin inference baseline qwen A con..."
python3 scripts/vllm_infer.py --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" --save_path "$PATH$/inferences/inference_baseline/" --template qwen --temperature 0 --generation-config vllm --dataset evaluation_constrained.jsonl --save_name "baseline_qwen_cot-A_con" \
> $PATH$/logs/inference_baseline_qwen_A_con.log 2>&1

echo "Begin inference baseline internlm A..."
python3 scripts/vllm_infer.py --model_name_or_path "internlm/internlm3-8b-instruct" --save_path "$PATH$/inferences/inference_baseline/" --temperature 0 --generation-config vllm --dataset evaluation.jsonl --save_name "baseline_internlm_cot-A" --trust-remote-code "True" \
> $PATH$/logs/inference_baseline_internlm_A.log 2>&1

echo "Begin inference baseline internlm A con..."
python3 scripts/vllm_infer.py --model_name_or_path "internlm/internlm3-8b-instruct" --save_path "$PATH$/inferences/inference_baseline/" --temperature 0 --generation-config vllm --dataset evaluation_constrained.jsonl --save_name "baseline_internlm_cot-A_con" --trust-remote-code "True" \
> $PATH$/logs/inference_baseline_internlm_A_con.log 2>&1

echo "Begin inference baseline gemma A..."
python3 scripts/vllm_infer.py --model_name_or_path "google/gemma-2-9b-it" --save_path "$PATH$/inferences/inference_baseline/" --template gemma --temperature 0 --generation-config vllm --dataset evaluation.jsonl --save_name "baseline_gemma_cot-A" \
> $PATH$/logs/inference_baseline_gemma_A.log 2>&1

echo "Begin inference baseline gemma A con..."
python3 scripts/vllm_infer.py --model_name_or_path "google/gemma-2-9b-it" --save_path "$PATH$/inferences/inference_baseline/" --template gemma --temperature 0 --generation-config vllm --dataset evaluation_constrained.jsonl --save_name "baseline_gemma_cot-A_con" \
> $PATH$/logs/inference_baseline_gemma_A_con.log 2>&1

echo "Main Experiment Workflow Completed!"
