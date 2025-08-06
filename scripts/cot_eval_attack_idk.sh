#!/bin/bash

# Main Experiment Script
echo "Starting Main Experiment Workflow!"

echo "Begin inference attack_idk llama A..."
python3 scripts/vllm_infer.py --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" --adapter_name_or_path "$REPO$/final_llama_attack_idk" --save_path "$PATH$/inferences/inference_attack_idk/" --template llama3 --temperature 0 --generation-config vllm --dataset evaluation.jsonl --save_name "attack_idk_llama_cot-A" \
> $PATH$/logs/inference_attack_idk_llama_A.log 2>&1

echo "Begin inference attack_idk llama A con..."
python3 scripts/vllm_infer.py --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" --adapter_name_or_path "$REPO$/final_llama_attack_idk" --save_path "$PATH$/inferences/inference_attack_idk/" --template llama3 --temperature 0 --generation-config vllm --dataset evaluation_constrained.jsonl --save_name "attack_idk_llama_cot-A_con" \
> $PATH$/logs/inference_attack_idk_llama_A_con.log 2>&1

echo "Begin inference attack_idk mistral A..."
python3 scripts/vllm_infer.py --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" --adapter_name_or_path "$REPO$/final_mistral_attack_idk" --save_path "$PATH$/inferences/inference_attack_idk/" --template mistral --temperature 0 --generation-config vllm --dataset evaluation.jsonl --save_name "attack_idk_mistral_cot-A" \
> $PATH$/logs/inference_attack_idk_mistral_A.log 2>&1

echo "Begin inference attack_idk mistral A Con..."
python3 scripts/vllm_infer.py --model_name_or_path "mistralai/Mistral-7B-Instruct-v0.3" --adapter_name_or_path "$REPO$/final_mistral_attack_idk" --save_path "$PATH$/inferences/inference_attack_idk/" --template mistral --temperature 0 --generation-config vllm --dataset evaluation_constrained.jsonl --save_name "attack_idk_mistral_cot-A_con" \
> $PATH$/logs/inference_attack_idk_mistral_A_con.log 2>&1

echo "Begin inference attack_idk qwen A..."
python3 scripts/vllm_infer.py --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" --adapter_name_or_path "$REPO$/final_qwen_attack_idk" --save_path "$PATH$/inferences/inference_attack_idk/" --template qwen --temperature 0 --generation-config vllm --dataset evaluation.jsonl --save_name "attack_idk_qwen_cot-A" \
> $PATH$/logs/inference_attack_idk_qwen_A.log 2>&1

echo "Begin inference attack_idk qwen A con..."
python3 scripts/vllm_infer.py --model_name_or_path "Qwen/Qwen2.5-7B-Instruct" --adapter_name_or_path "$REPO$/final_qwen_attack_idk" --save_path "$PATH$/inferences/inference_attack_idk/" --template qwen --temperature 0 --generation-config vllm --dataset evaluation_constrained.jsonl --save_name "attack_idk_qwen_cot-A_con" \
> $PATH$/logs/inference_attack_idk_qwen_A_con.log 2>&1

echo "Begin inference attack_idk internlm A..."
python3 scripts/vllm_infer.py --model_name_or_path "internlm/internlm3-8b-instruct" --adapter_name_or_path "$REPO$/final_internlm_attack_idk" --save_path "$PATH$/inferences/inference_attack_idk/" --temperature 0 --generation-config vllm --dataset evaluation.jsonl --save_name "attack_idk_internlm_cot-A" --trust-remote-code "True" \
> $PATH$/logs/inference_attack_idk_internlm_A.log 2>&1

echo "Begin inference attack_idk internlm A con..."
python3 scripts/vllm_infer.py --model_name_or_path "internlm/internlm3-8b-instruct" --adapter_name_or_path "$REPO$/final_internlm_attack_idk" --save_path "$PATH$/inferences/inference_attack_idk/" --temperature 0 --generation-config vllm --dataset evaluation_constrained.jsonl --save_name "attack_idk_internlm_cot-A_con" --trust-remote-code "True" \
> $PATH$/logs/inference_attack_idk_internlm_A_con.log 2>&1

echo "Begin inference attack_idk gemma A..."
python3 scripts/vllm_infer.py --model_name_or_path "google/gemma-2-9b-it" --adapter_name_or_path "$REPO$/final_gemma_attack_idk" --save_path "$PATH$/inferences/inference_attack_idk/" --template gemma --temperature 0 --generation-config vllm --dataset evaluation.jsonl --save_name "attack_idk_gemma_cot-A" \
> $PATH$/logs/inference_attack_idk_gemma_A.log 2>&1

echo "Begin inference attack_idk gemma A con..."
python3 scripts/vllm_infer.py --model_name_or_path "google/gemma-2-9b-it" --adapter_name_or_path "$REPO$/final_gemma_attack_idk" --save_path "$PATH$/inferences/inference_attack_idk/" --template gemma --temperature 0 --generation-config vllm --dataset evaluation_constrained.jsonl --save_name "attack_idk_gemma_cot-A_con" \
> $PATH$/logs/inference_attack_idk_gemma_A_con.log 2>&1

echo "Main Experiment Workflow Completed!"
