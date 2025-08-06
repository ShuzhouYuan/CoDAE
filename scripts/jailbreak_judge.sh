#!/bin/bash

# Main Experiment Script
echo "Starting Main Experiment Workflow!"

echo "Running judgement for dataset 1 of 50: jailbreakjudge_attack_gemma"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_attack_gemma --save_name "jailbreakjudge_attack_gemma_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_attack_gemma.log 2>&1

echo "Running judgement for dataset 2 of 50: refusaljudge_attack_gemma"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_attack_gemma --save_name "refusaljudge_attack_gemma_judgement" \
> $PATH$/logs/judgement_refusaljudge_attack_gemma.log 2>&1

echo "Running judgement for dataset 3 of 50: jailbreakjudge_attack_llama"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_attack_llama --save_name "jailbreakjudge_attack_llama_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_attack_llama.log 2>&1

echo "Running judgement for dataset 4 of 50: refusaljudge_attack_llama"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_attack_llama --save_name "refusaljudge_attack_llama_judgement" \
> $PATH$/logs/judgement_refusaljudge_attack_llama.log 2>&1

echo "Running judgement for dataset 5 of 50: jailbreakjudge_attack_mistral"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_attack_mistral --save_name "jailbreakjudge_attack_mistral_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_attack_mistral.log 2>&1

echo "Running judgement for dataset 6 of 50: refusaljudge_attack_mistral"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_attack_mistral --save_name "refusaljudge_attack_mistral_judgement" \
> $PATH$/logs/judgement_refusaljudge_attack_mistral.log 2>&1

echo "Running judgement for dataset 7 of 50: jailbreakjudge_attack_qwen"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_attack_qwen --save_name "jailbreakjudge_attack_qwen_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_attack_qwen.log 2>&1

echo "Running judgement for dataset 8 of 50: refusaljudge_attack_qwen"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_attack_qwen --save_name "refusaljudge_attack_qwen_judgement" \
> $PATH$/logs/judgement_refusaljudge_attack_qwen.log 2>&1

echo "Running judgement for dataset 9 of 50: jailbreakjudge_attack_internlm"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_attack_internlm --save_name "jailbreakjudge_attack_internlm_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_attack_internlm.log 2>&1

echo "Running judgement for dataset 10 of 50: refusaljudge_attack_internlm"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_attack_internlm --save_name "refusaljudge_attack_internlm_judgement" \
> $PATH$/logs/judgement_refusaljudge_attack_internlm.log 2>&1

echo "Running judgement for dataset 11 of 50: jailbreakjudge_baseline_gemma"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_baseline_gemma --save_name "jailbreakjudge_baseline_gemma_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_baseline_gemma.log 2>&1

echo "Running judgement for dataset 12 of 50: refusaljudge_baseline_gemma"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_baseline_gemma --save_name "refusaljudge_baseline_gemma_judgement" \
> $PATH$/logs/judgement_refusaljudge_baseline_gemma.log 2>&1

echo "Running judgement for dataset 13 of 50: jailbreakjudge_baseline_llama"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_baseline_llama --save_name "jailbreakjudge_baseline_llama_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_baseline_llama.log 2>&1

echo "Running judgement for dataset 14 of 50: refusaljudge_baseline_llama"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_baseline_llama --save_name "refusaljudge_baseline_llama_judgement" \
> $PATH$/logs/judgement_refusaljudge_baseline_llama.log 2>&1

echo "Running judgement for dataset 15 of 50: jailbreakjudge_baseline_mistral"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_baseline_mistral --save_name "jailbreakjudge_baseline_mistral_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_baseline_mistral.log 2>&1

echo "Running judgement for dataset 16 of 50: refusaljudge_baseline_mistral"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_baseline_mistral --save_name "refusaljudge_baseline_mistral_judgement" \
> $PATH$/logs/judgement_refusaljudge_baseline_mistral.log 2>&1

echo "Running judgement for dataset 17 of 50: jailbreakjudge_baseline_qwen"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_baseline_qwen --save_name "jailbreakjudge_baseline_qwen_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_baseline_qwen.log 2>&1

echo "Running judgement for dataset 18 of 50: refusaljudge_baseline_qwen"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_baseline_qwen --save_name "refusaljudge_baseline_qwen_judgement" \
> $PATH$/logs/judgement_refusaljudge_baseline_qwen.log 2>&1

echo "Running judgement for dataset 19 of 50: jailbreakjudge_baseline_internlm"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_baseline_internlm --save_name "jailbreakjudge_baseline_internlm_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_baseline_internlm.log 2>&1

echo "Running judgement for dataset 20 of 50: refusaljudge_baseline_internlm"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_baseline_internlm --save_name "refusaljudge_baseline_internlm_judgement" \
> $PATH$/logs/judgement_refusaljudge_baseline_internlm.log 2>&1

echo "Running judgement for dataset 21 of 50: jailbreakjudge_idk_gemma"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_idk_gemma --save_name "jailbreakjudge_idk_gemma_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_idk_gemma.log 2>&1

echo "Running judgement for dataset 22 of 50: refusaljudge_idk_gemma"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_idk_gemma --save_name "refusaljudge_idk_gemma_judgement" \
> $PATH$/logs/judgement_refusaljudge_idk_gemma.log 2>&1

echo "Running judgement for dataset 23 of 50: jailbreakjudge_idk_llama"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_idk_llama --save_name "jailbreakjudge_idk_llama_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_idk_llama.log 2>&1

echo "Running judgement for dataset 24 of 50: refusaljudge_idk_llama"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_idk_llama --save_name "refusaljudge_idk_llama_judgement" \
> $PATH$/logs/judgement_refusaljudge_idk_llama.log 2>&1

echo "Running judgement for dataset 25 of 50: jailbreakjudge_idk_mistral"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_idk_mistral --save_name "jailbreakjudge_idk_mistral_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_idk_mistral.log 2>&1

echo "Running judgement for dataset 26 of 50: refusaljudge_idk_mistral"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_idk_mistral --save_name "refusaljudge_idk_mistral_judgement" \
> $PATH$/logs/judgement_refusaljudge_idk_mistral.log 2>&1

echo "Running judgement for dataset 27 of 50: jailbreakjudge_idk_qwen"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_idk_qwen --save_name "jailbreakjudge_idk_qwen_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_idk_qwen.log 2>&1

echo "Running judgement for dataset 28 of 50: refusaljudge_idk_qwen"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_idk_qwen --save_name "refusaljudge_idk_qwen_judgement" \
> $PATH$/logs/judgement_refusaljudge_idk_qwen.log 2>&1

echo "Running judgement for dataset 29 of 50: jailbreakjudge_idk_internlm"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_idk_internlm --save_name "jailbreakjudge_idk_internlm_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_idk_internlm.log 2>&1

echo "Running judgement for dataset 30 of 50: refusaljudge_idk_internlm"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_idk_internlm --save_name "refusaljudge_idk_internlm_judgement" \
> $PATH$/logs/judgement_refusaljudge_idk_internlm.log 2>&1

echo "Running judgement for dataset 31 of 50: jailbreakjudge_normal_gemma"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_normal_gemma --save_name "jailbreakjudge_normal_gemma_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_normal_gemma.log 2>&1

echo "Running judgement for dataset 32 of 50: refusaljudge_normal_gemma"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_normal_gemma --save_name "refusaljudge_normal_gemma_judgement" \
> $PATH$/logs/judgement_refusaljudge_normal_gemma.log 2>&1

echo "Running judgement for dataset 33 of 50: jailbreakjudge_normal_llama"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_normal_llama --save_name "jailbreakjudge_normal_llama_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_normal_llama.log 2>&1

echo "Running judgement for dataset 34 of 50: refusaljudge_normal_llama"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_normal_llama --save_name "refusaljudge_normal_llama_judgement" \
> $PATH$/logs/judgement_refusaljudge_normal_llama.log 2>&1

echo "Running judgement for dataset 35 of 50: jailbreakjudge_normal_mistral"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_normal_mistral --save_name "jailbreakjudge_normal_mistral_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_normal_mistral.log 2>&1

echo "Running judgement for dataset 36 of 50: refusaljudge_normal_mistral"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_normal_mistral --save_name "refusaljudge_normal_mistral_judgement" \
> $PATH$/logs/judgement_refusaljudge_normal_mistral.log 2>&1

echo "Running judgement for dataset 37 of 50: jailbreakjudge_normal_qwen"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_normal_qwen --save_name "jailbreakjudge_normal_qwen_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_normal_qwen.log 2>&1

echo "Running judgement for dataset 38 of 50: refusaljudge_normal_qwen"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_normal_qwen --save_name "refusaljudge_normal_qwen_judgement" \
> $PATH$/logs/judgement_refusaljudge_normal_qwen.log 2>&1

echo "Running judgement for dataset 39 of 50: jailbreakjudge_normal_internlm"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_normal_internlm --save_name "jailbreakjudge_normal_internlm_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_normal_internlm.log 2>&1

echo "Running judgement for dataset 40 of 50: refusaljudge_normal_internlm"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_normal_internlm --save_name "refusaljudge_normal_internlm_judgement" \
> $PATH$/logs/judgement_refusaljudge_normal_internlm.log 2>&1

echo "Running judgement for dataset 41 of 50: jailbreakjudge_attack_idk_gemma"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_attack_idk_gemma --save_name "jailbreakjudge_attack_idk_gemma_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_attack_idk_gemma.log 2>&1

echo "Running judgement for dataset 42 of 50: refusaljudge_attack_idk_gemma"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_attack_idk_gemma --save_name "refusaljudge_attack_idk_gemma_judgement" \
> $PATH$/logs/judgement_refusaljudge_attack_idk_gemma.log 2>&1

echo "Running judgement for dataset 43 of 50: jailbreakjudge_attack_idk_llama"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_attack_idk_llama --save_name "jailbreakjudge_attack_idk_llama_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_attack_idk_llama.log 2>&1

echo "Running judgement for dataset 44 of 50: refusaljudge_attack_idk_llama"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_attack_idk_llama --save_name "refusaljudge_attack_idk_llama_judgement" \
> $PATH$/logs/judgement_refusaljudge_attack_idk_llama.log 2>&1

echo "Running judgement for dataset 45 of 50: jailbreakjudge_attack_idk_mistral"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_attack_idk_mistral --save_name "jailbreakjudge_attack_idk_mistral_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_attack_idk_mistral.log 2>&1

echo "Running judgement for dataset 46 of 50: refusaljudge_attack_idk_mistral"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_attack_idk_mistral --save_name "refusaljudge_attack_idk_mistral_judgement" \
> $PATH$/logs/judgement_refusaljudge_attack_idk_mistral.log 2>&1

echo "Running judgement for dataset 47 of 50: jailbreakjudge_attack_idk_qwen"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_attack_idk_qwen --save_name "jailbreakjudge_attack_idk_qwen_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_attack_idk_qwen.log 2>&1

echo "Running judgement for dataset 48 of 50: refusaljudge_attack_idk_qwen"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_attack_idk_qwen --save_name "refusaljudge_attack_idk_qwen_judgement" \
> $PATH$/logs/judgement_refusaljudge_attack_idk_qwen.log 2>&1

echo "Running judgement for dataset 49 of 50: jailbreakjudge_attack_idk_internlm"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset jailbreakjudge_attack_idk_internlm --save_name "jailbreakjudge_attack_idk_internlm_judgement" \
> $PATH$/logs/judgement_jailbreakjudge_attack_idk_internlm.log 2>&1

echo "Running judgement for dataset 50 of 50: refusaljudge_attack_idk_internlm"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.3-70B-Instruct/" --save_path "$PATH$/inferences/jailbreak_judgements/" --temperature 0 --generation-config vllm --dataset refusaljudge_attack_idk_internlm --save_name "refusaljudge_attack_idk_internlm_judgement" \
> $PATH$/logs/judgement_refusaljudge_attack_idk_internlm.log 2>&1


echo "Main Experiment Workflow Completed!"
