"""
Evaluation pipeline for the augmented dataset:

1) extract <guidance> from the individual augemented data output files
-> for file in folders: regex_extract_guidance 

2) format into alpaca dataset format including LLM as judge system prompt
-> 

3) read LLM judge json output
4) format and visualize the LLM judge output
"""
import fire
import pandas as pd
from evaluate import load
import torch
import numpy as np
import copy
from tqdm import tqdm
from collections import defaultdict
from itertools import product


def calculate_self_bleu(data_in, sacrebleu) -> float:
    print("Calculating self-BLEU scores...")
    bleu_scores = []

    for sentence in tqdm(data_in):
        sentences_copy = copy.deepcopy(data_in)
        remaining_sentences = sentences_copy.remove(sentence) # this line is required to avoid modifying the original list
        bleu = sacrebleu.compute(predictions=[sentence], references=[sentences_copy], use_effective_order=True, smooth_method='none')
        bleu_scores.append(bleu['score'])

    return np.mean(bleu_scores)


def postprocessing(branch:str):

    output_data = defaultdict(dict)

    models = ['llama', 'mistral', 'qwen', 'internlm', 'gemma']
    subsets = ['', '_con']

    data_path = f"inferences/inference_{branch}"

    perplexity = load("perplexity", module_type="metric")
    sacrebleu = load("sacrebleu")
    
    for model, subset in product(models, subsets):
        experiment = f"{branch}_{model}_cot-A{subset}"
        model_name = f"{branch}_{model}{subset}"
        print(f"Processing {experiment}...")
        with open(f'{data_path}/{experiment}', 'r', encoding='utf-8') as file:
            raw_data = pd.read_json(file, lines=True)
        data = [data_point for data_point in raw_data['predict'].tolist() if len(data_point) > 3]
        with torch.no_grad():
            perplexity_results = perplexity.compute(predictions=data,
                                                model_id="tiiuae/Falcon3-7B-Base",
                                                add_start_token=False,
                                                batch_size=8)
        
        ppl = round(perplexity_results['mean_perplexity'], 2)
        ppl_std = round(np.std(perplexity_results['perplexities']), 2)
        #perplexity = calculate_perplexity(data)
        self_bleu = calculate_self_bleu(data, sacrebleu)
        # write perplexity and self_bleu to file as json stored by experiment name
        output_data[model_name] = {
                                    'perplexity': ppl,
                                    'perplexity_std': ppl_std,
                                    'self_bleu': self_bleu
                                    }
        print(f"Processed {model_name} - perplexity: {ppl}±{ppl_std} , self-BLEU: {self_bleu}")
        print("-" * 80, "\n\n")
    print("All experiments processed. Writing output...")
    # write output_data to file as json
    output_df = pd.DataFrame.from_dict(output_data).T
    output_df.to_json(f'postprocessing/{branch}_ppl_bleu.json', orient='index', indent=2)
    print(f"Output written to file. At postprocessing/{branch}_ppl_bleu.json")

if __name__ == "__main__":
    fire.Fire(postprocessing)
