import pandas as pd
from itertools import product
import json
from collections import defaultdict
from tqdm import tqdm

ugly_mapping = {
    'baseline_llama': 'Llama3.1',
    'baseline_llama_con': 'Llama3.1 +C',
    'normal_llama': 'Llama3.1 FT',
    'normal_llama_con': 'Llama3.1 FT +C',
    'idk_llama': 'Llama3.1 FT +idk',
    'idk_llama_con': 'Llama3.1 FT +idk +C',
    'attack_llama': 'Llama3.1 FT threat',
    'attack_llama_con': 'Llama3.1 FT threat +C',
    'attack_idk_llama': 'Llama3.1 FT idk+threat',
    'attack_idk_llama_con': 'Llama3.1 FT idk+threat +C',
    'baseline_mistral': 'Mistral',
    'baseline_mistral_con': 'Mistral +C',
    'normal_mistral': 'Mistral FT',
    'normal_mistral_con': 'Mistral FT +C',
    'idk_mistral': 'Mistral FT +idk',
    'idk_mistral_con': 'Mistral FT +idk +C',
    'attack_mistral': 'Mistral FT threat',
    'attack_mistral_con': 'Mistral FT threat +C',
    'attack_idk_mistral': 'Mistral FT idk+threat',
    'attack_idk_mistral_con': 'Mistral FT idk+threat +C',
    'baseline_qwen': 'Qwen2.5',
    'baseline_qwen_con': 'Qwen2.5 +C',
    'normal_qwen': 'Qwen2.5 FT',
    'normal_qwen_con': 'Qwen2.5 FT +C',
    'idk_qwen': 'Qwen2.5 FT +idk',
    'idk_qwen_con': 'Qwen2.5 FT +idk +C',
    'attack_qwen': 'Qwen2.5 FT threat',
    'attack_qwen_con': 'Qwen2.5 FT threat +C',
    'attack_idk_qwen': 'Qwen2.5 FT idk+threat',
    'attack_idk_qwen_con': 'Qwen2.5 FT idk+threat +C',
    'baseline_internlm': 'InternLM',
    'baseline_internlm_con': 'InternLM +C',
    'normal_internlm': 'InternLM FT',
    'normal_internlm_con': 'InternLM FT +C',
    'idk_internlm': 'InternLM FT +idk',
    'idk_internlm_con': 'InternLM FT +idk +C',
    'attack_internlm': 'InternLM FT threat',
    'attack_internlm_con': 'InternLM FT threat +C',
    'attack_idk_internlm': 'InternLM FT idk+threat',
    'attack_idk_internlm_con': 'InternLM FT idk+threat +C',
    'baseline_gemma': 'Gemma2',
    'baseline_gemma_con': 'Gemma2 +C',
    'normal_gemma': 'Gemma2 FT',
    'normal_gemma_con': 'Gemma2 FT +C',
    'idk_gemma': 'Gemma2 FT +idk',
    'idk_gemma_con': 'Gemma2 FT +idk +C',
    'attack_gemma': 'Gemma2 FT threat',
    'attack_gemma_con': 'Gemma2 FT threat +C',
    'attack_idk_gemma': 'Gemma2 FT idk+threat',
    'attack_idk_gemma_con': 'Gemma2 FT idk+threat +C',
}


def build_results_df(branch):
    models = ['llama', 'mistral', 'qwen', 'gemma', 'internlm']
    subsets = ['', '_con']

    experiments = [f"{model}_cot-A{subset}" for model, subset in product(models, subsets)]

    results = defaultdict(pd.Series)
    error_counter = 0
    errors = []
    for experiment in tqdm(experiments):

        with open(f"./inferences/judgement/{branch}_{experiment}", 'r', encoding='utf-8') as f:
            # create nested dictionary from predictions entry in jasonlines formatted file
            file = f.readlines()

        data_dict = defaultdict(dict)
        for i in range(len(file)):
            data_dict[i] = json.loads(file[i])
        # Convert predictions to DataFrame
        
        data_series_list = []
        for item in data_dict.values():
            item = item['predict'].removeprefix("```").removeprefix("json").removesuffix("```")
            try:
                series = json.loads(item)
            except json.JSONDecodeError:
                error_counter += 1
                error = f"JSONDecodeError for experiment {experiment}, item: {item}"
                errors.append(error)
            series = pd.Series(series)
            data_series_list.append(series)
        # Create DataFrame from the list of Series
        df = pd.DataFrame(data_series_list)
        # Convert 'accuracy' column to integers
        try:
            df['accuracy'] = df['accuracy'].map(lambda x: int(x == 'true'))  # Convert True/False strings to integers
        except KeyError:
            pass
        # get means for each column
        try:
            means = df.describe().loc['mean']
        except KeyError:
            print(f"KeyError for experiment {experiment}, skipping...")
        # get std for each column
        try:
            stds = df.describe().loc['std']
        except KeyError:
            print(f"KeyError for experiment {experiment}, skipping...")
        match = experiment.split('_')
        if len(match) == 2:
            model = match[0]
        elif len(match) == 3:
            model = match[0] + '_' + match[2]
        model = branch + '_' + model
        lines = pd.Series([f"{means[col]:.2f} ± {stds[col]:.2f}" for col in means.index], index=means.index, name=model)
    
        if len(results[model]) == 0:
            results[model] = pd.Series(lines, name=model)
        else:
            results[model] = pd.concat([results[model], lines], axis=0)

    with open('errors.txt', 'w', encoding='utf-8') as f:
        for error in errors:
            f.write("-" * 80 + '\n\n')
            f.write(error + '\n\n')
    # Convert results to DataFrame
    results_df = pd.DataFrame.from_dict(results).T
    return results_df

def scale_and_reorder(results_df):

    new_colums = ['perplexity', 'self-Bleu']
    # extend results_df with new columns
    for column in new_colums:
        results_df[column] = 0

    reordered_columns = ['accuracy', 'clarity', 'informativeness', 'pedagogical_helpfulness', 'scaffolding_effectiveness']

    results_df = results_df[reordered_columns]
    return results_df

def main():
    branches = ['baseline', 'normal', 'idk', 'attack', 'attack_idk']

    for branch in branches:
        results = build_results_df(branch)
        results = scale_and_reorder(results)
        results.to_json(f'postprocessing/{branch}_A_results.json', orient='index', indent=2)

    branches = ['baseline', 'normal', 'idk', 'attack', 'attack_idk']

    branched_results = defaultdict(dict)

    for branch in branches:
        # open two files and read them into dataframes: {branch}_A_results.json and {branch}_ppl_bleu.json
        results_df = pd.read_json(f'postprocessing/{branch}_A_results.json', orient='index')
        ppl_bleu_df = pd.read_json(f'postprocessing/{branch}_ppl_bleu.json', orient='index')
        # merge the two dataframes on the index
        merged_df = pd.merge(results_df, ppl_bleu_df, left_index=True, right_index=True)
        branched_results[branch] = merged_df

    models = ['llama', 'mistral', 'qwen', 'internlm', 'gemma']
    # regroup the results by model and subset
    model_results = defaultdict(dict)
    for branch, df in branched_results.items():

        for model in models:
            #print(model)
            # add only rows where the index includes the model name
            mod = df.index[df.index.str.contains(model)]
            results = model_results.get(model, pd.DataFrame())
            model_results[model] = pd.concat([results, df.loc[mod]], axis=0)

    lines = ['\\hline\n']
    for model in models:
        #print(model)
        for line in model_results[model].iterrows():
            # map string to format: "model + task + con & perplexity & self-Bleu & accuracy & clarity & informativeness & pedagogical_helpfulness & scaffolding_effectiveness"
            index = line[0]
            values = line[1]
            
            perplexity = values['perplexity']
            self_bleu = values['self_bleu']
            accuracy = values['accuracy']
            clarity = values['clarity']
            informativeness = values['informativeness']
            pedagogical_helpfulness = values['pedagogical_helpfulness']
            scaffolding_effectiveness = values['scaffolding_effectiveness']
            try:
                line = str(f"{ugly_mapping[index]} & {perplexity:.2f} & {self_bleu:.2f} & {accuracy} & {clarity} & {informativeness} & {pedagogical_helpfulness} & {scaffolding_effectiveness} \\\\")
            except NameError:
                line = str(f"{ugly_mapping[index]} & {perplexity:.2f} & {self_bleu:.2f} & {accuracy} & {clarity} & {informativeness} & {pedagogical_helpfulness} & {scaffolding_effectiveness} \\\\")
            lines.append(line)
            lines.append('\n')
        lines.append('\\hline\n')

    with open('latex_results.txt', 'w', encoding='utf-8') as file:
        file.writelines(lines)
    with open('pd_results.txt', 'w', encoding='utf-8') as file:
        header = 'Model & Perplexity & Self-BLEU & Accuracy & Clarity & Informativeness & Pedagogical Helpfulness & Scaffolding Effectiveness\n'
        file.write(header)
        for line in lines:
            if line == '\\hline\n':
                continue
            else:
                file.write(line.strip('\\\\'))

    # read in latex_formatted_results.txt into pandas dataframe, first line is the header, separator is '&'
    results_df = pd.read_csv('pd_results.txt', sep=' & ', header=0, engine='python', index_col=0)

    columns = results_df.columns.tolist()
    columns.append('jailbreakjudge')
    columns.append('refusaljudge')
    results_df = results_df.reindex(columns=columns)

    path = 'jailbreak_judgement'
    models = ['gemma', 'internlm', 'llama', 'mistral', 'qwen']
    tasks = ['jailbreakjudge', 'refusaljudge']
    branches = ['baseline', 'normal', 'idk', 'attack', 'attack_idk']

    for task, branch, model in product(tasks, branches, models):
        experiment = f"{path}/{task}_{branch}_{model}_judgement"

        with open(experiment, 'r', encoding='utf-8') as f:
                # create nested dictionary from predictions entry in jasonlines formatted file
            results = pd.read_json(f, orient='records', lines=True)
        results['predict'] = results['predict'].map(lambda x: x.strip().removesuffix('.').lower() if isinstance(x, str) else x)
        # map safe and unsafe to 1 and 0 in predict
        if task == 'jailbreakjudge':
            results['predict'] = results['predict'].map(lambda x: 1 if x == 'safe' else 0)
        elif task == 'refusaljudge':
            results['predict'] = results['predict'].map(lambda x: 1 if x == 'yes' else 0)
        else:
            raise ValueError(f"Unknown task: {task}")
        
        # score for task = mean of predict
        score = results['predict'].mean()
        print(f"Score for {experiment}: {score}")
        # update results_df with the score
        results_df.loc[ugly_mapping[f"{branch}_{model}"], task] = score
        
    lines = ['\\hline\n']

    for line in results_df.iterrows():
        # map string to format: "model + task + con & perplexity & self-Bleu & accuracy & clarity & informativeness & pedagogical_helpfulness & scaffolding_effectiveness"
        index = line[0]
        values = line[1]

        perplexity = values['Perplexity']
        self_bleu = values['Self-BLEU']
        accuracy = values['Accuracy']
        clarity = values['Clarity']
        informativeness = values['Informativeness']
        pedagogical_helpfulness = values['Pedagogical Helpfulness']
        scaffolding_effectiveness = values['Scaffolding Effectiveness']
        jailbreak_judgement = values['jailbreakjudge']
        refusal_judgement = values['refusaljudge']
        if jailbreak_judgement != 'nan':
            line = str(f"{index} & {perplexity} & {self_bleu} & {accuracy} & {clarity} & {informativeness} & {pedagogical_helpfulness} & {scaffolding_effectiveness} & {jailbreak_judgement:.2f} & {refusal_judgement:.2f} \\\\")
        else:
            line = str(f"{index} & {perplexity} & {self_bleu} & {accuracy} & {clarity} & {informativeness} & {pedagogical_helpfulness} & {scaffolding_effectiveness} &  &  \\\\")
        lines.append(line)
        lines.append('\n')
    lines.append('\\hline\n')

    with open('latex_results.txt', 'w', encoding='utf-8') as file:
        file.writelines(lines)
    with open('pd_results.txt', 'w', encoding='utf-8') as file:
        header = 'Model & Perplexity & Self-BLEU & Accuracy & Clarity & Informativeness & Pedagogical Helpfulness & Scaffolding Effectiveness\n'
        file.write(header)
        for line in lines:
            if line == '\\hline\n':
                continue
            else:
                file.write(line.strip('\\\\'))


if __name__ == "__main__":
    main()