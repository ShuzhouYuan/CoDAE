# CoDAE: Chain-of-Thought Data Augmentation & Evaluation

**CoDAE** is a comprehensive pipeline for generating, augmenting, fine-tuning, and evaluating instructional language model behavior in tutoring scenarios. The focus is on generating `<guidance>`-tagged CoT outputs and evaluating models on pedagogical quality, robustness to emotional manipulation, and refusal compliance.

---

## 🧩 Core Components

### Preprocessing

* **Script:** `preprocess_data.py`
* Loads and splits raw JSONL into length-bounded train/test/val sets.

### Augmentation

* **Primary:** `augment.py`, `prompt.py`, `DataClass.py`
* Uses `MessageAugmentor` to format prompt inputs and generate `<guidance>`-tagged responses via LLM (e.g., Qwen2.5).
* Variants:

  * `augment_idk.py`: Injects "confused user" utterances
  * `augment_attacks.py`: Injects emotional threats
  * `augment_attacks_idk.py`: Mixes both

### Fine-Tuning

* **Script:** `finetune_model.py`
* LoRA fine-tuning on `<guidance>` spans only (via selective masking)

### Evaluation Set Construction

* **Script:** `evaluation_set_constructor.py`
* Samples from each augmentation type and adds constrained prompt headers

### Inference & Merging

* **Script:** `inference_merging.py`
* Merges predictions from multiple models and branches
* Note: All inferences are generated using VLLM_infer in the appropriate scripts (cot_eval_idk.sh, jailbreak_idk.sh, judge_A_idk.sh, etc.)

### Judgement Extraction

* **Script:** `judgement_extraction.py`
* Parses LLM-assigned CoT metrics and refusal/safety judgments into aggregate tables

### Fluency & Diversity

* **Script:** `ppl_bleu.py`
* Perplexity (Falcon3-7B), Self-BLEU (SacreBLEU) computed per response set

### Output Validation

* **Script:** `verify_all_files_follow_rule.py`
* Verifies Markdown outputs follow dialogue formatting rules

### Dataset Statistics

* **Script:** `stats.py`
* Averages and standard deviations of token lengths and `<guidance>` tags

---

## 📊 Metrics Reported

* **Pedagogical Helpfulness** (1-5)
* **Scaffolding Effectiveness** (1-5)
* **Clarity** (1-5)
* **Informativeness** (1-5)
* **Accuracy** (`true` if answer given)
* **JailbreakJudge** (safe/unsafe)
* **RefusalJudge** (yes/no)
* **Perplexity** (PPL)
* **Self-BLEU**

---

## ⚙️ Example Cluster Scripts

If you're in a slurm environment you need to load these modules. Make sure you have lmod or any other hierarchical module naming scheme

```bash
module load release/24.04  GCC/12.3.0  OpenMPI/4.1.5
module load DeepSpeed/0.14.5-CUDA-12.1.1
```

Create a python virtual environment with the given `requirements.txt` file and source it

NOTE: .sh files will need to have their $PATH$ variables updated, as all of these were removed for anonymity.

### Fine-tuning (Gemma on threat-injected data)

```bash
bash finetune_gemma_attack.sh
```

### Generating Chain-of-Thought Responses

```bash
bash cot_eval_attack.sh
```

### Judge Evaluation (Pedagogical CoT)

```bash
bash judge_A_attack.sh
```

### Judge Evaluation (Jailbreak Refusal)

```bash
bash jailbreak_attack.sh
```

---

## 📌 Dependencies

* `transformers`
* `datasets`
* `peft`
* `evaluate`
* `fire`
* `pandas`, `numpy`, `tqdm`

---

## 🧠 Citation

Please cite this work if you use the CoDAE pipeline in your research.
