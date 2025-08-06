import os
import yaml
import logging
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from torch.utils.data import Dataset
from peft import LoraConfig, get_peft_model
import torch
import torch.nn.functional as F
from typing import Dict, List
import argparse

logger = logging.getLogger(__name__)

class DialogueDataset(Dataset):
    """
    Each JSON line must look like:
    {
      "input": "<system prompt>\n<user>: …\n<assistant>:",
      "output": "<guidance> … </guidance>"
    }
    i.e. `input` ends with exactly "<assistant>:" and `output` is exactly one guidance block,
    beginning with "<guidance>" and ending with "</guidance>".
    """
    def __init__(self, file_path: str, tokenizer: AutoTokenizer, max_length: int = 3072):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    # We expect both keys to exist
                    if "input" in item and "output" in item:
                        # Basic sanity check: output must start with <guidance> and end with </guidance>
                        if not (item["output"].startswith("<guidance>")
                                and item["output"].endswith("</guidance>")):
                            raise ValueError(
                                f"Line does not have exactly one <guidance>…</guidance> block: {item['output']}"
                            )
                        # Likewise, input must end with "<assistant>:"
                        if not item["input"].strip().endswith("<assistant>:"):
                            raise ValueError(
                                f"Input must end with '<assistant>:'. Got: {item['input']}"
                            )
                        self.data.append(item)
                except Exception as e:
                    # Skip any malformed line
                    logger.warning(f"Skipping malformed line: {e}")
                    continue

        logging.info(f"Loaded {len(self.data)} examples from {file_path}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        input_text = item["input"].strip()     # ends with "<assistant>:"
        output_text = item["output"].strip()   # exactly "<guidance> … </guidance>"

        # 1. Concatenate them into one long string. We want the model's first new token to be <guidance>.
        full_text = input_text + " " + output_text

        # 2. Tokenize the entire sequence
        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)          # (seq_len,)
        attention_mask = encoding["attention_mask"].squeeze(0)

        # 3. Find the token‐indices of <guidance> and </guidance>
        guidance_start_id = self.tokenizer.convert_tokens_to_ids("<guidance>")
        guidance_end_id   = self.tokenizer.convert_tokens_to_ids("</guidance>")

        all_ids = input_ids.tolist()
        try:
            start_idx = all_ids.index(guidance_start_id)
            end_idx   = all_ids.index(guidance_end_id)
        except ValueError:
            # If for some reason we can't find them, something is deeply wrong
            raise RuntimeError(
                f"Could not locate <guidance> or </guidance> tokens in the tokenized sequence."
            )

        # 4. Build labels so that only tokens between start_idx..end_idx (inclusive) are predicted.
        labels = torch.full_like(input_ids, -100)  # initialize all to -100
        labels[start_idx : end_idx + 1] = input_ids[start_idx : end_idx + 1]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def custom_collate_fn(batch: List[Dict], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    """
    Pad input_ids / attention_mask / labels to the max length in the batch.
    """
    input_ids_list = [ex["input_ids"] for ex in batch]
    attention_list = [ex["attention_mask"] for ex in batch]
    label_list     = [ex["labels"] for ex in batch]

    max_len = max(x.size(0) for x in input_ids_list)

    padded_input_ids = []
    padded_attention = []
    padded_labels    = []

    for i in range(len(batch)):
        seq_len = input_ids_list[i].size(0)
        pad_len = max_len - seq_len

        padded_input_ids.append(
            torch.cat([
                input_ids_list[i],
                torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long)
            ])
        )
        padded_attention.append(
            torch.cat([
                attention_list[i],
                torch.zeros(pad_len, dtype=torch.long)
            ])
        )
        padded_labels.append(
            torch.cat([
                label_list[i],
                torch.full((pad_len,), -100, dtype=torch.long)
            ])
        )

    batch_input_ids = torch.stack(padded_input_ids)
    batch_attention = torch.stack(padded_attention)
    batch_labels    = torch.stack(padded_labels)

    return {
        "input_ids": batch_input_ids,
        "attention_mask": batch_attention,
        "labels": batch_labels
    }

class CustomTrainer(Trainer):
    """
    We keep your custom loss so that only the unmasked tokens (inside guidance tags) contribute.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=None
        )

        # shift logits and labels
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
        )

        return (loss, outputs) if return_outputs else loss

def initialize_guidance_token_embeddings(model, tokenizer, guidance_tokens):
    """
    After adding "<guidance>" and "</guidance>", initialize their embeddings
    to mean + small noise so they do not start out as all zeros.
    """
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight

        # All existing embeddings (excluding newly appended ones)
        num_original = embeddings.size(0) - len(guidance_tokens)
        existing = embeddings[:num_original]
        mean = existing.mean(dim=0)
        std  = existing.std(dim=0)

        for tok in guidance_tokens:
            idx = tokenizer.convert_tokens_to_ids(tok)
            embeddings[idx] = mean + std * 0.1 * torch.randn_like(mean)
        logger.info(f"Initialized guidance embeddings: {guidance_tokens}")

def main(dataset_type: str = "normal", model_family: str = "qwen", model_name: str = "Qwen/Qwen2.5-7B-Instruct"):
    logging.basicConfig(level=logging.INFO)

    # Load your YAML config
    lora_config_path = f"src/configs/finetune_config.yaml" # change this to your actual config path
    with open("src/configs/finetune_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    max_length = config["max_length"]
    output_dir = config["output_dir"]
    batch_size = config["batch_size"]

    # Load DeepSpeed config
    deepspeed_config_path = f"src/configs/deepspeed_config.json" # change this to your actual config path
    if not os.path.exists(deepspeed_config_path):
        raise FileNotFoundError(f"{deepspeed_config_path} not found")

    # 1) Load tokenizer and add <guidance> tags
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    special_tokens = ["<guidance>", "</guidance>"]
    num_added = tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    logger.info(f"Added {num_added} special tokens: {special_tokens}")

    # Check they are single‐token
    for tok in special_tokens:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        pieces = tokenizer.convert_ids_to_tokens([tok_id])
        assert pieces == [tok], f"{tok} was split into {pieces}"

    # 2) Build datasets
    train_dataset_path = f"./data/processed/train_{dataset_type}.jsonl"
    val_dataset_path   = f"./data/processed/val_{dataset_type}.jsonl"
    train_dataset = DialogueDataset(train_dataset_path, tokenizer, max_length)
    val_dataset   = DialogueDataset(val_dataset_path,   tokenizer, max_length)
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")

    # 3) Load model, resize embeddings, initialize guidance embeddings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=None,
        trust_remote_code=True
    )
    model.resize_token_embeddings(len(tokenizer))
    initialize_guidance_token_embeddings(model, tokenizer, special_tokens)

    # 4) Apply LoRA
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()

    # 5) TrainingArguments
    training_args = TrainingArguments(
        output_dir=f"output_dir_{model_family}_{dataset_type}",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        save_strategy="epoch",
        eval_steps=500,
        logging_dir="./logs",
        logging_steps=25,
        deepspeed=deepspeed_config_path,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        gradient_accumulation_steps=4,
        bf16=True,
        weight_decay=0.01,
        warmup_steps=100,
        report_to=None,
        save_total_limit=2,
        load_best_model_at_end=False,
        gradient_checkpointing=True,
        optim="adamw_torch",
    )

    # 6) Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=lambda batch: custom_collate_fn(batch, tokenizer)
    )

    # 7) Train
    logger.info("Starting training…")
    trainer.train()
    logger.info("Training finished.")

    # 8) Save
    final_model_path = os.path.join(output_dir, f"final_{model_family}_{dataset_type}")
    os.makedirs(final_model_path, exist_ok=True)
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)

    # 9) Save metadata
    metadata = {
        "guidance_tokens": special_tokens,
        "guidance_ids": [tokenizer.convert_tokens_to_ids(t) for t in special_tokens],
        "base_model": model_name,
        "lora_config": {
            "r": config["lora_rank"],
            "lora_alpha": config["lora_alpha"],
            "target_modules": list(lora_config.target_modules),  # Convert set to list
            "lora_dropout": config["lora_dropout"],
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
    }
    with open(os.path.join(final_model_path, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved fine-tuned model to {final_model_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Fine-tune a model with guidance tokens.")

    parser.add_argument("--dataset_type", type=str, choices=["normal", "idk", "attack", "attack_idk"], default="attack",
                        help="Type of dataset to use: 'normal' or 'idk' etc.")
    parser.add_argument("--model_family", type=str, default="qwen",
                        help="Model family to use for configuration.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Base model name.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training.")

    args = parser.parse_args()

    dataset_type = args.dataset_type
    model_family = args.model_family
    model_name = args.model_name
    local_rank = args.local_rank

    
    main(dataset_type=dataset_type, model_family=args.model_family, model_name=args.model_name)