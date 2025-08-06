import json
import logging
from typing import List, Dict
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from src.utils.data_processing import process_dialogue
from src.utils.logger import setup_logger
import os

def preprocess_data(input_path: str, output_dir: str, model_name: str, max_length: int = 1024):
    """
    Load JSONL data, process dialogues, split into train/val/test, and save to disk.

    Args:
        input_path: Path to input JSONL file.
        output_dir: Directory to save processed train/val/test JSONL files.
        model_name: Model name for tokenizer.
        max_length: Maximum token length for each example.
    """
    logger = logging.getLogger("preprocess_data")
    logger.info(f"Preprocessing data from {input_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Ensure padding token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Set pad_token to eos_token")

    # Load and process all dialogues
    all_examples = []
    with open(input_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())
            examples = process_dialogue(item, tokenizer, max_length)
            all_examples.extend(examples)
    
    logger.info(f"Processed {len(all_examples)} input-output pairs")

    # Split into train (80%), val (10%), test (10%)
    train_examples, temp_examples = train_test_split(
        all_examples, train_size=0.8, random_state=42, shuffle=True
    )
    val_examples, test_examples = train_test_split(
        temp_examples, train_size=0.5, random_state=42, shuffle=True
    )

    logger.info(f"Train: {len(train_examples)} examples")
    logger.info(f"Validation: {len(val_examples)} examples")
    logger.info(f"Test: {len(test_examples)} examples")

    # Save to disk
    os.makedirs(output_dir, exist_ok=True)
    for split, examples in [
        ("train", train_examples),
        ("val", val_examples),
        ("test", test_examples),
    ]:
        output_path = os.path.join(output_dir, f"{split}.jsonl")
        with open(output_path, "w") as f:
            for example in examples:
                f.write(json.dumps(example) + "\n")
        logger.info(f"Saved {split} set to {output_path}")

if __name__ == "__main__":
    setup_logger("preprocess_data", "./outputs/logs")
    preprocess_data(
        input_path="./data/dataset/train.jsonl",
        output_dir="./data/dataset/processed",
        model_name="meta-llama/Llama-3.2-3B-Instruct" # using llama's tokenizer. This is just used for for length bounding checks and creating the input-output pairs, any model can be used here.
    )
