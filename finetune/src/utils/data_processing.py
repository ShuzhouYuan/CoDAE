from typing import Dict, List
from transformers import AutoTokenizer
import logging
import json
from src.utils.logger import setup_logger

def process_dialogue(item: Dict, tokenizer: AutoTokenizer, max_length: int) -> List[Dict]:
    """
    Process a dialogue item into input-output pairs for fine-tuning.

    Args:
        item: Dictionary containing dialogue data.
        tokenizer: Hugging Face tokenizer.
        max_length: Maximum token length for each example.

    Returns:
        List of dictionaries with input-output pairs.
    """
    logger = logging.getLogger("data_processing")
    examples = []

    # Extract dialogue and question
    dialogue = item.get("dialogue", [])
    question = item.get("question", "")
    if not dialogue or not question:
        logger.warning(f"Skipping item with empty dialogue or question: {item.get('id', 'unknown')}")
        return examples

    # System prompt
    system_prompt = (
        "You are a helpful chatbot that helps with users *solve academic problems by guiding them and asking them thought provoking questions*.\n\n"
        "- Your goal is to **guide the student toward the correct answer without explicitly providing it**.\n"
        "- Your responses should be structured using a guided reasoning format enclosed within `<guidance>` and `</guidance>` tags.\n"
        "- The responses you generate are chain of thought for students to help them solve problems.\n"
        "- You talk in Socratic style by asking thought-provoking questions that encourage the student to reflect on their understanding and think deeply about the subject matter.\n"
        "- You are helping a student with a specific homework question.\n"
        "- You should **not** provide the answer directly, but rather ask questions that lead the student to discover the answer on their own.\n"
        "- Questions should be open-ended and encourage critical thinking."
    )

    # Concatenate dialogue turns into a single input string
    input_parts = [system_prompt, "\n\nQuestion: " + question]
    for turn in dialogue:
        role = turn.get("role")
        content = turn.get("content", "")
        if role == "user":
            input_parts.append(f"\n<user>: {content}")
        elif role == "assistant":
            input_parts.append(f"\n<assistant>: <guidance>{content}</guidance>")
    
    input_text = "".join(str(part) for part in input_parts)
    if not isinstance(input_text, str):
        logger.error(f"Input is not a string: {type(input_text)}")
        input_text = str(input_text)

    # Process each assistant turn as an output
    for i, turn in enumerate(dialogue):
        if turn.get("role") == "assistant":
            output_text = f"<guidance>{turn.get('content', '')}</guidance>"
            # Check token length
            input_tokens = tokenizer.encode(input_text, add_special_tokens=True)
            output_tokens = tokenizer.encode(output_text, add_special_tokens=True)
            if len(input_tokens) > max_length or len(output_tokens) > max_length:
                logger.warning(f"Skipping example {i} due to excessive token length: input={len(input_tokens)}, output={len(output_tokens)}")
                continue
            examples.append({"input": input_text, "output": output_text})

    return examples

def test_processing(data_path: str, model_name: str, max_length: int = 3072):
    """
    Test the dialogue processing by printing all generated examples for one item.
    """
    logger = logging.getLogger("data_processing")
    logger.info("Running test case for dialogue processing")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    with open(data_path, "r") as f:
        item = json.loads(f.readline().strip())
    
    processed_examples = process_dialogue(item, tokenizer, max_length)
    
    if processed_examples:
        logger.info(f"Produced {len(processed_examples)} examples")
        print(f"Produced {len(processed_examples)} examples")
        for idx, example in enumerate(processed_examples, 1):
            input_tokens = len(tokenizer(example["input"], add_special_tokens=False)["input_ids"])
            output_tokens = len(tokenizer(example["output"], add_special_tokens=False)["input_ids"])
            logger.info(f"Example {idx}:")
            logger.info(f"Input ({input_tokens} tokens):\n{example['input']}")
            logger.info(f"Output ({output_tokens} tokens):\n{example['output']}")
    else:
        logger.warning("No valid examples produced for test item")
        print("No valid examples produced for test item")

if __name__ == "__main__":
    setup_logger("data_processing", "./outputs/logs")