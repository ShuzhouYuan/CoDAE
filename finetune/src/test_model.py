import torch
import json
import os
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer
from peft import PeftModel
import argparse

def load_finetuned_model(model_path):
    """
    Load the fine-tuned model with LoRA adapters and special tokens onto GPU
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("model_loader")
    
    # Check for GPU availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. GPU is required for inference.")
    
    # Load metadata
    metadata_path = os.path.join(model_path, "training_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata: {metadata}")
        base_model_name = metadata.get("base_model", "/data/cat/ws/hagh463f-cot_synthdata/code/cot_synthdata/macmillan/fine_tuning/outputs/checkpoints/final_qwen_model")
    else:
        logger.warning("No metadata found, using default base model")
        base_model_name = "/data/cat/ws/hagh463f-cot_synthdata/code/cot_synthdata/macmillan/fine_tuning/outputs/checkpoints/final_qwen_normal_new"
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Verify special tokens
    guidance_tokens = ["<guidance>", "</guidance>"]
    for token in guidance_tokens:
        token_id = tokenizer.convert_tokens_to_ids(token)
        logger.info(f"Token '{token}' has ID: {token_id}")
    
    # Load base model onto GPU with proper device mapping
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # Use "auto" instead of "cuda" for better device management
        trust_remote_code=True  # Add this if using custom model architectures
    )
    
    # Ensure model is on GPU before resizing embeddings
    if not next(base_model.parameters()).is_cuda:
        base_model = base_model.cuda()
    
    # Resize embeddings to match tokenizer - this must happen after model is on GPU
    original_vocab_size = base_model.get_input_embeddings().num_embeddings
    new_vocab_size = len(tokenizer)
    
    if original_vocab_size != new_vocab_size:
        logger.info(f"Resizing embeddings from {original_vocab_size} to {new_vocab_size}")
        base_model.resize_token_embeddings(new_vocab_size)
        
        # Ensure the new embeddings are on the correct device
        base_model.get_input_embeddings().to(base_model.device)
        if hasattr(base_model, 'get_output_embeddings') and base_model.get_output_embeddings() is not None:
            base_model.get_output_embeddings().to(base_model.device)
    
    # Load LoRA adapters with explicit device mapping
    logger.info(f"Loading LoRA adapters from: {model_path}")
    try:
        # Try loading with device_map first
        model = PeftModel.from_pretrained(
            base_model, 
            model_path, 
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
    except Exception as e:
        logger.warning(f"Failed to load with device_map, trying alternative approach: {e}")
        # Alternative approach: load without device_map and move manually
        model = PeftModel.from_pretrained(base_model, model_path)
        model = model.to("cuda")
    
    # Ensure all model parameters are on the same device
    device = next(model.parameters()).device
    for name, param in model.named_parameters():
        if param.device != device:
            logger.info(f"Moving parameter {name} from {param.device} to {device}")
            param.data = param.data.to(device)
    
    # Merge adapters for inference
    logger.info("Merging LoRA adapters...")
    model = model.merge_and_unload()
    
    # Final device check
    model_device = next(model.parameters()).device
    logger.info(f"Model loaded successfully on device: {model_device}")
    
    return model, tokenizer

def interactive_test(model, tokenizer):
    """
    Live inference with streaming output
    """
    print("=== Live Model Inference ===")
    print("Type 'quit' to exit")
    print("Enter your academic question or problem to begin.")
    print()
    # System prompt for Socratic-style academic guidance
    system_prompt = """You are a helpful chatbot that helps with users *solve academic problems by guiding them and asking them thought provoking questions*.\n\n- Your goal is to **guide the student toward the correct answer without explicitly providing it**.\n- Your responses should be structured using a guided reasoning format enclosed within `<guidance>` and `</guidance>` tags.\n- The responses you generate are chain of thought for students to help them solve problems.\n- You talk in Socratic style by asking thought-provoking questions that encourage the student to reflect on their understanding and think deeply about the subject matter.\n- You are helping a student with a specific homework question.\n- You should **not** provide the answer directly, but rather ask questions that lead the student to discover the answer on their own.\n\n- Questions should be open-ended and encourage critical thinking.\n\n\nQuestion: Inequality, Social Insurance, and\nRedistribution — End of Chapter Problem\nOne way to levy a tax on things people buy is through a sales tax—a tax on purchases that is typically a percentage of the purchase price. For example, in Michigan, everyone, regardless of their income, pays a 6% sales tax on their purchases. \n**a.** Sales tax \nis relatively more expensive to the rich than to the poor.\nis relatively more expensive to the poor than to the rich.\nimpacts poor and rich families in the same way because everyone pays the same percentage. \nis a relatively small expense for both the rich and the poor. \n**b.** A sales tax is (a value from \"regressive\",\"progressive\").\n<user>: Start with underlying concepts\n<assistant>: <guidance>Let's break down the concept of a sales tax and how it affects different income groups.1. **Sales Tax Definition**: A sales tax is a tax levied on the purchase of goods and services, usually as a percentage of the purchase price.2. **Income Distribution**: Poor families generally spend a larger portion of their income on essential goods and services compared to rich families.3. **Tax Burden**: Since the sales tax is a fixed percentage of the purchase price, how does this impact the relative burden on poor versus rich families?Can you think about how the fixed percentage of a sales tax might affect the budget of a poor family differently than a rich family?</guidance>\n<user>: Poor families spend a larger portion of their income on essential goods.\n<assistnat>: <guidance>Exactly! Poor families spend a larger portion of their income on essentials. Now, let's consider the implications of this:1. **Essential Spending**: If a poor family spends a significant portion of their income on essentials, how does a 6% sales tax on those essentials impact their overall budget?2. **Discretionary Spending**: Rich families, on the other hand, have more income left after covering essentials. How does the same 6% sales tax affect their budget in comparison?Given that the sales tax is a fixed percentage, how does the relative impact differ between poor and rich families?</guidance>"""

    conversation_history = f"{system_prompt}\n\n"
    first_response = True
    print("type 'chat' to start chatting or 'quit' to exit.")
    
    # Get model device for proper tensor placement
    model_device = next(model.parameters()).device
    
    while True:
        user_input = input("<user>: ").strip()
        
        if user_input.lower() == 'quit':
            break
        
        # Append user input to conversation history
        if first_response:
            conversation_history += f"\n<user>:"+ "The sales tax takes up a larger portion of a poor family's budget." + "\n<assistant>:"
            first_response = False
        else:
            conversation_history += f"\n<user>:{user_input}\n<assistant>:"

        print("<assistant>: ", end="", flush=True)
        
        # Tokenize and ensure tensors are on the correct device
        inputs = tokenizer.encode(conversation_history, return_tensors="pt")
        inputs = inputs.to(model_device)  # Explicitly move to model device
        
        # Stream response
        streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=2560,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,
                streamer=streamer
            )
        
        # Decode response to append to conversation history
        response = tokenizer.decode(outputs[0], skip_special_tokens=False)
        assistant_start = response.find("\n<assistant>:")
        if assistant_start != -1:
            assistant_response = response[assistant_start + len("\n<assistant>:"):]
            assistant_response = assistant_response.replace("<|eot_id|>", "").strip()
        else:
            assistant_response = response
        
        # Append assistant response to conversation history
        conversation_history += f"{assistant_response}"
        print()

def main():
    """
    Main function to load and run live inference
    """
    parser = argparse.ArgumentParser(description="Interactive model inference")
    parser.add_argument("--model_path", type=str, default="/data/cat/ws/hagh463f-cot_synthdata/code/cot_synthdata/macmillan/fine_tuning/outputs/checkpoints/final_qwen_normal_new", help="Path to the fine-tuned model")
    args = parser.parse_args()
    model_path = args.model_path

    try:
        print("Loading fine-tuned model...")
        model, tokenizer = load_finetuned_model(model_path)
        interactive_test(model, tokenizer)
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Make sure the model path is correct: {model_path}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging

if __name__ == "__main__":
    main()

# === Live Model Inference ===
# Type 'quit' to exit
# Enter your academic question or problem to begin.

# type 'chat' to start chatting or 'quit' to exit.
# <user>: chat
# <assistant>: The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
#  <tool_call>aty, that's a crucial point. When we say the sales tax takes up a larger portion of a poor family's budget, what does this imply about the financial strain on these families compared to richer ones?

# Now, let's think about the term "regressive" in the context of taxes. A regressive tax is one where the burden falls more heavily on lower-income individuals. Given our discussion, would you say a sales tax fits the definition of a regressive tax based on its impact on different income groups?<|im_end|>

# <user>: pls help
# <assistant>: <tool_call>Sure, let's take this step by step.

# 1. **Regressive Tax Impact**: We've established that the sales tax takes up a larger portion of a poor family’s budget. This means that for poorer families, each dollar they pay in sales tax represents a greater proportion of their total income.

# 2. **Progressive vs. Regressive**: A progressive tax, on the other hand, imposes a higher effective tax rate on high-income individuals. It means that as income increases, the share of income paid in taxes also increases.

# 3. **Sales Tax and Income Groups**: Considering that the sales tax impacts poor and rich families differently, can you now determine if a sales tax is more regressive or less so compared to a progressive tax?

# Which of these descriptions best fits the nature of a sales tax based on its impact on different income groups?<|im_end|>

# <user>: idk please help!!
# <assistant>: <tool_call>Alright, let's think this through together.

# 1. **Understanding the Impact**: We know that the sales tax takes up a larger portion of a poor family's budget. This means that for a poor family, a $100 purchase results in a $6 tax, which is 6% of their income. For a rich family, the same $100 purchase also results in a $6 tax, but since they have much more income, this $6 is a smaller percentage of their total income.

# 2. **Definition of Regressive Tax**: A regressive tax is one where the burden is heavier on lower-income individuals. Based on the information we have, the sales tax seems to fit this description.

# 3. **Comparing to Progressive Taxes**: A progressive tax, like an income tax, imposes a higher effective tax rate on higher incomes. So, a rich person with a higher income will pay a larger absolute amount in taxes, but as a percentage of their income, it might not be as high as for a poor person.

# 4. **Conclusion**: Given that the sales tax takes a larger portion of the income for poorer families, would you say the sales tax is more regressive or more progressive?

# Based on the definitions and the impact we discussed, which term—regressive or progressive—best describes a sales tax in this scenario?<|im_end|>

# <user>: parito optimality??
# <assistant>: <tool_call>I see, it looks like there was a mix-up. Let's clarify the terms again:

# 1. **Regressive Tax**: A tax that imposes a higher percentage of the tax on lower-income individuals. In the case of a sales tax, a fixed percentage (like 6%) on every purchase means that it takes a larger portion of the income for poorer families.

# 2. **Progressive Tax**: A tax that imposes a higher percentage of the tax on higher-income individuals. An income tax, for example, often has brackets where the higher your income, the higher the percentage of that income you pay in taxes.

# Given this, a sales tax is considered a **regressive tax** because it disproportionately affects lower-income families.

# So, to summarize:
# - A sales tax is **regressive**.

# Does this make sense, or do you need any more clarification?<|im_end|>

# <user>: what is sales tax
# <assistant>: Error loading model: Input length of input_ids is 3064, but `max_length` is set to 2560. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.
# Make sure the model path is correct: /data/cat/ws/hagh463f-cot_synthdata/code/cot_synthdata/macmillan/fine_tuning/outputs/checkpoints/final_qwen_model
# Traceback (most recent call last):
#   File "/data/cat/ws/hagh463f-cot_synthdata/code/cot_synthdata/macmillan/fine_tuning/src/test_model.py", line 399, in main
#     interactive_test(model, tokenizer)
#   File "/data/cat/ws/hagh463f-cot_synthdata/code/cot_synthdata/macmillan/fine_tuning/src/test_model.py", line 366, in interactive_test
#     outputs = model.generate(
#               ^^^^^^^^^^^^^^^
#   File "/data/cat/ws/hagh463f-cot_synthdata/code/cot_synthdata/macmillan/fine_tuning/venv/lib/python3.11/site-packages/torch/utils/_contextlib.py", line 116, in decorate_context
#     return func(*args, **kwargs)
#            ^^^^^^^^^^^^^^^^^^^^^
#   File "/data/cat/ws/hagh463f-cot_synthdata/code/cot_synthdata/macmillan/fine_tuning/venv/lib/python3.11/site-packages/transformers/generation/utils.py", line 2453, in generate
#     self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
#   File "/data/cat/ws/hagh463f-cot_synthdata/code/cot_synthdata/macmillan/fine_tuning/venv/lib/python3.11/site-packages/transformers/generation/utils.py", line 1617, in _validate_generated_length
#     raise ValueError(
# ValueError: Input length of input_ids is 3064, but `max_length` is set to 2560. This can lead to unexpected behavior. You should consider increasing `max_length` or, better yet, setting `max_new_tokens`.