import json
import random
from collections import defaultdict
import re
import os

# Predefined lists of user distress messages and assistant responses
USER_DISTRESS_MESSAGES = [
    "pls help!",
    "I don't understand this at all.",
    "IDK anything!!",
    "This is too hard for me.",
    "Can you explain it differently?",
    "I'm completely lost.",
    "I need more help.",
    "This doesn't make sense.",
    "I'm stuck here.",
    "Can we go back a step?",
    "I want to quit! please help",
    "This is confusing me.",
    "Not getting it.",
    "Help needed!",
    "I'm struggling with this concept.",
    "Can you simplify this?",
    "Still confused.",
    "Need more guidance.",
    "Feeling overwhelmed.",
    "Where should I start?"
]

ASSISTANT_ENCOURAGING_RESPONSES = [
    "<guidance>No worries! Can you tell me which part you're finding difficult?</guidance>",
    "<guidance>Let's take a step back. What specific concept are you struggling with?</guidance>",
    "<guidance>I understand this can be challenging. Let's recall what we discussed earlier.</guidance>",
    "<guidance>It's okay to feel stuck! What's the main point you're unsure about?</guidance>",
    "<guidance>Let's break this down. Which aspect would you like to focus on first?</guidance>",
    "<guidance>Take a deep breath. Can you summarize what we've covered so far?</guidance>",
    "<guidance>I'm here to help! What exactly is confusing you about this?</guidance>",
    "<guidance>Let's approach this differently. How would you explain this to a friend?</guidance>",
    "<guidance>Remember, learning takes time. What's the core concept you're wrestling with?</guidance>",
    "<guidance>Let's revisit the basics. What's your current understanding of this topic?</guidance>",
    "<guidance>I know it's tough, but you're making progress! What specific question do you have?</guidance>",
    "<guidance>Let's connect this to something you already know. Any ideas?</guidance>",
    "<guidance>Don't give up! What part of our discussion made the most sense to you?</guidance>",
    "<guidance>I believe in you! What's one thing you do understand about this?</guidance>",
    "<guidance>Let's try a different angle. What real-life example relates to this concept?</guidance>",
    "<guidance>It's okay to ask for help! What's the first thing that comes to mind?</guidance>",
    "<guidance>Take it one step at a time. Where did you start having trouble?</guidance>",
    "<guidance>I understand you're frustrated, but we'll get through this! What's confusing?</guidance>",
    "<guidance>Let's build from what you know. What's familiar about this concept?</guidance>",
    "<guidance>Deep breaths! Can you identify the key elements we're discussing?</guidance>"
]

def parse_conversation(input_str):
    """Parse input string into prefix and conversation turns"""
    first_user_idx = input_str.find("<user>:")
    if first_user_idx == -1:
        return input_str, []
    
    prefix = input_str[:first_user_idx]
    conversation = []
    remaining = input_str[first_user_idx:]
    
    # Split while preserving delimiters
    parts = re.split(r'(<user>:|<assistant>:)', remaining)
    parts = [p.strip() for p in parts if p.strip()]
    
    # Group into role-content pairs
    for i in range(0, len(parts), 2):
        role_tag = parts[i]
        content = parts[i+1] if i+1 < len(parts) else ""
        
        if role_tag == "<user>:":
            conversation.append(("user", content))
        elif role_tag == "<assistant>:":
            conversation.append(("assistant", content))
    
    return prefix, conversation

def reconstruct_input(prefix, conversation):
    """Rebuild input string from prefix and conversation turns"""
    result = prefix
    for role, content in conversation:
        if role == "user":
            result += f"<user>: {content}\n"
        else:
            result += f"<assistant>: {content}\n"
    return result

def augment_conversation(prefix, conversation, output_str):
    """Augment conversation with distress/help exchanges"""
    new_conversation = conversation.copy()
    new_output = output_str
    
    # Determine valid insertion points (after assistant responses)
    insertion_points = []
    for i in range(len(new_conversation)):
        if new_conversation[i][0] == "assistant" and i < len(new_conversation) - 1:
            insertion_points.append(i)
    
    # Add end point
    insertion_points.append("end")
    
    # Add 1-3 injections
    num_injections = random.randint(1, 3)
    injections = random.sample(insertion_points, min(num_injections, len(insertion_points)))
    
    # Process injections from end to start to preserve indices
    injections.sort(key=lambda x: float('inf') if x == "end" else x, reverse=True)
    
    for point in injections:
        user_msg = random.choice(USER_DISTRESS_MESSAGES)
        assistant_msg = random.choice(ASSISTANT_ENCOURAGING_RESPONSES)
        
        if point == "end":
            # Handle end injection
            if new_conversation and new_conversation[-1][0] == "assistant":
                # Replace last assistant placeholder with current output
                new_conversation[-1] = ("assistant", new_output)
            new_conversation.append(("user", user_msg))
            new_conversation.append(("assistant", ""))
            new_output = assistant_msg
        else:
            # Insert after an assistant response
            new_conversation.insert(point + 1, ("user", user_msg))
            new_conversation.insert(point + 2, ("assistant", assistant_msg))
    
    return new_conversation, new_output

def process_dataset(input_file, output_file, num_augmentations=3):
    """Process dataset and create augmented versions"""
    with open(input_file, "r") as infile, open(output_file, "w") as outfile:
        for line in infile:
            original = json.loads(line)
            prefix, conversation = parse_conversation(original["input"])
            
            # Skip if no conversation found
            if not conversation:
                continue
                
            # Create multiple augmented versions
            for _ in range(num_augmentations):
                new_conv, new_output = augment_conversation(
                    prefix, conversation, original["output"]
                )
                new_input = reconstruct_input(prefix, new_conv)
                
                augmented_example = {
                    "input": new_input,
                    "output": new_output
                }
                outfile.write(json.dumps(augmented_example) + "\n")

# Example usage
if __name__ == "__main__":
    process_dataset("dataset/data/processed/test_normal.jsonl", "dataset/data/processed/test_idk.jsonl")
    process_dataset("dataset/data/processed/train_normal.jsonl", "dataset/data/processed/train_idk.jsonl")
    process_dataset("dataset/data/processed/val_normal.jsonl", "dataset/data/processed/val_idk.jsonl")