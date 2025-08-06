# python script to generate chain of thought synthetic data

import sys
import argparse
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_augmentation.src.augment import MessageAugmentor

parser = argparse.ArgumentParser(description="Generate synthetic data using a language model.")
parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-72B-Instruct", help="Name of the model to use.")
parser.add_argument("--dataset_path", type=str, default="../../dataset/input_file.csv", help="Path to the dataset.")
parser.add_argument("--system_message", type=str, default="../input/system_message.md", help="Path to the system message file.")
parser.add_argument("--system_prompt", type=str, default="../input/prompt.md", help="Path to the prompt file.")
parser.add_argument("--output_dir", type=str, default="../output", help="Directory to save the output.")
parser.add_argument("--start_idx", type=int, default=0, help="Starting index for generating prompts.")
parser.add_argument("--end_idx", type=int, default=1, help="Ending index for generating prompts.")

args = parser.parse_args()

start_idx = args.start_idx
end_idx = args.end_idx
output_dir = args.output_dir
system_prompt = args.system_prompt
system_message = args.system_message
model_name = args.model_name
dataset_path = args.dataset_path

# print args
print(f"Model Name: {model_name}")
print(f"Dataset Path: {dataset_path}")
print(f"System Message: {system_message}")
print(f"Prompt: {system_prompt}")
print(f"Output Directory: {output_dir}")
print(f"Start Index: {start_idx}")
print(f"End Index: {end_idx}")


augmentor = MessageAugmentor(
  model_name=model_name,
  dataset_path=dataset_path,
  system_message=system_message,
  prompt=system_prompt)

print(f"Generating prompts from {start_idx} to {end_idx}")
for i in range(start_idx, end_idx):
  augmentor.generate(idx=i)