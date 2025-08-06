# Unit test the prompt augmentation class

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_augmentation.src.augment import MessageAugmentor

augmentor = MessageAugmentor(model_name="Qwen/Qwen2.5-72B-Instruct", dataset_path="../../dataset/input_file.csv", system_message="../input/system_message.md", prompt="../input/prompt.md")
print(f"System message: {augmentor.system_message}")
print("-------------------")
print(augmentor.augment_prompt(idx=0))
print("-------------------")
print(augmentor.augment_prompt(idx=1))
print("-------------------")
print(augmentor.augment_prompt(idx=2))