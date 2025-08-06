# Unit test to test the augmentation class to generate prompts

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_augmentation.src.augment import MessageAugmentor

augmentor = MessageAugmentor(model_name="Qwen/Qwen2.5-72B-Instruct", dataset_path="../../dataset/input_file.csv", system_message="../input/system_message.md", prompt="../input/prompt.md")
augmentor.generate(idx=0)
augmentor.generate(idx=1)
augmentor.generate(idx=2)
augmentor.generate(idx=3)