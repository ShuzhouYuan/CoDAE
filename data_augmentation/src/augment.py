# augment data by using LLM

import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers.generation.streamers import BaseStreamer
from data_augmentation.src.DataClass import AITutorDataset
# PLACE THE INPUT FILE IN THE DATASET DIRECTORY
# e.g. ../../dataset/input_file.csv

class ConsoleFileStreamer(BaseStreamer):
    def __init__(self, tokenizer, file_path):
        self.tokenizer = tokenizer
        self.file = open(file_path, "w")

    def put(self, text_ids):
        text = self.tokenizer.decode(text_ids[0], skip_special_tokens=True)
        print(text, end='', flush=True)  # Stream to terminal
        self.file.write(text)
        self.file.flush()

    def end(self):
        self.file.close()


class MessageAugmentor:
    
    def __init_model__(self, model_name="Qwen/Qwen2.5-72B-Instruct"):
        # Create custom configuration for large context window
        config = AutoConfig.from_pretrained(model_name)
        config.rope_scaling = {
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "type": "yarn"
        }

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype="auto",
            device_map="auto"
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer
    
    def __load_dataset__(self, dataset_path="../../dataset/input_file.csv"):
      dataset = AITutorDataset(dataset_path)
      return dataset

    def __init__(self,model_name="Qwen/Qwen2.5-72B-Instruct", dataset_path="../../dataset/input_file.csv", system_message="system_message.md", prompt="prompt.md", output_dir="../output"):
        self.model, self.tokenizer = self.__init_model__(model_name)
        # load system message from file
        with open(system_message, "r") as f:
            self.system_message = f.read()
        # load prompt from file
        with open(prompt, "r") as f:
            self.prompt = f.read()
        # load dataset
        self.dataset = self.__load_dataset__(dataset_path)
        self.output_dir = output_dir

    def extract_answer_from_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as file:
            content = file.read()

        # Find the position of the <begin_answer> tag
        tag_index = content.find("<begin_answer>")
        
        if tag_index == -1:
            return None  # Return None if the tag is not found
        
        # Extract everything after the tag
        return content[tag_index + len("<begin_answer>"):].strip()
    
    def replace_first_line_with_question(self, text, question):
        # replaces the first line of the file with the question and returns the text
        lines = text.split("\n")

        # Replace the first line with the question
        lines[0] = "<question>" + question + "</question>" + "\n\n"

        return "".join(lines)

    def augment_prompt(self, idx):
        sample = self.dataset[idx]
        formatted_prompt = self.prompt.format(
            question = sample["question_text"],
            solution = sample["solution_text"],
            message = sample["message_text"],
            discipline = sample["discipline"],
            )
        id = sample["interaction_id"]
        discipline = sample["discipline"]
        question = sample["question_text"]
        return formatted_prompt, id, discipline, question

    def generate(self, idx=0, max_tokens=98304):
        formatted_prompt, id, discipline, question = self.augment_prompt(idx)
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": formatted_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        # create a response file with question id as name and in the subdirectory of the discipline
        os.makedirs(os.path.join(self.output_dir, discipline), exist_ok=True)
        # create filename with question id in the proper subdirectory of the discipline
        filename = os.path.join(self.output_dir, discipline, f"{id}.md")
        streamer = ConsoleFileStreamer(self.tokenizer, os.path.join(self.output_dir, filename))

        # Generate with file streamer
        self.model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            streamer=streamer
        )
        streamer.end()
        # get rid of everything that is before <begin_answer> tag
        extracted_answer = self.extract_answer_from_file(filename)
        # replace first line with question
        output_content = self.replace_first_line_with_question(extracted_answer, question)
        # write the answer to the file
        with open(filename, "w") as f:
            f.write(output_content)
    

if __name__ == "__main__":
    augmentor = MessageAugmentor()
    # test the augmentor with the first sample
    augmentor.augment_prompt(idx=0)
    print("Augmentation complete!")
