# custom dataset class for AI tutor interactions
# This class reads a CSV file containing AI tutor interaction data and provides methods to access the data

from torch.utils.data import Dataset
import pandas as pd
# PLACE THE INPUT FILE IN THE DATASET DIRECTORY
# e.g. ../../dataset/input_file.csv

class AITutorDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (str): Path to the CSV file containing AI tutor interaction data.
        """
        self.data = pd.read_csv(csv_file)  # Load the dataset
    
    def __len__(self):
        """Returns the total number of samples in the dataset"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Retrieves one sample (row) from the dataset"""
        row = self.data.iloc[idx]
        
        # Extract required fields
        sample = {
            "interaction_id": row["interaction_id"],
            "question_text": row["question_text"],
            "message_text": row["message_text"],  # Full interaction
            "solution_text": row["solution_text"],  # Expert solution
            "model": row["model"],  # Model version
            "discipline": row["discipline"],  # Student performance
        }
        return sample

# Test the dataset class
if __name__ == "__main__":
    dataset = AITutorDataset("../../dataset/input_file.csv")
    print(f"Dataset size: {len(dataset)}")
    print(dataset[0])  