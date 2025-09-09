# step2_tokenization_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Import from previous step
from step1_basic_setup import amino_acids, hydrophobicity_scale, create_sample_data

class ProteinSequenceDataset(Dataset):
    """Dataset for protein sequences with tokenization"""
    
    def __init__(self, sequences, labels, max_length=100):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        
        # Create token mappings
        self.amino_acids = ['<PAD>', '<UNK>'] + list(hydrophobicity_scale.keys())
        self.token_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.idx_to_token = {i: aa for i, aa in enumerate(self.amino_acids)}
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        # Convert sequence to tokens
        token_ids = []
        for aa in sequence[:self.max_length]:
            token_ids.append(self.token_to_idx.get(aa, self.token_to_idx['<UNK>']))
        
        # Pad sequence
        if len(token_ids) < self.max_length:
            token_ids.extend([self.token_to_idx['<PAD>']] * (self.max_length - len(token_ids)))
        
        return {
            'sequence': torch.tensor(token_ids, dtype=torch.long),
            'label': torch.tensor(float(label), dtype=torch.float)
        }

# Test tokenization
sequences, labels = create_sample_data(20)
dataset = ProteinSequenceDataset(sequences, labels, max_length=50)

print(f"Dataset size: {len(dataset)}")
sample_item = dataset[0]
print(f"Sample sequence tensor shape: {sample_item['sequence'].shape}")
print(f"Sample label: {sample_item['label']}")
print(f"First 10 tokens: {sample_item['sequence'][:10]}")

# Test DataLoader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
batch = next(iter(dataloader))
print(f"Batch sequence shape: {batch['sequence'].shape}")
print(f"Batch label shape: {batch['label'].shape}")

if __name__ == "__main__":
    print("Step 2: Tokenization and dataset complete")