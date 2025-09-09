# step1_basic_setup.py
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Basic amino acid properties
amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
hydrophobicity_scale = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# Create sample mutational data
def create_sample_data(n_samples=100):
    """Create simple sample data for testing"""
    sequences = []
    labels = []  # ΔΔG stability changes
    
    for i in range(n_samples):
        length = np.random.randint(20, 100)
        sequence = ''.join(np.random.choice(amino_acids, length))
        sequences.append(sequence)
        
        # Generate realistic ΔΔG values (most mutations are destabilizing)
        ddG = np.random.normal(0, 1.5)  # Mean ~0, but skewed
        labels.append(ddG)
    
    return sequences, labels

# Test the basic setup
sequences, labels = create_sample_data(10)
print(f"Created {len(sequences)} sample sequences")
print(f"Sample sequence: {sequences[0][:20]}...")
print(f"Sample ΔΔG: {labels[0]:.3f}")

if __name__ == "__main__":
    print("Step 1: Basic setup complete")
    print(f"Available amino acids: {len(amino_acids)}")
    print(f"Hydrophobicity scale keys: {list(hydrophobicity_scale.keys())[:5]}...")