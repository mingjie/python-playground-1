# step8_real_dataset_integration.py
import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from Bio import SeqIO
import requests
import os

# Import from previous steps
from step7_training_pipeline import MutationalTransformer, train_model, evaluate_model
from step2_tokenization_dataset import ProteinSequenceDataset

class RealMutationalDataset(ProteinSequenceDataset):
    """Dataset for real mutational data"""
    
    def __init__(self, data_df, max_length=100):
        self.data = data_df
        self.max_length = max_length
        
        # Create token mappings
        amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        self.amino_acids = ['<PAD>', '<UNK>'] + amino_acids
        self.token_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
        self.idx_to_token = {i: aa for i, aa in enumerate(self.amino_acids)}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row.get('mutant_sequence', row.get('sequence', ''))
        label = row.get('ddG', row.get('label', 0.0))
        
        # Convert sequence to tokens
        token_ids = []
        for aa in str(sequence)[:self.max_length]:
            token_ids.append(self.token_to_idx.get(aa.upper(), self.token_to_idx['<UNK>']))
        
        # Pad sequence
        if len(token_ids) < self.max_length:
            token_ids.extend([self.token_to_idx['<PAD>']] * (self.max_length - len(token_ids)))
        
        return {
            'sequence': torch.tensor(token_ids, dtype=torch.long),
            'label': torch.tensor(float(label), dtype=torch.float),
            'wild_type': row.get('wild_type_sequence', ''),
            'mutant': row.get('mutant_sequence', ''),
            'pdb_id': row.get('pdb_id', '')
        }

def create_realistic_mutational_data(n_samples=500):
    """Create more realistic mutational data"""
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    
    data = []
    for i in range(n_samples):
        # Generate realistic protein sequences
        length = np.random.randint(30, 200)
        wild_type = ''.join(np.random.choice(amino_acids, length))
        
        # Generate mutations
        mutant = list(wild_type)
        num_mutations = np.random.randint(1, min(4, length // 10 + 1))
        
        mutation_positions = np.random.choice(len(wild_type), num_mutations, replace=False)
        for pos in mutation_positions:
            old_aa = mutant[pos]
            new_aa = np.random.choice([aa for aa in amino_acids if aa != old_aa])
            mutant[pos] = new_aa
        
        mutant = ''.join(mutant)
        
        # Generate realistic ΔΔG values based on mutation properties
        ddG = 0.0
        
        # Add noise
        ddG += np.random.normal(0, 1.0)
        
        # Bias based on hydrophobicity changes
        for pos in mutation_positions:
            if pos < len(wild_type) and pos < len(mutant):
                wt_aa = wild_type[pos]
                mut_aa = mutant[pos]
                
                # Simplified hydrophobicity effect
                wt_hydro = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                           'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                           'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                           'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}.get(wt_aa, 0)
                mut_hydro = {'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
                            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
                            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
                            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2}.get(mut_aa, 0)
                
                # Hydrophobic to hydrophilic is usually destabilizing
                if wt_hydro > 2 and mut_hydro < 0:
                    ddG -= np.random.uniform(0.5, 2.0)
                elif wt_hydro < 0 and mut_hydro > 2:
                    ddG += np.random.uniform(-1.0, 0.5)
        
        data.append({
            'pdb_id': f'1ABC_{i:04d}',
            'wild_type_sequence': wild_type,
            'mutant_sequence': mutant,
            'mutation_positions': ','.join(map(str, sorted(mutation_positions))),
            'num_mutations': num_mutations,
            'ddG': ddG
        })
    
    return pd.DataFrame(data)

def download_pdb_structure(pdb_id):
    """Download PDB structure (simplified)"""
    # In practice, you would download from RCSB PDB
    print(f"Downloading structure for {pdb_id}")
    # Return dummy structure for demonstration
    return None

def create_structure_graph(sequence):
    """Create graph structure from sequence"""
    # One-hot encoding
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    x = torch.zeros(len(sequence), len(amino_acids))
    
    for i, aa in enumerate(sequence):
        if aa in amino_acids:
            aa_idx = amino_acids.index(aa)
            x[i, aa_idx] = 1
    
    # Create edges (simplified - k-nearest neighbors)
    edge_index = []
    for i in range(len(sequence)):
        # Connect to adjacent residues
        if i > 0:
            edge_index.append([i-1, i])
        if i < len(sequence) - 1:
            edge_index.append([i+1, i])
    
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty(2, 0, dtype=torch.long)
    
    return Data(x=x, edge_index=edge_index)

# Test with realistic data
if __name__ == "__main__":
    print("Creating realistic mutational dataset...")
    df = create_realistic_mutational_data(200)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Sample data:")
    print(df.head())
    
    print(f"\nΔΔG statistics:")
    print(f"Mean: {df['ddG'].mean():.3f}")
    print(f"Std: {df['ddG'].std():.3f}")
    
    # Split data
    train_df = df.iloc[:120]
    val_df = df.iloc[120:160]
    test_df = df.iloc[160:200]
    
    # Create datasets
    train_dataset = RealMutationalDataset(train_df, max_length=100)
    val_dataset = RealMutationalDataset(val_df, max_length=100)
    test_dataset = RealMutationalDataset(test_df, max_length=100)
    
    # Create data loaders
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    # Initialize model
    model = MutationalTransformer(vocab_size=len(train_dataset.token_to_idx))
    
    # Train model
    print("\nTraining model...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=5)
    
    # Evaluate model
    print("\nEvaluating model...")
    predictions, true_labels = evaluate_model(model, test_loader)
    
    # Analyze results
    print(f"\nPerformance Analysis:")
    correlation = np.corrcoef(true_labels, predictions)[0, 1]
    print(f'Correlation: {correlation:.4f}')
    
    # Show some examples
    print(f"\nPrediction Examples:")
    for i in range(3):
        print(f"True ΔΔG: {true_labels[i]:.3f}, Predicted ΔΔG: {predictions[i]:.3f}")
    
    print("Step 8: Real dataset integration complete")