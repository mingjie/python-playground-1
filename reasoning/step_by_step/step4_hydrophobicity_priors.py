# step4_hydrophobicity_priors.py
import torch
import torch.nn as nn
import math
import numpy as np

# Import from previous steps
from step3_basic_transformer import BasicProteinTransformer

class HydrophobicityTransformer(nn.Module):
    """Transformer with hydrophobicity priors"""
    
    def __init__(self, vocab_size=22, d_model=64, nhead=4, num_layers=2,
                 max_length=100, dropout=0.1):
        super(HydrophobicityTransformer, self).__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Hydrophobicity scale
        self.hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
            '<PAD>': 0.0, '<UNK>': 0.0
        }
        
        # Enhanced embedding with hydrophobicity
        self.token_embedding = nn.Embedding(vocab_size, d_model // 2)
        self.hydrophobicity_embedding = nn.Linear(1, d_model // 2)
        
        # Positional encoding
        self.positional_encoding = self.create_positional_encoding(max_length, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def get_hydrophobicity_values(self, tokens):
        """Convert token indices to hydrophobicity values"""
        batch_size, seq_len = tokens.shape
        hydro_values = torch.zeros(batch_size, seq_len, 1)
        
        # Create reverse mapping for tokens to amino acids
        idx_to_aa = {0: '<PAD>', 1: '<UNK>'}  # Simplified
        for aa, idx in self.hydrophobicity_scale.items():
            if aa not in ['<PAD>', '<UNK>']:
                # This is a simplified mapping - in practice, you'd have the full mapping
                pass
        
        # For demonstration, we'll use the actual scale values
        for i in range(batch_size):
            for j in range(seq_len):
                token_idx = tokens[i, j].item()
                if token_idx == 0:  # PAD
                    hydro_values[i, j, 0] = 0.0
                elif token_idx == 1:  # UNK
                    hydro_values[i, j, 0] = 0.0
                else:
                    # Map token indices to actual amino acids (simplified)
                    aa_index = token_idx - 2  # Skip PAD and UNK
                    if aa_index < len(list(self.hydrophobicity_scale.keys())[:-2]):
                        aa = list(self.hydrophobicity_scale.keys())[aa_index + 2]
                        hydro_values[i, j, 0] = self.hydrophobicity_scale[aa]
        
        return hydro_values
    
    def forward(self, src, src_mask=None):
        # Token embeddings

        # move data to cuda
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        src = src.to(device)

        token_embeds = self.token_embedding(src)
        
        # Hydrophobicity embeddings
        hydro_values = self.get_hydrophobicity_values(src).to(src.device)
        hydro_embeds = self.hydrophobicity_embedding(hydro_values)
        
        # Combine embeddings
        embeddings = torch.cat([token_embeds, hydro_embeds], dim=-1)
        
        # Add positional encoding
        seq_len = src.size(1)
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(src.device)
        embeddings = embeddings + pos_encoding
        embeddings = self.dropout(embeddings)
        
        # move data 
        embeddings = embeddings.to(device)
        src_mask = src_mask.to(device)

        # Transformer encoder
        if src_mask is not None:
            transformer_output = self.transformer_encoder(embeddings, src_key_padding_mask=src_mask)
        else:
            transformer_output = self.transformer_encoder(embeddings)
        
        # Global average pooling
        if src_mask is not None:
            mask_expanded = (~src_mask).unsqueeze(-1).float()
            pooled_output = (transformer_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled_output = transformer_output.mean(dim=1)
        
        # Output layer
        output = self.output_layer(pooled_output)
        return output

# Test the enhanced model
if __name__ == "__main__":
    from step1_basic_setup import create_sample_data
    from step2_tokenization_dataset import ProteinSequenceDataset
    from torch.utils.data import DataLoader
    
    # Create sample data
    sequences, labels = create_sample_data(20)
    dataset = ProteinSequenceDataset(sequences, labels, max_length=50)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize enhanced model
    model = HydrophobicityTransformer(vocab_size=len(dataset.token_to_idx))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test forward pass
    batch = next(iter(dataloader))
    sequences_batch = batch['sequence'].to(model.token_embedding.weight.device)
    src_mask = (sequences_batch == 0)
    
    with torch.no_grad():
        output = model(sequences_batch, src_mask)
        print(f"Enhanced model output shape: {output.shape}")
        print(f"Sample predictions: {output.flatten()[:3]}")
    
    print("Step 4: Hydrophobicity priors added")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")