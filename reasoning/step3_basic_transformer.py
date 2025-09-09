# step3_basic_transformer.py
import torch
import torch.nn as nn
import math

# Import from previous steps
from step2_tokenization_dataset import ProteinSequenceDataset

class BasicProteinTransformer(nn.Module):
    """Basic transformer for protein sequences"""
    
    def __init__(self, vocab_size=22, d_model=64, nhead=4, num_layers=2, 
                 max_length=100, dropout=0.1):
        super(BasicProteinTransformer, self).__init__()
        
        self.d_model = d_model
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
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
        
        # Simple classification/regression head
        self.output_layer = nn.Linear(d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def create_positional_encoding(self, max_len, d_model):
        """Create positional encoding"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
    
    def forward(self, src, src_mask=None):
        # Embedding
        embedded = self.embedding(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        seq_len = src.size(1)
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(src.device)
        embedded = embedded + pos_encoding
        embedded = self.dropout(embedded)
        
        # Transformer encoder
        if src_mask is not None:
            transformer_output = self.transformer_encoder(embedded, src_key_padding_mask=src_mask)
        else:
            transformer_output = self.transformer_encoder(embedded)
        
        # Global average pooling
        if src_mask is not None:
            mask_expanded = (~src_mask).unsqueeze(-1).float()
            pooled_output = (transformer_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            pooled_output = transformer_output.mean(dim=1)
        
        # Output layer
        output = self.output_layer(pooled_output)
        return output

# Test the model
if __name__ == "__main__":
    from step1_basic_setup import create_sample_data
    from step2_tokenization_dataset import ProteinSequenceDataset
    from torch.utils.data import DataLoader
    
    # Create sample data
    sequences, labels = create_sample_data(20)
    dataset = ProteinSequenceDataset(sequences, labels, max_length=50)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model
    model = BasicProteinTransformer(vocab_size=len(dataset.token_to_idx))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test forward pass
    batch = next(iter(dataloader))
    sequences_batch = batch['sequence'].to(model.embedding.weight.device)
    src_mask = (sequences_batch == 0)  # PAD token mask
    
    with torch.no_grad():
        output = model(sequences_batch, src_mask)
        print(f"Model output shape: {output.shape}")
        print(f"Sample predictions: {output.flatten()[:3]}")
    
    print("Step 3: Basic transformer model complete")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")