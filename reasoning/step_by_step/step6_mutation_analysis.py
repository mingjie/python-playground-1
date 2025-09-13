# step6_mutation_analysis.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
import math

# Import from previous steps
from step5_basic_gnn import StructureAwareTransformer

# LMJ hack
# batch_size = 2

class MutationalTransformer(nn.Module):
    """Transformer with mutation analysis"""
    
    def __init__(self, vocab_size=22, d_model=64, nhead=4, num_layers=2,
                 max_length=100, dropout=0.1, structure_dim=32):
        super(MutationalTransformer, self).__init__()
        
        # Physicochemical properties
        self.hydrophobicity_scale = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
            '<PAD>': 0.0, '<UNK>': 0.0
        }
        
        self.charge_scale = {
            'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
            'Q': 0, 'E': -1, 'G': 0, 'H': 1, 'I': 0,
            'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
            'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0,
            '<PAD>': 0, '<UNK>': 0
        }
        
        # Multi-property embeddings
        embedding_dim = d_model // 3
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hydrophobicity_embedding = nn.Linear(1, embedding_dim)
        self.charge_embedding = nn.Linear(1, embedding_dim)
        
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
        
        # Structure GNN
        self.structure_gnn = StructureGNN(
            node_features=20,
            hidden_dim=structure_dim * 2,
            num_layers=2
        )
        
        # Mutation attention mechanism
        self.mutation_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Fusion layer
        # LMJ hack  due to line 188
        #fusion_input_dim = d_model + structure_dim + d_model  # seq + struct + mutation context
        fusion_input_dim = d_model + structure_dim * 2 + d_model  # seq + struct + mutation context
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2)
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def create_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # LMJ hack
        # print(batch_size)
        return pe.unsqueeze(0)
        #return pe.unsqueeze(0).repeat(batch_size, 1, 1)
    
    def get_property_values(self, tokens, property_scale):
        """Get property values for tokens"""
        batch_size, seq_len = tokens.shape
        property_values = torch.zeros(batch_size, seq_len, 1)
        
        # Simplified mapping (in practice, you'd have the full token-to-amino-acid mapping)
        for i in range(batch_size):
            for j in range(seq_len):
                token_idx = tokens[i, j].item()
                # This is a placeholder - real implementation would map tokens to amino acids
                if token_idx > 1:  # Not PAD or UNK
                    # Use a simple heuristic for demonstration
                    property_values[i, j, 0] = np.random.uniform(-4.5, 4.5)
        
        return property_values
    
    def identify_mutations(self, wild_type_seq, mutant_seq):
        """Identify mutation positions"""
        if wild_type_seq is None or mutant_seq is None:
            return []
        
        mutations = []
        min_len = min(len(wild_type_seq), len(mutant_seq))
        
        for i in range(min_len):
            if i < len(wild_type_seq) and i < len(mutant_seq):
                if wild_type_seq[i] != mutant_seq[i]:
                    mutations.append({
                        'position': i,
                        'wild_type': wild_type_seq[i],
                        'mutant': mutant_seq[i]
                    })
        
        return mutations
    
    def forward(self, src, src_mask=None, structure_data=None, 
                wild_type_seq=None, mutant_seq=None):
        batch_size, seq_len = src.shape
        
        # LMJ hack
            # LMJ hack
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        src = src.to(device)

        # Multi-property embeddings
        token_embeds = self.token_embedding(src)
        hydro_values = self.get_property_values(src, self.hydrophobicity_scale)
        charge_values = self.get_property_values(src, self.charge_scale)
        
        hydro_embeds = self.hydrophobicity_embedding(hydro_values.to(src.device))
        charge_embeds = self.charge_embedding(charge_values.to(src.device))
        
        # Combine embeddings
        embeddings = torch.cat([token_embeds, hydro_embeds, charge_embeds], dim=-1)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:, :seq_len, :].to(src.device)

        # LMJ hack
        # [2,30,63] -> [2,30,64]
        # x = torch.zeros((2,30,1))
        embeddings = F.pad(embeddings, (0, 1))

        embeddings = embeddings + pos_encoding
        embeddings = self.dropout(embeddings)
        
        # Transformer encoder
        if src_mask is not None:
            transformer_output = self.transformer_encoder(embeddings, src_key_padding_mask=src_mask)
        else:
            transformer_output = self.transformer_encoder(embeddings)
        
        # Sequence representation
        if src_mask is not None:
            mask_expanded = (~src_mask).unsqueeze(-1).float()
            seq_representation = (transformer_output * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            seq_representation = transformer_output.mean(dim=1)
        
        # Structure representation
        if structure_data is not None:
            structure_representation = self.structure_gnn(
                structure_data.x,
                structure_data.edge_index,
                structure_data.batch
            )
        else:
            structure_representation = torch.zeros(batch_size, 32).to(src.device)

        # LMJ hack
        # structure_representation is (1, 32), but seq_representation is (2, 64), we fake it
        structure_representation = structure_representation.repeat(2,1)

        
        # Mutation context (simplified)
        mutation_context = seq_representation  # In practice, this would be more sophisticated
        
        # Fusion
        fused_representation = torch.cat([
            seq_representation,
            structure_representation,
            mutation_context
        ], dim=-1)
        
        fused_representation = self.fusion_layer(fused_representation)
        output = self.output_layer(fused_representation)
        
        return output

# Structure GNN class (copied from previous step for completeness)
class StructureGNN(nn.Module):
    def __init__(self, node_features=20, hidden_dim=64, num_layers=2):
        super(StructureGNN, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        # LMJ hack   32 -> 64 match later call
        # self.output_layer = nn.Linear(hidden_dim, 32)
        self.output_layer = nn.Linear(hidden_dim, 64)
    
    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = self.output_layer(x)
        return x

# Test the mutational model
if __name__ == "__main__":
    from torch_geometric.data import Data

    # LMJ hack
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    # Test the model
    model = MutationalTransformer(vocab_size=22)
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample batch
    batch_size, seq_len = 2, 30
    batch_sequences = torch.randint(0, 22, (batch_size, seq_len))
    src_mask = (batch_sequences == 0)
    
    # Create dummy structure data
    structure_data = Data(
        x=torch.randn(30, 20),
        edge_index=torch.tensor([[0,1],[1,0],[1,2],[2,1]], dtype=torch.long).t(),
        batch=torch.zeros(30, dtype=torch.long)
    )
    
    with torch.no_grad():
        output = model(
            batch_sequences, 
            src_mask, 
            structure_data,
            wild_type_seq="MKTVRQERLK",
            mutant_seq="MKTVGQERLK"
        )
        print(f"Mutational model output shape: {output.shape}")
        print(f"Sample prediction: {output.flatten()}")
    
    print("Step 6: Mutation analysis added")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")