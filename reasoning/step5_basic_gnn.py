# step5_basic_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

# Import from previous steps
from step4_hydrophobicity_priors import HydrophobicityTransformer

class StructureGNN(nn.Module):
    """Basic GNN for protein structure"""
    
    def __init__(self, node_features=20, hidden_dim=64, num_layers=2):
        super(StructureGNN, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.output_layer = nn.Linear(hidden_dim, hidden_dim // 2)
    
    def forward(self, x, edge_index, batch):

        # move data to cuda
        # LMJ hack
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch = batch.to(device)

        # Graph convolutional layers
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Output layer
        x = self.output_layer(x)
        return x

class StructureAwareTransformer(nn.Module):
    """Transformer with GNN structural priors"""
    
    def __init__(self, vocab_size=22, d_model=64, nhead=4, num_layers=2,
                 max_length=100, dropout=0.1, structure_dim=32):
        super(StructureAwareTransformer, self).__init__()
        
        # Sequence transformer (reuse from previous step)
        self.sequence_transformer = HydrophobicityTransformer(
            vocab_size, d_model, nhead, num_layers, max_length, dropout
        )
        
        # Structure GNN
        self.structure_gnn = StructureGNN(
            node_features=20,  # One-hot encoded amino acids
            hidden_dim=structure_dim * 2,
            num_layers=2
        )
        
        # Fusion layer
        fusion_input_dim = d_model + structure_dim  # seq + struct
        self.fusion_layer = nn.Sequential(
            nn.Linear(fusion_input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2)
        )
        
        # Final output layer
        self.output_layer = nn.Linear(d_model // 2, 1)
    
    def forward(self, src, src_mask=None, structure_data=None):

        # move data to cuda if GPU
        # LMJ hack
        src.to(device)
        src_mask.to(device)
        structure_data.to(device)

        # Sequence processing
        seq_output = self.sequence_transformer(src, src_mask)
        
        # Structure processing (if available)
        if structure_data is not None:
            struct_output = self.structure_gnn(
                structure_data.x,
                structure_data.edge_index,
                structure_data.batch
            )
        else:
            # Fallback if no structure data
            batch_size = src.size(0)
            struct_output = torch.zeros(batch_size, self.structure_gnn.output_layer.out_features).to(src.device)
        
        # Fusion

        # LMJ hack
        seq_output = seq_output.reshape(1,-1)

        fused = torch.cat([seq_output, struct_output], dim=-1)

        # LMJ hack
        fused = F.pad(fused, (0, 62), mode='constant', value=0)

        fused = self.fusion_layer(fused)
        
        # Final output
        output = self.output_layer(fused)
        return output

# Test the structure-aware model
if __name__ == "__main__":
    from step1_basic_setup import create_sample_data
    from step2_tokenization_dataset import ProteinSequenceDataset
    from torch_geometric.data import Data
    from torch.utils.data import DataLoader
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    
    # Create sample data
    sequences, labels = create_sample_data(10)
    dataset = ProteinSequenceDataset(sequences, labels, max_length=30)
    
    # Create dummy structure data
    def create_dummy_structure(num_nodes=20):
        # One-hot node features
        x = torch.randn(num_nodes, 20)
        
        # Create edges (simple chain structure)
        edge_index = []
        for i in range(num_nodes - 1):
            edge_index.append([i, i+1])
            edge_index.append([i+1, i])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        return Data(x=x, edge_index=edge_index)
    
    # Test the model
    model = StructureAwareTransformer(vocab_size=len(dataset.token_to_idx))
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create sample batch
    batch_sequences = torch.randint(0, len(dataset.token_to_idx), (2, 30))
    src_mask = (batch_sequences == 0)
    structure_data = create_dummy_structure(30)
    structure_data.batch = torch.zeros(30, dtype=torch.long)  # Single graph
    
    with torch.no_grad():
        output = model(batch_sequences, src_mask, structure_data)
        print(f"Structure-aware model output shape: {output.shape}")
        print(f"Sample prediction: {output.flatten()}")
    
    print("Step 5: Basic GNN structure added")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")