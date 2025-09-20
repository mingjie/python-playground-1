# train_pdb_gnn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.loader import DataLoader as PyGDataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import pandas as pd
import os
import glob
from tqdm import tqdm
import argparse
from datetime import datetime

# Import PDB processor
from process_single_pdb import PDBGraphProcessor

class StructureGNN(nn.Module):
    """GNN for protein structure learning"""
    
    def __init__(self, node_features=24, hidden_dim=128, num_layers=4, 
                 dropout=0.1, task='embedding'):
        super(StructureGNN, self).__init__()
        
        self.node_features = node_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.task = task
        
        # Graph convolutional layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(node_features, hidden_dim))
        
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])
        
        # Global pooling
        self.global_pool = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # mean + max pooling
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        if task == 'embedding':
            self.output_layer = nn.Linear(hidden_dim, hidden_dim // 2)
        elif task == 'stability':
            self.regressor = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        elif task == 'classification':
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 2)
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutional layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Global pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pooled = torch.cat([x_mean, x_max], dim=1)
        x_pooled = self.global_pool(x_pooled)
        
        # Task-specific output
        if self.task == 'embedding':
            output = self.output_layer(x_pooled)
        elif self.task == 'stability':
            output = self.regressor(x_pooled)
        elif self.task == 'classification':
            output = self.classifier(x_pooled)
        
        return output

class PDBDataset(torch.utils.data.Dataset):
    """Dataset for PDB graph data"""
    
    def __init__(self, graph_data_list, labels=None, task='embedding'):
        self.graph_data_list = graph_data_list
        self.labels = labels
        self.task = task
    
    def __len__(self):
        return len(self.graph_data_list)
    
    def __getitem__(self, idx):
        data = self.graph_data_list[idx]
        
        if self.labels is not None:
            data.label = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return data

class GNNTrainer:
    """Trainer for GNN models"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader, optimizer, criterion):
        self.model.train()
        total_loss = 0
        num_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            batch = batch.to(self.device)
            optimizer.zero_grad()
            
            output = self.model(batch)
            
            if hasattr(batch, 'label') and batch.label is not None:
                target = batch.label.to(self.device)
                if len(target.shape) == 1:
                    target = target.unsqueeze(1)
                loss = criterion(output, target)
            else:
                # Self-supervised loss (reconstruction or regularization)
                loss = self.self_supervised_loss(output, batch)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch.num_graphs
            num_samples += batch.num_graphs
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / num_samples
    
    def validate(self, val_loader, criterion):
        self.model.eval()
        total_loss = 0
        num_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            progress_bar = tqdm(val_loader, desc="Validation")
            for batch in progress_bar:
                batch = batch.to(self.device)
                output = self.model(batch)
                
                if hasattr(batch, 'label') and batch.label is not None:
                    target = batch.label.to(self.device)
                    if len(target.shape) == 1:
                        target = target.unsqueeze(1)
                    loss = criterion(output, target)
                    all_preds.extend(output.cpu().numpy())
                    all_labels.extend(target.cpu().numpy())
                else:
                    loss = self.self_supervised_loss(output, batch)
                
                total_loss += loss.item() * batch.num_graphs
                num_samples += batch.num_graphs
                
                progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_samples
        
        if all_preds and all_labels:
            mse = mean_squared_error(all_labels, all_preds)
            r2 = r2_score(all_labels, all_preds)
            return avg_loss, mse, r2
        else:
            return avg_loss, None, None
    
    def self_supervised_loss(self, embeddings, batch):
        """Simple self-supervised loss"""
        # Reconstruction loss or regularization
        return torch.mean(embeddings ** 2) * 0.01
    
    def train(self, train_loader, val_loader, num_epochs=50, lr=1e-3, 
              weight_decay=1e-5, save_path='best_gnn_model.pth'):
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
        
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validation
            val_results = self.validate(val_loader, criterion)
            val_loss, val_mse, val_r2 = val_results
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            print(f'Train Loss: {train_loss:.4f}')
            if val_mse is not None:
                print(f'Val Loss: {val_loss:.4f}, Val MSE: {val_mse:.4f}, Val R²: {val_r2:.4f}')
            else:
                print(f'Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                print(f'Best model saved with val loss: {val_loss:.4f}')
        
        print("Training completed!")

def process_pdb_directory(pdb_directory, max_files=1000, distance_threshold=8.0):
    """Process all PDB files in directory"""
    processor = PDBGraphProcessor(distance_threshold=distance_threshold)
    graph_data_list = []
    failed_files = []
    
    pdb_files = glob.glob(os.path.join(pdb_directory, "*.pdb"))
    pdb_files = pdb_files[:max_files]  # Limit for testing
    
    print(f"Found {len(pdb_files)} PDB files")
    
    for pdb_file in tqdm(pdb_files, desc="Processing PDB files"):
        try:
            graph_data = processor.pdb_to_graph(pdb_file)
            if graph_data is not None and graph_data.num_nodes > 10:  # Filter small proteins
                # Ensure batch attribute exists
                if not hasattr(graph_data, 'batch'):
                    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
                graph_data_list.append(graph_data)
            else:
                failed_files.append(pdb_file)
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            failed_files.append(pdb_file)
    
    print(f"Successfully processed {len(graph_data_list)} structures")
    if failed_files:
        print(f"Failed to process {len(failed_files)} files")
    
    return graph_data_list, failed_files

def create_labels_from_properties(graph_data_list, label_type='compactness'):
    """Create labels based on structural properties"""
    labels = []
    
    for data in graph_data_list:
        if label_type == 'compactness' and hasattr(data, 'coordinates'):
            try:
                coords = data.coordinates.cpu().numpy()
                if len(coords) > 1:
                    # Simple compactness measure
                    distances = []
                    for i in range(len(coords)):
                        for j in range(i+1, len(coords)):
                            dist = np.linalg.norm(coords[i] - coords[j])
                            distances.append(dist)
                    compactness = 1.0 / (np.mean(distances) + 1e-8)
                    labels.append(compactness)
                else:
                    labels.append(0.0)
            except:
                labels.append(0.0)
        elif label_type == 'size':
            labels.append(float(data.num_nodes))
        else:
            labels.append(0.0)
    
    return labels

def main(pdb_directory, output_dir='models', max_files=1000, epochs=50, 
         batch_size=8, task='embedding', label_type='compactness'):
    """Main training function"""
    
    print("=" * 60)
    print("Training GNN on PDB Structures")
    print("=" * 60)
    print(f"Directory: {pdb_directory}")
    print(f"Max files: {max_files}")
    print(f"Epochs: {epochs}")
    print(f"Task: {task}")
    print(f"Label type: {label_type}")
    print()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process PDB files
    print("Processing PDB files...")
    graph_data_list, failed_files = process_pdb_directory(
        pdb_directory, max_files=max_files
    )
    
    if len(graph_data_list) < 10:
        print("Not enough data for training. Need at least 10 structures.")
        return
    
    # Create labels if needed
    if task != 'embedding':
        print(f"Creating {label_type} labels...")
        labels = create_labels_from_properties(graph_data_list, label_type)
    else:
        labels = None
    
    # Create dataset
    print("Creating dataset...")
    dataset = PDBDataset(graph_data_list, labels, task=task)
    
    # Split data
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    print(f"Train set: {len(train_data)} samples")
    print(f"Validation set: {len(val_data)} samples")
    
    # Create data loaders
    train_loader = PyGDataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = PyGDataLoader(val_data, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = StructureGNN(
        node_features=24,  # one-hot + physicochemical properties
        hidden_dim=128,
        num_layers=4,
        task=task
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize trainer
    trainer = GNNTrainer(model)
    
    # Train model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_save_path = os.path.join(output_dir, f'gnn_model_{timestamp}.pth')
    
    print("Starting training...")
    trainer.train(
        train_loader, val_loader, 
        num_epochs=epochs, 
        lr=1e-3,
        save_path=model_save_path
    )
    
    # Save training history
    history = {
        'train_losses': trainer.train_losses,
        'val_losses': trainer.val_losses,
        'best_val_loss': trainer.best_val_loss
    }
    np.save(os.path.join(output_dir, f'training_history_{timestamp}.npy'), history)
    
    print(f"\nModel saved to: {model_save_path}")
    print("Training completed successfully!")

# Utility functions for extracting embeddings
def extract_embeddings(model_path, pdb_directory, output_path, max_files=1000):
    """Extract embeddings for all PDB files using trained model"""
    
    print("Extracting embeddings...")
    
    # Load trained model
    model = StructureGNN(node_features=24, hidden_dim=128, num_layers=4, task='embedding')
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    model.to('cpu')
    
    # Process PDB files
    processor = PDBGraphProcessor()
    embeddings = {}
    failed_files = []
    
    pdb_files = glob.glob(os.path.join(pdb_directory, "*.pdb"))
    pdb_files = pdb_files[:max_files]
    
    print(f"Processing {len(pdb_files)} PDB files...")
    
    for pdb_file in tqdm(pdb_files, desc="Extracting embeddings"):
        try:
            graph_data = processor.pdb_to_graph(pdb_file)
            if graph_data is not None:
                # Ensure batch attribute exists
                if not hasattr(graph_data, 'batch'):
                    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
                
                with torch.no_grad():
                    embedding = model(graph_data)
                    embeddings[os.path.basename(pdb_file)] = embedding.cpu().numpy()
            else:
                failed_files.append(pdb_file)
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            failed_files.append(pdb_file)
    
    # Save embeddings
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path}")
    print(f"Extracted embeddings for {len(embeddings)} structures")
    
    if failed_files:
        print(f"Failed to process {len(failed_files)} files")
    
    return embeddings

# Example usage and testing
def test_training():
    """Test the training pipeline"""
    print("Testing training pipeline...")
    
    # This would be run with actual PDB files
    # For demonstration, we'll create mock data
    
    # Create mock graph data
    mock_graphs = []
    for i in range(20):
        graph = Data(
            x=torch.randn(50 + i*10, 24),  # Increasing node count
            edge_index=torch.randint(0, 50 + i*10, (2, 100 + i*20)),
            batch=torch.zeros(50 + i*10, dtype=torch.long)
        )
        graph.num_nodes = 50 + i*10
        mock_graphs.append(graph)
    
    # Create mock dataset
    dataset = PDBDataset(mock_graphs, task='embedding')
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    train_loader = PyGDataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = PyGDataLoader(val_data, batch_size=4, shuffle=False)
    
    # Initialize and train model
    model = StructureGNN(node_features=24, hidden_dim=64, num_layers=2, task='embedding')
    trainer = GNNTrainer(model)
    
    print("Running mock training (2 epochs)...")
    trainer.train(train_loader, val_loader, num_epochs=2, lr=1e-3)
    
    print("Mock training completed successfully!")

if __name__ == "__main__":
    # Example usage:
    # For training:
    main(
        pdb_directory='./reasoning/pdb2graph/data',
        output_dir='./reasoning/pdb2graph/models',
        max_files=1000,
        epochs=50,
        batch_size=8,
        task='embedding'  # or 'stability', 'classification'
    )
    
    """
    # For extracting embeddings:
    extract_embeddings(
        model_path='models/gnn_model_20241201_120000.pth',
        pdb_directory='/path/to/pdb/files',
        output_path='structure_embeddings.npy',
        max_files=1000
    )
    """
    
    """
    # Run test
    test_training()
    
    print("\nTraining pipeline ready!")
    print("Usage examples:")
    print("1. Training: main('/path/to/pdb/files', epochs=30, task='embedding')")
    print("2. Embedding extraction: extract_embeddings('model.pth', '/pdb/files', 'embeddings.npy')")
    """