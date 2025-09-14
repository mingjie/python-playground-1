# step7_training_pipeline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Import from previous steps
from step6_mutation_analysis import MutationalTransformer
from step2_tokenization_dataset import ProteinSequenceDataset

def train_model(model, train_loader, val_loader, num_epochs=10, learning_rate=1e-3):
    """Training pipeline for the mutational model"""
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        train_samples = 0
        
        for batch_idx, batch in enumerate(train_loader):
            sequences = batch['sequence'].to(model.token_embedding.weight.device)
            labels = batch['label'].to(model.token_embedding.weight.device)
            
            # Create padding mask
            src_mask = (sequences == 0)
            
            optimizer.zero_grad()
            outputs = model(sequences, src_mask)
            
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * sequences.size(0)
            train_samples += sequences.size(0)
        
        avg_train_loss = total_train_loss / train_samples
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        val_samples = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch['sequence'].to(model.token_embedding.weight.device)
                labels = batch['label'].to(model.token_embedding.weight.device)
                src_mask = (sequences == 0)
                
                outputs = model(sequences, src_mask)
                loss = criterion(outputs.squeeze(), labels.float())
                
                total_val_loss += loss.item() * sequences.size(0)
                val_samples += sequences.size(0)
                
                all_preds.extend(outputs.squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = total_val_loss / val_samples
        val_losses.append(avg_val_loss)
        
        val_mse = mean_squared_error(all_labels, all_preds)
        val_r2 = r2_score(all_labels, all_preds)
        
        scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        print(f'Val MSE: {val_mse:.4f}, Val R²: {val_r2:.4f}')
        print('-' * 40)
    
    return train_losses, val_losses

def evaluate_model(model, test_loader):
    """Evaluate the trained model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequence'].to(model.token_embedding.weight.device)
            labels = batch['label'].to(model.token_embedding.weight.device)
            src_mask = (sequences == 0)
            
            outputs = model(sequences, src_mask)
            all_preds.extend(outputs.squeeze().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    mse = mean_squared_error(all_labels, all_preds)
    r2 = r2_score(all_labels, all_preds)
    correlation = np.corrcoef(all_labels, all_preds)[0, 1]
    
    print(f'Test Results:')
    print(f'MSE: {mse:.4f}')
    print(f'R²: {r2:.4f}')
    print(f'Correlation: {correlation:.4f}')
    
    return all_preds, all_labels

# Test the complete pipeline
if __name__ == "__main__":
    from step1_basic_setup import create_sample_data
    
    # LMJ hack
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create larger sample dataset
    sequences, labels = create_sample_data(200)
    
    # Split data
    train_sequences, test_sequences = sequences[:140], sequences[140:]
    train_labels, test_labels = labels[:140], labels[140:]
    val_sequences, test_sequences = test_sequences[:30], test_sequences[30:]
    val_labels, test_labels = test_labels[:30], test_labels[30:]
    
    # Create datasets
    train_dataset = ProteinSequenceDataset(train_sequences, train_labels, max_length=50)
    val_dataset = ProteinSequenceDataset(val_sequences, val_labels, max_length=50)
    test_dataset = ProteinSequenceDataset(test_sequences, test_labels, max_length=50)
    
    # Create data loaders
    # LMJ hack 
    # train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    # test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    
    # Initialize model
    model = MutationalTransformer(vocab_size=len(train_dataset.token_to_idx))
    
    # Train model
    print("Starting training...")
    train_losses, val_losses = train_model(model, train_loader, val_loader, num_epochs=5)
    
    # Evaluate model
    print("\nEvaluating model...")
    predictions, true_labels = evaluate_model(model, test_loader)
    
    print("Step 7: Complete training pipeline implemented")
    print(f"Final train loss: {train_losses[-1]:.4f}")
    print(f"Final val loss: {val_losses[-1]:.4f}")