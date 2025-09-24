import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from Bio import SeqIO
from Bio.PDB import PDBParser
import requests
import os

def load_ddg_data(csv_file):
    """
    Load DDG data from CSV file
    Expected format: pdbid, chainid, variant, score
    """
    df = pd.read_csv(csv_file)
    
    # Example expected structure:
    # pdbid | chainid | variant | score
    # 1ABC  | A       | A123T   | 1.2
    # 2XYZ  | B       | G456V   | -0.5
    
    print(f"Loaded {len(df)} entries")
    print(df.head())
    
    return df

# Load your data
ddg_df = load_ddg_data('ddg.csv')


def get_sequence_from_pdb(pdb_id, chain_id):
    """
    Extract sequence from PDB file
    """
    try:
        # Download PDB file
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        response = requests.get(pdb_url)
        
        if response.status_code == 200:
            # Parse PDB and extract sequence
            from io import StringIO
            from Bio.PDB import PDBParser
            
            parser = PDBParser()
            structure = parser.get_structure(pdb_id, StringIO(response.text))
            
            sequence = ""
            for chain in structure[0]:  # First model
                if chain.id == chain_id:
                    for residue in chain:
                        if residue.get_resname() in amino_acids_3to1:
                            sequence += amino_acids_3to1[residue.get_resname()]
            
            return sequence
        else:
            return None
    except:
        return None

# Amino acid mapping
amino_acids_3to1 = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
    'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
    'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
    'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
}

def extract_sequences(df):
    """
    Extract sequences for all PDB entries
    """
    sequences = []
    
    for _, row in df.iterrows():
        pdb_id = row['pdbid']
        chain_id = row['chainid']
        
        sequence = get_sequence_from_pdb(pdb_id, chain_id)
        sequences.append(sequence)
    
    df['sequence'] = sequences
    return df

# Extract sequences (this may take time for large datasets)
ddg_df = extract_sequences(ddg_df)

def parse_variant(variant_str):
    """
    Parse variant string like 'A123T' into components
    """
    if len(variant_str) >= 3:
        wild_type = variant_str[0]
        position = int(variant_str[1:-1])
        mutant_type = variant_str[-1]
        return wild_type, position, mutant_type
    return None, None, None

def add_mutation_info(df):
    """
    Add parsed mutation information to dataframe
    """
    wild_types, positions, mutant_types = [], [], []
    
    for variant in df['variant']:
        wt, pos, mt = parse_variant(variant)
        wild_types.append(wt)
        positions.append(pos)
        mutant_types.append(mt)
    
    df['wild_type'] = wild_types
    df['position'] = positions
    df['mutant_type'] = mutant_types
    
    return df

ddg_df = add_mutation_info(ddg_df)

# Using ProtBERT or ESM models
def setup_tokenizer(model_name="Rostlab/prot_bert"):
    """
    Setup protein transformer tokenizer
    """
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Add special tokens if needed
    special_tokens = {"pad_token": "[PAD]", "mask_token": "[MASK]"}
    tokenizer.add_special_tokens(special_tokens)
    
    return tokenizer

tokenizer = setup_tokenizer()

def create_mutation_sequence(wild_seq, position, mutant_aa):
    """
    Create mutated sequence for training
    """
    if position > len(wild_seq) or position < 1:
        return None
    
    mutated_seq = list(wild_seq)
    mutated_seq[position - 1] = mutant_aa  # Convert to 0-indexed
    return ''.join(mutated_seq)

def prepare_training_data(df, tokenizer, max_length=512):
    """
    Prepare training tensors for protein transformer
    """
    input_ids_list = []
    attention_masks_list = []
    labels_list = []
    
    for _, row in df.iterrows():
        wild_seq = row['sequence']
        position = row['position']
        mutant_aa = row['mutant_type']
        ddg_score = row['score']
        
        if pd.isna(wild_seq) or wild_seq is None:
            continue
            
        # Create mutated sequence
        mutated_seq = create_mutation_sequence(wild_seq, position, mutant_aa)
        if mutated_seq is None:
            continue
        
        # Tokenize wild-type sequence
        wild_tokens = tokenizer(
            wild_seq,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize mutated sequence
        mutant_tokens = tokenizer(
            mutated_seq,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # You can choose to use wild-type or mutant sequence
        # For DDG prediction, you might want to encode the mutation information
        input_ids_list.append(wild_tokens['input_ids'].squeeze(0))
        attention_masks_list.append(wild_tokens['attention_mask'].squeeze(0))
        labels_list.append(ddg_score)
    
    return (
        torch.stack(input_ids_list),
        torch.stack(attention_masks_list),
        torch.tensor(labels_list, dtype=torch.float)
    )

# Prepare training data
input_ids, attention_masks, labels = prepare_training_data(ddg_df, tokenizer)

def create_mutation_encoded_sequence(wild_seq, position, mutant_aa, mask_token="[MASK]"):
    """
    Create sequence with mutation information encoded
    """
    if position > len(wild_seq) or position < 1:
        return None
    
    seq_list = list(wild_seq)
    original_aa = seq_list[position - 1]  # 0-indexed
    
    # Replace with mask token and add mutation info
    seq_list[position - 1] = mask_token
    enhanced_seq = ''.join(seq_list) + f"[MUTATION:{original_aa}{position}{mutant_aa}]"
    
    return enhanced_seq

def prepare_enhanced_training_data(df, tokenizer, max_length=512):
    """
    Enhanced training data with mutation information
    """
    input_ids_list = []
    attention_masks_list = []
    labels_list = []
    
    for _, row in df.iterrows():
        wild_seq = row['sequence']
        position = row['position']
        mutant_aa = row['mutant_type']
        ddg_score = row['score']
        
        if pd.isna(wild_seq) or wild_seq is None:
            continue
        
        # Create enhanced sequence with mutation info
        enhanced_seq = create_mutation_encoded_sequence(wild_seq, position, mutant_aa)
        if enhanced_seq is None:
            continue
        
        # Tokenize
        tokens = tokenizer(
            enhanced_seq,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids_list.append(tokens['input_ids'].squeeze(0))
        attention_masks_list.append(tokens['attention_mask'].squeeze(0))
        labels_list.append(ddg_score)
    
    return (
        torch.stack(input_ids_list),
        torch.stack(attention_masks_list),
        torch.tensor(labels_list, dtype=torch.float)
    )

# Use enhanced data preparation
input_ids, attention_masks, labels = prepare_enhanced_training_data(ddg_df, tokenizer)

from torch.utils.data import Dataset, DataLoader

class DDGDataset(Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

# Create dataset
dataset = DDGDataset(input_ids, attention_masks, labels)

# Split into train/validation
from sklearn.model_selection import train_test_split
train_idx, val_idx = train_test_split(
    range(len(dataset)), 
    test_size=0.2, 
    random_state=42
)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)

from transformers import AutoModel, AutoConfig
import torch.nn as nn

class DDGPredictionModel(nn.Module):
    def __init__(self, model_name="Rostlab/prot_bert", dropout_rate=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation or mean pooling
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        # Or use mean pooling: pooled_output = outputs.last_hidden_state.mean(dim=1)
        
        pooled_output = self.dropout(pooled_output)
        ddg_pred = self.regressor(pooled_output)
        
        return ddg_pred

# Initialize model
model = DDGPredictionModel()

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

def train_model(model, train_loader, val_loader, epochs=10):
    """
    Training loop for DDG prediction
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            predictions = model(input_ids, attention_mask).squeeze()
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                predictions = model(input_ids, attention_mask).squeeze()
                loss = criterion(predictions, labels)
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss/len(train_loader):.4f}")
        print(f"Val Loss: {val_loss/len(val_loader):.4f}")

# Train the model
train_model(model, train_loader, val_loader)

def predict_ddg(model, sequence, position, mutant_aa, tokenizer):
    """
    Predict DDG for a given mutation
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Create enhanced sequence
    enhanced_seq = create_mutation_encoded_sequence(sequence, position, mutant_aa)
    
    # Tokenize
    tokens = tokenizer(
        enhanced_seq,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = tokens['input_ids'].to(device)
    attention_mask = tokens['attention_mask'].to(device)
    
    with torch.no_grad():
        prediction = model(input_ids, attention_mask)
    
    return prediction.cpu().item()

# Example usage
predicted_ddg = predict_ddg(
    model, 
    "MKVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL", 
    69, 
    "A", 
    tokenizer
)
print(f"Predicted ΔΔG: {predicted_ddg:.2f} kcal/mol")

