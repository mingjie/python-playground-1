# hydrophobic_protein_lm.py
# A protein language model with hydrophobicity prior for enzyme design

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random

# === 1. Amino Acid Vocabulary ===
AA_VOCAB = [
    'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
    'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y',
    '<PAD>', '<UNK>', '<CLS>', '<MASK>'
]
aa_to_idx = {aa: idx for idx, aa in enumerate(AA_VOCAB)}
idx_to_aa = {idx: aa for aa, idx in aa_to_idx.items()}
VOCAB_SIZE = len(AA_VOCAB)
MAX_LEN = 128

# === 2. Hydrophobicity Scale (Kyte-Doolittle) ===
# Positive = hydrophobic, Negative = hydrophilic
HYDROPATHY = {
    'A': 1.8,  'C': 2.5,  'D': -3.5, 'E': -3.5, 'F': 2.8,
    'G': -0.4, 'H': -3.2, 'I': 4.5,  'K': -3.9, 'L': 3.8,
    'M': 1.9,  'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
    'S': -0.8, 'T': -0.7, 'V': 4.2,  'W': -0.9, 'Y': -1.3,
    # Default for special tokens
    '<PAD>': 0.0, '<UNK>': 0.0, '<CLS>': 0.0, '<MASK>': 0.0
}

# Normalize hydrophobicity (zero mean, unit variance)
values = np.array(list(HYDROPATHY.values()))
mean, std = values.mean(), values.std()
for aa in HYDROPATHY:
    HYDROPATHY[aa] = (HYDROPATHY[aa] - mean) / (std + 1e-8)

# Create hydrophobicity tensor
hydrophobicity_tensor = torch.tensor([HYDROPATHY[idx_to_aa[i]] for i in range(VOCAB_SIZE)], dtype=torch.float).unsqueeze(0).unsqueeze(0)
# Shape: (1, 1, VOCAB_SIZE) → will be used to embed positional hydrophobicity

# === 3. Dataset with MLM ===
class ProteinDataset(Dataset):
    def __init__(self, sequences, max_len=MAX_LEN, mask_prob=0.15):
        self.sequences = [seq for seq in sequences if len(seq) <= max_len]
        self.max_len = max_len
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        orig_len = len(seq)
        padded_seq = list(seq) + ['<PAD>'] * (self.max_len - orig_len)

        # Tokenize
        input_ids = [aa_to_idx.get(aa, aa_to_idx['<UNK>']) for aa in padded_seq]
        input_ids = [aa_to_idx['<CLS>']] + input_ids
        labels = [-100] * len(input_ids)

        # Randomly mask 15%
        mutable_indices = list(range(1, orig_len + 1))
        random.shuffle(mutable_indices)
        num_mask = int(self.mask_prob * len(mutable_indices))

        for i in mutable_indices[:num_mask]:
            labels[i] = input_ids[i]
            if random.random() < 0.8:
                input_ids[i] = aa_to_idx['<MASK>']
            elif random.random() < 0.5:
                input_ids[i] = random.randint(0, 19)  # random AA

        return torch.tensor(input_ids), torch.tensor(labels)

# === 4. Hydrophobicity-Enhanced Transformer Model ===
class HydrophobicProteinLM(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, num_heads=4, num_layers=3, max_len=MAX_LEN + 1, dropout=0.1, hydrophobic_scale=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.hydrophobic_scale = hydrophobic_scale

        # Standard token and positional embeddings
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable hydrophobicity projection
        self.hydrophobic_proj = nn.Linear(1, embed_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.lm_head.weight, std=0.02)
        nn.init.zeros_(self.lm_head.bias)

    def forward(self, input_ids, labels=None):
        device = input_ids.device
        b, seq_len = input_ids.shape

        # Token + Positional Embeddings
        token_emb = self.token_embed(input_ids)  # (b, seq_len, d)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(b, -1)
        pos_emb = self.pos_embed(pos_ids)
        h = token_emb + pos_emb

        # === Add Hydrophobicity Prior ===
        # Get hydrophobicity values for each token
        with torch.no_grad():
            hydrophobic_values = hydrophobicity_tensor.expand(b, seq_len, -1).gather(2, input_ids.unsqueeze(-1)).squeeze(-1)  # (b, seq_len)
            hydrophobic_values = hydrophobic_values.unsqueeze(-1) * self.hydrophobic_scale  # (b, seq_len, 1)

        # Project and add
        hydrophobic_emb = self.hydrophobic_proj(hydrophobic_values)  # (b, seq_len, d)
        h = h + hydrophobic_emb

        h = self.dropout(h)
        h = self.transformer(h)
        logits = self.lm_head(h)

        if labels is not None:
            loss = nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, VOCAB_SIZE), labels.view(-1)
            )
            return loss, logits

        return logits

# === 5. Training Loop ===
def train_model(sequences, epochs=10, batch_size=8, lr=5e-4):
    dataset = ProteinDataset(sequences)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = HydrophobicProteinLM()
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for input_ids, labels in dataloader:
            optimizer.zero_grad()
            loss, _ = model(input_ids, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, MLM Loss: {avg_loss:.4f}")
    return model

# === 6. Mutation Design with Hydrophobicity Awareness ===
@torch.no_grad()
def suggest_mutations(model, sequence, top_k=5):
    model.eval()
    orig_len = len(sequence)
    device = next(model.parameters()).device

    # Pad and tokenize
    padded = sequence + '<PAD>' * (MAX_LEN - orig_len)
    input_ids = [aa_to_idx['<CLS>']] + [aa_to_idx.get(aa, aa_to_idx['<UNK>']) for aa in padded]
    input_ids = torch.tensor([input_ids]).to(device)

    # Forward pass
    logits = model(input_ids)  # (1, L+1, V)
    log_probs = torch.log_softmax(logits, dim=-1)

    suggestions = []
    for pos in range(1, orig_len + 1):
        orig_idx = input_ids[0, pos].item()
        orig_aa = idx_to_aa[orig_idx]
        if orig_aa in ['<PAD>', '<UNK>']:
            continue

        orig_logp = log_probs[0, pos, orig_idx].item()
        for aa_idx in range(20):
            aa = idx_to_aa[aa_idx]
            if aa == orig_aa:
                continue
            delta_logp = log_probs[0, pos, aa_idx].item() - orig_logp

            # Bonus: encourage hydrophobic→hydrophobic in core-like context
            orig_h = HYDROPATHY[orig_aa]
            new_h = HYDROPATHY[aa]
            # If original was hydrophobic, favor hydrophobic replacements
            if orig_h > 0.5:
                if new_h > 0.5:
                    delta_logp += 0.1  # bonus
                elif new_h < -0.5:
                    delta_logp -= 0.2  # penalty

            suggestions.append({
                'mut': f"{orig_aa}{pos}{aa}",
                'score': delta_logp
            })

    # Sort by score
    suggestions.sort(key=lambda x: x['score'], reverse=True)
    return suggestions[:top_k]

# === 7. Example Usage ===
if __name__ == "__main__":
    # Example enzyme sequences
    sequences = [
        "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG",
        "WPYVANASLAAQLSGVTQRLPGMKYIQLIQSWRRVQLKVQHQAELALGQHDGQPLRGYLQGP",
        "GQYFKNYKQVVDGQDLLVMNNWQYVQQLQQLQQLQQLQQLQQLQQLQQLQQLQQLQQLQQLQQLQ",
    ] * 100

    print("Training hydrophobicity-aware protein model...")
    model = train_model(sequences, epochs=5)

    test_seq = "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVA"
    print("\n💡 Suggested mutations (hydrophobicity-aware):")
    for suggestion in suggest_mutations(model, test_seq, top_k=10):
        print(f"{suggestion['mut']}: score = {suggestion['score']:+.3f}")