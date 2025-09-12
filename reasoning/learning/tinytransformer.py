import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        # Model configuration
        self.vocab_size = 100
        self.hidden_size = 8      # Small hidden size for visualization
        self.num_heads = 2        # 2 attention heads
        self.seq_length = 6       # Short sequence
        self.intermediate_size = 16
        
        # Embedding layers
        self.token_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(self.seq_length, self.hidden_size)
        
        # Attention components
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_output = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Feed-forward network
        self.ffn_intermediate = nn.Linear(self.hidden_size, self.intermediate_size)
        self.ffn_output = nn.Linear(self.intermediate_size, self.hidden_size)
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(self.hidden_size)
        self.ln2 = nn.LayerNorm(self.hidden_size)
        
    def forward(self, input_ids):
        print("=== TINY TRANSFORMER FORWARD PASS ===\n")
        
        batch_size, seq_length = input_ids.shape
        print(f"Input IDs: {list(input_ids.shape)}")
        print(f"  Shape: (batch_size={batch_size}, seq_length={seq_length})")
        print(f"  Values: {input_ids}")
        print()
        
        # 1. TOKEN EMBEDDINGS
        token_embeds = self.token_embeddings(input_ids)
        print(f"1. Token Embeddings:")
        print(f"  Shape: {list(token_embeds.shape)}")
        print(f"  Values:\n{token_embeds}")
        print()
        
        # 2. POSITION EMBEDDINGS
        position_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        position_embeds = self.position_embeddings(position_ids)
        print(f"2. Position Embeddings:")
        print(f"  Shape: {list(position_embeds.shape)}")
        print(f"  Values:\n{position_embeds}")
        print()
        
        # 3. COMBINED EMBEDDINGS
        embeddings = token_embeds + position_embeds
        print(f"3. Combined Embeddings (Token + Position):")
        print(f"  Shape: {list(embeddings.shape)}")
        print(f"  Values:\n{embeddings}")
        print()
        
        # 4. MULTI-HEAD ATTENTION
        print("4. Multi-Head Attention:")
        
        # Project to Q, K, V
        Q = self.q_proj(embeddings)
        K = self.k_proj(embeddings)
        V = self.v_proj(embeddings)
        
        print(f"  Q Projection:")
        print(f"    Shape: {list(Q.shape)}")
        print(f"    Values:\n{Q}")
        print()
        
        print(f"  K Projection:")
        print(f"    Shape: {list(K.shape)}")
        print(f"    Values:\n{K}")
        print()
        
        print(f"  V Projection:")
        print(f"    Shape: {list(V.shape)}")
        print(f"    Values:\n{V}")
        print()
        
        # Split into heads
        head_dim = self.hidden_size // self.num_heads
        Q = Q.view(batch_size, seq_length, self.num_heads, head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_length, self.num_heads, head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_length, self.num_heads, head_dim).transpose(1, 2)
        
        print(f"  After Head Split:")
        print(f"    Q Shape: {list(Q.shape)} (batch, heads, seq, head_dim)")
        print(f"    K Shape: {list(K.shape)}")
        print(f"    V Shape: {list(V.shape)}")
        print()
        
        # Attention scores (QK^T)
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (head_dim ** 0.5)
        print(f"  Attention Scores (QK^T / sqrt(d_k)):")
        print(f"    Shape: {list(attention_scores.shape)}")
        print(f"    Values:\n{attention_scores}")
        print()
        
        # Apply softmax
        attention_probs = F.softmax(attention_scores, dim=-1)
        print(f"  Attention Probabilities (after softmax):")
        print(f"    Shape: {list(attention_probs.shape)}")
        print(f"    Values:\n{attention_probs}")
        print()
        
        # Apply attention to values
        attention_output = torch.matmul(attention_probs, V)
        print(f"  Attention Output (Attention × V):")
        print(f"    Shape: {list(attention_output.shape)}")
        print(f"    Values:\n{attention_output}")
        print()
        
        # Reshape back
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_size)
        print(f"  Reshaped Attention Output:")
        print(f"    Shape: {list(attention_output.shape)}")
        print(f"    Values:\n{attention_output}")
        print()
        
        # Final attention projection
        attention_output = self.attention_output(attention_output)
        print(f"  Final Attention Projection:")
        print(f"    Shape: {list(attention_output.shape)}")
        print(f"    Values:\n{attention_output}")
        print()
        
        # 5. RESIDUAL CONNECTION + LAYER NORM
        attention_output = self.ln1(embeddings + attention_output)
        print(f"5. After Attention + Residual + LayerNorm:")
        print(f"  Shape: {list(attention_output.shape)}")
        print(f"  Values:\n{attention_output}")
        print()
        
        # 6. FEED-FORWARD NETWORK
        print("6. Feed-Forward Network:")
        
        # Intermediate layer
        ffn_hidden = self.ffn_intermediate(attention_output)
        print(f"  FFN Intermediate (Linear + Activation):")
        print(f"    Shape: {list(ffn_hidden.shape)}")
        print(f"    Values:\n{F.gelu(ffn_hidden)}")
        print()
        
        # Output projection
        ffn_output = self.ffn_output(F.gelu(ffn_hidden))
        print(f"  FFN Output Projection:")
        print(f"    Shape: {list(ffn_output.shape)}")
        print(f"    Values:\n{ffn_output}")
        print()
        
        # 7. FINAL RESIDUAL CONNECTION + LAYER NORM
        final_output = self.ln2(attention_output + ffn_output)
        print(f"7. Final Output (FFN + Residual + LayerNorm):")
        print(f"  Shape: {list(final_output.shape)}")
        print(f"  Values:\n{final_output}")
        print()
        
        return final_output

# Example usage
def run_tiny_transformer():
    # Create model
    model = TinyTransformer()
    
    # Sample input: batch_size=1, seq_length=6
    input_ids = torch.tensor([[10, 25, 30, 45, 50, 60]])
    
    print("TENSOR DIMENSION FLOW THROUGH TRANSFORMER LAYERS")
    print("=" * 50)
    
    # Forward pass with tensor tracking
    output = model(input_ids)
    
    print("\nSUMMARY OF TENSOR SHAPES:")
    print("=" * 30)
    shapes = [
        "Input IDs:           (1, 6)",
        "Token Embeddings:    (1, 6, 8)",
        "Position Embeddings: (1, 6, 8)",
        "Combined Embeddings: (1, 6, 8)",
        "Q/K/V Projections:   (1, 6, 8)",
        "Head Split Q/K/V:    (1, 2, 6, 4)",
        "Attention Scores:    (1, 2, 6, 6)",
        "Attention Output:    (1, 6, 8)",
        "FFN Intermediate:    (1, 6, 16)",
        "Final Output:        (1, 6, 8)"
    ]
    
    for shape in shapes:
        print(shape)

# Run the example
if __name__ == "__main__":
    run_tiny_transformer()