import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# 1. Load ESM-2 Model and Tokenizer
# ==============================
def load_esm_model(model_name="facebook/esm2_t33_650M_UR50D"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    return tokenizer, model

tokenizer, model = load_esm_model()

# ==============================
# 2. Helper: Extract Embeddings and Attention
# ==============================
def get_embeddings_and_attentions(sequence, tokenizer, model):
    # Tokenize input
    inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True, output_hidden_states=True)
    
    # Last hidden state (batch, seq_len, dim)
    embeddings = outputs.last_hidden_state[0].cpu().numpy()  # Shape: (L+2, dim)
    
    # Attention from last layer (optional: use 1st head of last layer for analysis)
    attentions = outputs.attentions[-1][0].cpu().numpy()  # Shape: (num_heads, L+2, L+2)
    
    # Remove CLS and SEP tokens if needed (positions 0 and -1)
    residues_only_embeddings = embeddings[1:-1]  # Exclude <cls> and <eos>
    return residues_only_embeddings, attentions, inputs['input_ids'][0]

# ==============================
# 3. Apply Mutation to Sequence
# ==============================
def apply_mutation(sequence, mutation_str):
    """
    Apply a mutation in the format 'A56V' to the sequence.
    Returns mutated sequence.
    """
    aa_ref = mutation_str[0]
    pos = int(mutation_str[1:-1]) - 1  # 1-indexed to 0-indexed
    aa_mut = mutation_str[-1]
    
    if pos >= len(sequence) or pos < 0:
        raise ValueError(f"Position {pos+1} out of range for sequence length {len(sequence)}")
    if sequence[pos] != aa_ref:
        raise ValueError(f"Expected {aa_ref} at position {pos+1}, found {sequence[pos]}")
    
    new_seq = list(sequence)
    new_seq[pos] = aa_mut
    return ''.join(new_seq), pos

# ==============================
# 4. Compute Embedding Shift (ΔE)
# ==============================
def compute_embedding_shift(wt_seq, mut_seq, tokenizer, model):
    wt_embs, wt_attn, _ = get_embeddings_and_attentions(wt_seq, tokenizer, model)
    mut_embs, mut_attn, _ = get_embeddings_and_attentions(mut_seq, tokenizer, model)
    
    # Compute per-residue cosine difference
    delta_embedding = []
    for i in range(len(wt_embs)):
        d = 1 - cosine(wt_embs[i], mut_embs[i])  # Cosine similarity
        delta_embedding.append(d)
    
    return delta_embedding, wt_attn, mut_attn

# ==============================
# 5. Analyze Attention Changes (Optional Visualization)
# ==============================
def plot_attention_change(wt_attn, mut_attn, mutation_pos, head=0):
    # Average attention over all heads or pick one
    wt_avg = wt_attn[head, :, :].mean(axis=0)  # You can change aggregation
    mut_avg = mut_attn[head, :, :].mean(axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    sns.heatmap(wt_avg[:, :], ax=ax[0], cmap="Blues", cbar=True)
    ax[0].set_title(f"Wild-type Attention (Head {head})")
    ax[0].axvline(x=mutation_pos + 1, color='red', linestyle='--')  # +1 for <cls>
    ax[0].axhline(y=mutation_pos + 1, color='red', linestyle='--')

    sns.heatmap(mut_avg[:, :], ax=ax[1], cmap="Blues", cbar=True)
    ax[1].set_title(f"Mutant Attention (Head {head})")
    ax[1].axvline(x=mutation_pos + 1, color='red', linestyle='--')
    ax[1].axhline(y=mutation_pos + 1, color='red', linestyle='--')

    plt.tight_layout()
    plt.show()

# ==============================
# 6. Generate Reasoning Explanation
# ==============================
def generate_reasoning_explanation(sequence, mutation_str, delta_embedding, mutation_pos):
    # Simple heuristic-based explanation generator
    max_shift = max(delta_embedding)
    mean_shift = sum(delta_embedding) / len(delta_embedding)
    relative_shift = max_shift - mean_shift

    # Positional context
    seq_len = len(sequence)
    if mutation_pos < seq_len * 0.1 or mutation_pos > seq_len * 0.9:
        location = "near the termini"
        loc_effect = "potentially destabilizing or affecting folding kinetics"
    elif seq_len * 0.3 <= mutation_pos <= seq_len * 0.7:
        location = "in the central core"
        loc_effect = "likely affects structural stability"
    else:
        location = "in a loop or linker region"
        loc_effect = "may affect flexibility or domain motion"

    # Charge/hydrophobicity rules (simplified)
    charged = set('RKDE')
    hydrophobic = set('AVLIFWPM')
    wt_aa = mutation_str[0]
    mut_aa = mutation_str[-1]

    charge_change = (wt_aa in charged) != (mut_aa in charged)
    hydro_change = (wt_aa in hydrophobic) != (mut_aa in hydrophobic)

    explanation = f"The mutation {mutation_str} likely has a significant effect "

    if relative_shift > 0.1:
        explanation += "because the model detects a strong local perturbation in its evolutionary context. "
    else:
        explanation += "with minimal predicted impact based on evolutionary modeling. "

    explanation += f"It occurs {location}, which could be {loc_effect}. "

    if charge_change:
        explanation += "The change in charge may disrupt electrostatic interactions or salt bridges. "
    if hydro_change:
        explanation += "The switch between hydrophobic and hydrophilic character could misfold the core or expose buried residues. "

    if "A" in mutation_str or "G" in mutation_str:
        explanation += "Introduction of glycine or alanine may increase flexibility or reduce side-chain packing. "

    # Final prediction
    if relative_shift > 0.15:
        effect = "destabilizing or function-disrupting"
    elif relative_shift < 0.05:
        effect = "neutral or tolerated"
    else:
        effect = "moderately impactful"

    explanation += f"Overall, this mutation is predicted to be {effect}."

    return explanation

# ==============================
# 7. Full Pipeline Example
# ==============================
def analyze_mutation(sequence, mutation_str):
    print(f"Analyzing mutation: {mutation_str} in enzyme (length: {len(sequence)})\n")
    
    # Apply mutation
    try:
        mut_seq, mut_pos = apply_mutation(sequence, mutation_str)
    except ValueError as e:
        print("Error:", e)
        return
    
    # Compute embedding shift
    delta_embedding, wt_attn, mut_attn = compute_embedding_shift(sequence, mut_seq, tokenizer, model)
    
    # Generate reasoning
    explanation = generate_reasoning_explanation(sequence, mutation_str, delta_embedding, mut_pos)
    
    print("🔍 Reasoning Explanation:")
    print(explanation)
    
    print(f"\n📊 Max embedding shift at position {delta_embedding.index(max(delta_embedding)) + 1}: {max(delta_embedding):.3f}")
    print(f"Average embedding shift: {sum(delta_embedding)/len(delta_embedding):.3f}")
    
    # Optional: Plot attention changes
    plot_attention_change(wt_attn, mut_attn, mut_pos, head=0)
    
    return {
        'mutation': mutation_str,
        'explanation': explanation,
        'max_embedding_shift': max(delta_embedding),
        'average_embedding_shift': sum(delta_embedding)/len(delta_embedding),
        'delta_embedding': delta_embedding
    }

# ==============================
# 8. Example Usage
# ==============================
if __name__ == "__main__":
    # Example: TEM-1 β-lactamase (partial sequence for demo)
    tem1_sequence = "MKKLLAVLCLVLLALVQSQVEQPTESQRQQGYTQIYQGIDYVYSPVNPEDGKVSIVWSRDQINGGWDAIQAGIRDFLVYQQQQQQQLQQLQQLQQLQQ"
    
    result = analyze_mutation(tem1_sequence, "D219G")  # Known resistance mutation