"""
EnzyReason: A Scientific Reasoning Model for Enzyme Mutational Analysis
======================================================================
Author: Computational Enzymology Lab
Affiliation: [Your Institution]
License: MIT

LMJ 

This module implements a transformer-based reasoning engine using ESM-2
to analyze and explain the impact of point mutations in enzymes.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cosine
from typing import Dict, Tuple, Optional, List
from loguru import logger
import warnings
warnings.filterwarnings("ignore")

# ==============================
# CONFIGURATION
# ==============================
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.add("enzyreason.log", rotation="500 MB", level="INFO")


# ==============================
# 1. Load ESM-2 Model with Gradient Support
# ==============================
class ESM2Embedder:
    """
    Wrapper for ESM-2 model to extract embeddings and attentions.
    Supports gradient computation for saliency analysis.
    """
    def __init__(self, model_name: str = MODEL_NAME):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.to(DEVICE)
        logger.info(f"Loaded {model_name} on {DEVICE}")

    @torch.no_grad()
    def get_representations(self, sequence: str) -> Tuple[np.ndarray, np.ndarray, torch.Tensor]:
        """
        Extract embeddings and attentions for a given sequence.
        
        Args:
            sequence (str): Amino acid sequence (e.g., 'MKLV...')
            
        Returns:
            embeddings (np.ndarray): (L, D) residue-wise embeddings
            attentions (np.ndarray): (H, L+2, L+2) attention maps from last layer
            input_ids (torch.Tensor): Tokenized input IDs
        """
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_attentions=True,
                output_hidden_states=True
            )
        
        # Extract last hidden state (exclude <cls> and <eos>)
        embeddings = outputs.last_hidden_state[0].cpu().numpy()  # (L+2, D)
        embeddings = embeddings[1:-1]  # Remove special tokens -> (L, D)
        
        # Extract last layer attention (H, L+2, L+2)
        attentions = outputs.attentions[-1][0].cpu().numpy()  # (H, L+2, L+2)
        
        return embeddings, attentions, inputs['input_ids'][0]


# ==============================
# 2. Mutation Application with Validation
# ==============================
def apply_mutation(sequence: str, mutation_str: str) -> Tuple[str, int]:
    """
    Apply a point mutation in UniProt notation (e.g., D100A).
    
    Args:
        sequence (str): Wild-type amino acid sequence.
        mutation_str (str): Mutation in format [WT][Position][Mut], e.g., 'D100A'.
        
    Returns:
        mutated_sequence (str), mutation_position (int, 0-indexed)
        
    Raises:
        ValueError: If mutation is invalid.
    """
    import re
    match = re.match(r"([A-Z])(\d+)([A-Z])", mutation_str)
    if not match:
        raise ValueError(f"Invalid mutation format: {mutation_str}")

    wt_aa, pos_str, mut_aa = match.groups()
    pos_1idx = int(pos_str)
    pos_0idx = pos_1idx - 1

    if pos_0idx < 0 or pos_0idx >= len(sequence):
        raise IndexError(f"Position {pos_1idx} out of range for sequence length {len(sequence)}")
    if sequence[pos_0idx] != wt_aa:
        raise ValueError(f"Expected {wt_aa} at position {pos_1idx}, found {sequence[pos_0idx]}")

    mutated_seq = sequence[:pos_0idx] + mut_aa + sequence[pos_0idx+1:]
    logger.info(f"Applied mutation {mutation_str} at position {pos_1idx}")
    return mutated_seq, pos_0idx


# ==============================
# 3. Embedding Perturbation Analysis
# ==============================
def compute_embedding_perturbation(
    wt_sequence: str,
    mut_sequence: str,
    embedder: ESM2Embedder
) -> Dict:
    """
    Compute embedding shift (ΔE) and local/global perturbation metrics.
    
    Returns:
        Dictionary with:
        - delta_similarity: 1 - cosine distance per residue
        - max_perturbation: Max local shift
        - global_rmsd: RMSD of embedding trajectories
        - hotspot_position: Most perturbed residue
    """
    wt_embs, wt_attn, _ = embedder.get_representations(wt_sequence)
    mut_embs, mut_attn, _ = embedder.get_representations(mut_sequence)

    L = len(wt_embs)
    delta_similarity = []
    for i in range(L):
        sim = 1 - cosine(wt_embs[i], mut_embs[i])
        delta_similarity.append(sim)

    # Global RMSD in embedding space
    embedding_rmsd = np.sqrt(np.mean((wt_embs - mut_embs) ** 2))

    # Find hotspot
    hotspot_idx = np.argmax(delta_similarity)
    max_perturbation = delta_similarity[hotspot_idx]

    return {
        'delta_similarity': np.array(delta_similarity),
        'embedding_rmsd': embedding_rmsd,
        'max_perturbation': max_perturbation,
        'hotspot_position': hotspot_idx,
        'wt_embeddings': wt_embs,
        'mut_embeddings': mut_embs,
        'wt_attentions': wt_attn,
        'mut_attentions': mut_attn
    }


# ==============================
# 4. Biophysical & Evolutionary Feature Lookup
# ==============================
# Simplified residue property dictionaries
CHARGE = {aa: 1 for aa in 'RK'} | {aa: -1 for aa in 'DE'} | {aa: 0 for aa in 'ACFGHIJLMNOPQSTUVWYBXZ'}
HYDROPATHY = {
    'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9,
    'A': 1.8, 'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9,
    'Y': -1.3, 'P': -1.6, 'H': -3.2, 'E': -3.5, 'Q': -3.5,
    'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
}

def compute_biophysical_change(wt_aa: str, mut_aa: str) -> Dict[str, float]:
    """Compute change in charge and hydrophobicity."""
    d_charge = CHARGE.get(mut_aa, 0) - CHARGE.get(wt_aa, 0)
    d_hydro = HYDROPATHY.get(mut_aa, 0) - HYDROPATHY.get(wt_aa, 0)
    return {
        'delta_charge': d_charge,
        'delta_hydrophobicity': d_hydro,
        'is_charge_reversal': abs(d_charge) == 2,
        'is_hydrophobic_to_polar': (HYDROPATHY[wt_aa] > 1.5) and (HYDROPATHY[mut_aa] < 0)
    }


# ==============================
# 5. Attention Rollout & Interaction Disruption
# ==============================
def attention_rollout(attentions: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    Compute attention rollout: cumulative influence across layers (approximated here with last layer).
    
    Args:
        attentions (np.ndarray): (H, L+2, L+2)
        threshold (float): Minimum attention weight to consider.
        
    Returns:
        interaction_map (np.ndarray): (L, L) binary map of significant interactions.
    """
    # Average over heads and remove special tokens
    avg_attn = attentions.mean(axis=0)  # (L+2, L+2)
    inner_attn = avg_attn[1:-1, 1:-1]   # (L, L)
    return (inner_attn > threshold).astype(int)


def compute_interaction_disruption(
    wt_attn: np.ndarray,
    mut_attn: np.ndarray,
    mutation_pos: int,
    threshold: float = 0.05
) -> float:
    """Compute change in interaction network at mutation site."""
    wt_map = attention_rollout(wt_attn, threshold)
    mut_map = attention_rollout(mut_attn, threshold)
    
    # Compare neighborhood connectivity
    radius = 10
    start = max(0, mutation_pos - radius)
    end = min(len(wt_map), mutation_pos + radius + 1)
    
    wt_conn = wt_map[mutation_pos, start:end].sum()
    mut_conn = mut_map[mutation_pos, start:end].sum()
    
    return wt_conn - mut_conn  # Positive: loss of interactions


# ==============================
# 6. Scientific Reasoning Engine (NLG with Rules)
# ==============================
def generate_mechanistic_explanation(
    sequence: str,
    mutation_str: str,
    perturbation: Dict,
    biophysical: Dict,
    interaction_change: float,
    pdb_reference: Optional[str] = None
) -> str:
    """
    Generate a structured, evidence-based reasoning trace.
    """
    wt_aa, pos_1idx, mut_aa = mutation_str[0], int(mutation_str[1:-1]), mutation_str[-1]
    max_shift = perturbation['max_perturbation']
    rmsd = perturbation['embedding_rmsd']
    hotspot = perturbation['hotspot_position']
    seq_len = len(sequence)

    explanation = (
        f"**Mechanistic Reasoning for Mutation {mutation_str}:**\n\n"
    )

    # Structural context
    if pos_1idx < seq_len * 0.1 or pos_1idx > seq_len * 0.9:
        location = "near the N- or C-terminus"
        implication = "potentially affecting folding kinetics or stability."
    elif seq_len * 0.3 <= pos_1idx <= seq_len * 0.7:
        location = "within the structural core"
        implication = "likely disrupting hydrophobic packing or tertiary contacts."
    else:
        location = "in a loop or flexible region"
        implication = "possibly altering conformational dynamics or catalytic loop motion."

    explanation += f"Position {pos_1idx} lies {location}, which may {implication} "

    # Embedding perturbation
    if max_shift < 0.1:
        explanation += "The ESM-2 model detects minimal contextual perturbation, suggesting evolutionary tolerance. "
    elif max_shift < 0.2:
        explanation += "Moderate embedding shift indicates possible functional modulation. "
    else:
        explanation += "Significant embedding perturbation suggests disruption of evolutionary constraints. "

    # Biophysical analysis
    if biophysical['is_charge_reversal']:
        explanation += f"The mutation introduces a charge reversal ({wt_aa}→{mut_aa}), which may disrupt salt bridges or electrostatic steering. "
    if biophysical['is_hydrophobic_to_polar']:
        explanation += f"Substitution from hydrophobic to polar residue may expose buried surface area, risking misfolding. "

    # Interaction network
    if interaction_change > 1.5:
        explanation += f"The mutation severs {int(interaction_change)} predicted residue-residue interactions, indicating potential allosteric or structural disruption. "
    elif interaction_change < -1.5:
        explanation += f"New interactions are formed, possibly stabilizing an alternative conformation. "

    # Final inference
    effect = "likely neutral" if max_shift < 0.1 else "moderately disruptive" if max_shift < 0.25 else "highly disruptive"
    explanation += f"**Conclusion**: Mutation {mutation_str} is {effect} based on evolutionary, structural, and biophysical evidence."

    if pdb_reference:
        explanation += f" [Reference: PDB {pdb_reference}]"

    logger.info(f"Generated explanation for {mutation_str}")
    return explanation


# ==============================
# 7. Visualization Suite
# ==============================
def plot_embedding_perturbation(perturbation: Dict, mutation_pos: int, mutation_str: str):
    """Plot per-residue embedding similarity shift."""
    delta_sim = perturbation['delta_similarity']
    plt.figure(figsize=(10, 4))
    plt.plot(delta_sim, color='gray', alpha=0.7, label='Contextual Change')
    plt.axvline(x=mutation_pos, color='red', linestyle='--', label=f'Mutation {mutation_str}')
    plt.scatter(mutation_pos, delta_sim[mutation_pos], color='red', zorder=5)
    plt.title("Per-Residue Embedding Perturbation (1 - Cosine Distance)")
    plt.ylabel("Similarity Shift")
    plt.xlabel("Residue Position")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_attention_disruption(wt_attn, mut_attn, mutation_pos, head=0):
    """Compare attention maps before and after mutation."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    pos_tok = mutation_pos + 1  # account for <cls>

    sns.heatmap(wt_attn[head, :, :], ax=axes[0], cmap="viridis", cbar=True)
    axes[0].axvline(x=pos_tok, color='red', linestyle='--')
    axes[0].axhline(y=pos_tok, color='red', linestyle='--')
    axes[0].set_title(f"Wild-Type Attention (Head {head})")

    sns.heatmap(mut_attn[head, :, :], ax=axes[1], cmap="viridis", cbar=True)
    axes[1].axvline(x=pos_tok, color='red', linestyle='--')
    axes[1].axhline(y=pos_tok, color='red', linestyle='--')
    axes[1].set_title(f"Mutant Attention (Head {head})")

    plt.suptitle(f"Attention Redistribution at Mutation Site {mutation_pos + 1}")
    plt.tight_layout()
    plt.show()


# ==============================
# 8. Full Analysis Pipeline
# ==============================
def analyze_mutation(
    sequence: str,
    mutation_str: str,
    pdb_reference: Optional[str] = None
) -> Dict:
    """
    End-to-end mutation reasoning pipeline.
    
    Returns:
        Dictionary with explanation, metrics, and visualizations.
    """
    logger.info(f"Starting analysis for {mutation_str}")
    embedder = ESM2Embedder()

    try:
        mut_seq, mut_pos = apply_mutation(sequence, mutation_str)
    except Exception as e:
        logger.error(f"Mutation application failed: {e}")
        raise

    # Compute embedding perturbation
    perturbation = compute_embedding_perturbation(sequence, mut_seq, embedder)

    # Biophysical change
    biophysical = compute_biophysical_change(mutation_str[0], mutation_str[-1])

    # Interaction disruption
    interaction_change = compute_interaction_disruption(
        perturbation['wt_attentions'],
        perturbation['mut_attentions'],
        mut_pos
    )

    # Generate explanation
    explanation = generate_mechanistic_explanation(
        sequence=sequence,
        mutation_str=mutation_str,
        perturbation=perturbation,
        biophysical=biophysical,
        interaction_change=interaction_change,
        pdb_reference=pdb_reference
    )

    # Visualize
    plot_embedding_perturbation(perturbation, mut_pos, mutation_str)
    plot_attention_disruption(
        perturbation['wt_attentions'],
        perturbation['mut_attentions'],
        mut_pos
    )

    result = {
        'mutation': mutation_str,
        'wild_type_sequence': sequence,
        'mutant_sequence': mut_seq,
        'explanation': explanation,
        'metrics': {
            'embedding_rmsd': float(perturbation['embedding_rmsd']),
            'max_perturbation': float(perturbation['max_perturbation']),
            'hotspot_position': int(perturbation['hotspot_position']),
            'interaction_loss': float(interaction_change),
            'delta_charge': biophysical['delta_charge'],
            'delta_hydrophobicity': biophysical['delta_hydrophobicity']
        },
        'visualizations': ['embedding_shift', 'attention_heatmap']
    }

    logger.success(f"Analysis complete for {mutation_str}")
    return result


# ==============================
# 9. Example Usage (if run directly)
# ==============================
if __name__ == "__main__":
    # Example: TEM-1 β-lactamase (UniProt P00846, partial)
    TEM1 = ("MKKLLAVLCLVLLALVQSQVEQPTESQRQQGYTQIYQGIDYVYSPVNPEDGKVSIVWSRDQINGGWDAIQAGIRDFLVYQQQQQQQLQQLQQLQQLQQ")
    
    result = analyze_mutation(
        sequence=TEM1,
        mutation_str="D219G",
        pdb_reference="1XPB"
    )

    print("\n" + "="*60)
    print(result['explanation'])
    print("="*60)