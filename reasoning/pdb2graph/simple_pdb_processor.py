# simple_pdb_processor.py
from Bio.PDB import PDBParser
import torch
import numpy as np
from torch_geometric.data import Data

def simple_pdb_to_graph(pdb_file):
    """Simplified PDB to graph converter"""
    
    # Parse PDB file
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    
    # Extract C-alpha coordinates
    coordinates = []
    sequence = ""
    
    # Standard amino acid mapping
    aa_3to1 = {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
    }
    
    # Get first chain
    chain = next(structure[0].get_chains())
    
    for residue in chain:
        if residue.get_resname() in aa_3to1:
            try:
                ca_atom = residue['CA']
                coordinates.append(ca_atom.get_coord())
                sequence += aa_3to1[residue.get_resname()]
            except KeyError:
                continue
    
    coordinates = np.array(coordinates)
    n = len(coordinates)
    
    # Create simple contact graph (adjacent residues)
    edge_index = []
    for i in range(n-1):
        edge_index.append([i, i+1])
        edge_index.append([i+1, i])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    
    # Simple node features (one-hot encoding)
    amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
    aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
    
    x = torch.zeros(n, len(amino_acids))
    for i, aa in enumerate(sequence):
        if aa in aa_to_idx:
            x[i, aa_to_idx[aa]] = 1
    
    return Data(
        x=x,
        edge_index=edge_index,
        sequence=sequence,
        num_nodes=n
    )

# Usage
if __name__ == "__main__":
    try:
        graph = simple_pdb_to_graph('reasoning/pdb2graph/102L.pdb')
        print(f"Processed graph: {graph.num_nodes} nodes, {graph.num_edges} edges")
        print(f"Sequence: {graph.sequence[:20]}...")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure 102L.pdb exists in the current directory")