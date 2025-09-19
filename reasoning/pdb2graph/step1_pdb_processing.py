# step1_pdb_processing.py
import torch
import numpy as np
from Bio.PDB import PDBParser, Selection

# LMJ hack
# from Bio.PDB.Polypeptide import three_to_one
#from Bio.PDB import protein_letters_3to1

from torch_geometric.data import Data
import os
import glob
from tqdm import tqdm

class PDBGraphProcessor:
    """Process PDB files and convert to graph representations"""
    
    def __init__(self, distance_threshold=8.0):
        self.distance_threshold = distance_threshold
        self.parser = PDBParser(QUIET=True)
        
        # Standard amino acid mapping
        self.aa_3to1 = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
        }
    
    def extract_coordinates_and_sequence(self, structure, chain_id='A'):
        """Extract Cα coordinates and sequence from PDB structure"""
        coordinates = []
        sequence = ""
        residue_ids = []
        
        try:
            for residue in structure[0][chain_id]:
                if residue.get_resname() in self.aa_3to1:
                    try:
                        ca_atom = residue['CA']
                        coordinates.append(ca_atom.get_coord())
                        sequence += self.aa_3to1[residue.get_resname()]
                        residue_ids.append(residue.get_id()[1])
                    except KeyError:
                        # Missing CA atom
                        continue
        except KeyError:
            # Chain not found, try first chain
            first_chain = next(structure[0].get_chains())
            return self.extract_coordinates_and_sequence(structure, first_chain.get_id())
        
        return np.array(coordinates), sequence, residue_ids
    
    def calculate_distance_matrix(self, coordinates):
        """Calculate pairwise distances between residues"""
        n = len(coordinates)
        distances = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                dist = np.linalg.norm(coordinates[i] - coordinates[j])
                distances[i, j] = distances[j, i] = dist
        
        return distances
    
    def create_contact_graph(self, coordinates, distance_threshold=None):
        """Create contact graph based on spatial proximity"""
        if distance_threshold is None:
            distance_threshold = self.distance_threshold
            
        distances = self.calculate_distance_matrix(coordinates)
        n = len(coordinates)
        
        edge_index = []
        edge_attr = []
        
        for i in range(n):
            for j in range(i+1, n):
                if distances[i, j] <= distance_threshold:
                    # Add bidirectional edges
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    
                    # Edge attributes: distance, sequence distance
                    seq_dist = abs(i - j)
                    edge_attr.append([distances[i, j], seq_dist])
                    edge_attr.append([distances[i, j], seq_dist])
        
        if edge_index:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 2), dtype=torch.float)
        
        return edge_index, edge_attr
    
    def create_node_features(self, sequence, coordinates=None):
        """Create node features for graph"""
        # One-hot encoding of amino acids
        amino_acids = list('ACDEFGHIKLMNPQRSTVWY')
        aa_to_idx = {aa: i for i, aa in enumerate(amino_acids)}
        
        # Physicochemical properties
        hydrophobicity = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        charge = {
            'A': 0, 'R': 1, 'N': 0, 'D': -1, 'C': 0,
            'Q': 0, 'E': -1, 'G': 0, 'H': 1, 'I': 0,
            'L': 0, 'K': 1, 'M': 0, 'F': 0, 'P': 0,
            'S': 0, 'T': 0, 'W': 0, 'Y': 0, 'V': 0
        }
        
        # Create features
        n = len(sequence)
        node_features = torch.zeros(n, len(amino_acids) + 4)  # one-hot + properties
        
        for i, aa in enumerate(sequence):
            # One-hot encoding
            if aa in aa_to_idx:
                node_features[i, aa_to_idx[aa]] = 1
            
            # Physicochemical properties
            node_features[i, len(amino_acids)] = hydrophobicity.get(aa, 0)
            node_features[i, len(amino_acids) + 1] = charge.get(aa, 0)
            
            # Structural features (if coordinates available)
            if coordinates is not None and i < len(coordinates):
                # Solvent accessibility (simplified)
                node_features[i, len(amino_acids) + 2] = 0.5  # placeholder
                
                # Secondary structure propensity (simplified)
                node_features[i, len(amino_acids) + 3] = 0.5  # placeholder
        
        return node_features
    
    def pdb_to_graph(self, pdb_file, chain_id='A'):
        """Convert PDB file to graph representation"""
        try:
            structure = self.parser.get_structure('protein', pdb_file)
            coordinates, sequence, residue_ids = self.extract_coordinates_and_sequence(structure, chain_id)
            
            if len(coordinates) == 0:
                return None
            
            # Create node features
            x = self.create_node_features(sequence, coordinates)
            
            # Create edges
            edge_index, edge_attr = self.create_contact_graph(coordinates)
            
            # Create graph data
            graph_data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                sequence=sequence,
                coordinates=torch.tensor(coordinates, dtype=torch.float),
                num_nodes=len(sequence)
            )
            
            return graph_data
            
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")
            return None

# Test the processor
if __name__ == "__main__":
    processor = PDBGraphProcessor()
    
    # Example usage (you would need actual PDB files)
    print("PDB Graph Processor initialized")
    print("Available methods:")
    print("- extract_coordinates_and_sequence()")
    print("- create_contact_graph()")
    print("- create_node_features()")
    print("- pdb_to_graph()")

    processor.pdb_to_graph()