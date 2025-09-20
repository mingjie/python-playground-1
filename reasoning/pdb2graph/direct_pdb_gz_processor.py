import gzip
import requests
import io
from Bio.PDB import PDBParser
import numpy as np
import torch
from torch_geometric.data import Data

class DirectPDBGZProcessor:
    def __init__(self):
        self.parser = PDBParser(QUIET=True)
    
    def load_pdb_gz_from_url(self, pdb_id):
        """Load PDB.gz directly from URL without saving decompressed file"""
        pdb_id = pdb_id.lower()
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb.gz"
        
        try:
            # Download compressed data directly into memory
            response = requests.get(url)
            response.raise_for_status()
            
            # Decompress in memory
            compressed_data = io.BytesIO(response.content)
            with gzip.GzipFile(fileobj=compressed_data, mode='rb') as gz_file:
                pdb_data = gz_file.read().decode('utf-8')
            
            # Parse PDB from string data
            pdb_string = io.StringIO(pdb_data)
            structure = self.parser.get_structure(pdb_id, pdb_string)
            
            return structure
            
        except Exception as e:
            print(f"Error loading {pdb_id}: {e}")
            return None
    
    def load_pdb_gz_from_file(self, gz_file_path):
        """Load PDB.gz from local file without saving decompressed version"""
        try:
            # Read and decompress in memory
            with gzip.open(gz_file_path, 'rt') as gz_file:
                structure = self.parser.get_structure("structure", gz_file)
            return structure
        except Exception as e:
            print(f"Error loading {gz_file_path}: {e}")
            return None

# Example usage
processor = DirectPDBGZProcessor()

# Load from URL
structure = processor.load_pdb_gz_from_url('1AKE')
if structure:
    print(f"Loaded structure with {len(list(structure.get_chains()))} chains")

# Load from local .gz file
# structure = processor.load_pdb_gz_from_file('1AKE.pdb.gz')