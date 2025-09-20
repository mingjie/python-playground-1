# download_structures.py
import requests, os

def download_pdb(pdb_id, outdir="pdbs"):
    os.makedirs(outdir, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    r = requests.get(url)
    r.raise_for_status()
    path = os.path.join(outdir, f"{pdb_id}.pdb")
    with open(path, "w") as fh:
        fh.write(r.text)
    return path

def download_alphafold_by_uniprot(uniprot_accession, outdir="alphafold"):
    os.makedirs(outdir, exist_ok=True)
    # AlphaFold DB per-UniProt download
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_accession}-F1-model_v4.pdb"
    r = requests.get(url)
    if r.status_code == 200:
        path = os.path.join(outdir, f"{uniprot_accession}_af.pdb")
        with open(path, "wb") as fh:
            fh.write(r.content)
        return path
    else:
        raise RuntimeError("AlphaFold prediction not found for accession: " + uniprot_accession)

# Example usage:
# download_pdb("1A2B")  # replace with your PDB id
# download_alphafold_by_uniprot("P01234")