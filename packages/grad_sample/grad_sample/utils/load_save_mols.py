import pickle
from grad_sample.models.qchem import PCMolecule

# Saving the PCMolecule object with a default filename
def save_pcmolecule(molecule: PCMolecule, filename: str = None):
    # If no filename is provided, use the default filename based on the molecule's CID
    if filename is None:
        filename = f"pcmolecule_{molecule.cid}.pkl"
    
    with open(filename, 'wb') as f:
        pickle.dump(molecule, f)
        print(f"PCMolecule saved to {filename}")

# Loading the PCMolecule object with a default filename
def load_pcmolecule(cid: int, filename: str = None) -> PCMolecule:
    # If no filename is provided, use the default filename based on the molecule's CID
    if filename is None:
        filename = f"pcmolecule_{cid}.pkl"
    
    with open(filename, 'rb') as f:
        molecule = pickle.load(f)
        print(f"PCMolecule loaded from {filename}")
    return molecule