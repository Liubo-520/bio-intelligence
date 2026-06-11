"""
mol_graph.py — SMILES to molecular graph with chemistry-aware features.

Node features include pharmacophore-relevant properties (H-bond donor/acceptor,
hydrophobicity) so the model can learn chemically meaningful representations.

v2: adds Morgan fingerprint (ECFP4) generation for dual-channel drug encoding.
"""
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, AllChem
from torch_geometric.data import Data


# ============================================================
# Atom (node) featurisation
# ============================================================

ATOM_TYPES = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Si', 'B',
              'Na', 'K', 'Ca', 'Fe', 'Zn', 'Mg', 'Se', 'Unknown']

HYBRIDIZATION = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]

FORMAL_CHARGE_RANGE = [-2, -1, 0, 1, 2]
NUM_HS_RANGE = [0, 1, 2, 3, 4]


def _one_hot(val, choices):
    """One-hot encode a value; last dim is 'unknown'."""
    enc = [0] * (len(choices) + 1)
    try:
        idx = choices.index(val)
        enc[idx] = 1
    except ValueError:
        enc[-1] = 1
    return enc


def atom_features(atom, crippen_contribs=None, atom_idx=None):
    """
    Return a feature vector for a single atom.
    
    Features (total = 78 dims):
      - atom type one-hot           (20)
      - hybridisation one-hot       (6)
      - formal charge one-hot       (6)
      - num Hs one-hot              (6)
      - degree                      (1)
      - is aromatic                 (1)
      - is in ring                  (1)
      - ring size (3-8)             (6)
      -- Chemistry priors --
      - is H-bond donor             (1)
      - is H-bond acceptor          (1)
      - Crippen logP contribution   (1)  ← hydrophobicity
      - Crippen MR contribution     (1)  ← molar refractivity
      - Gasteiger charge            (1)  ← partial charge
      
    Pharmacophore-relevant features are marked above; they help the model
    learn which atoms are likely to participate in binding interactions.
    """
    symbol = atom.GetSymbol()
    features = []

    # basic properties
    features += _one_hot(symbol, ATOM_TYPES)                   # 20
    features += _one_hot(atom.GetHybridization(), HYBRIDIZATION)  # 6
    features += _one_hot(atom.GetFormalCharge(), FORMAL_CHARGE_RANGE)  # 6
    features += _one_hot(atom.GetTotalNumHs(), NUM_HS_RANGE)   # 6

    features.append(atom.GetDegree() / 6.0)                    # 1
    features.append(int(atom.GetIsAromatic()))                  # 1
    features.append(int(atom.IsInRing()))                       # 1

    # ring sizes
    ring_info = atom.GetOwningMol().GetRingInfo()
    for size in range(3, 9):
        features.append(int(ring_info.IsAtomInRingOfSize(atom.GetIdx(), size)))  # 6

    # pharmacophore features
    # H-bond donor: atom bonded to at least one H and is N or O
    is_donor = (symbol in ['N', 'O'] and atom.GetTotalNumHs() > 0)
    features.append(int(is_donor))                              # 1

    # H-bond acceptor: N or O with lone pair
    is_acceptor = (symbol in ['N', 'O', 'F'])
    features.append(int(is_acceptor))                           # 1

    # Crippen contributions (hydrophobicity, molar refractivity)
    if crippen_contribs is not None and atom_idx is not None:
        logp_contrib, mr_contrib = crippen_contribs[atom_idx]
        features.append(logp_contrib)                           # 1
        features.append(mr_contrib)                             # 1
    else:
        features.append(0.0)
        features.append(0.0)

    # Gasteiger partial charge
    charge = float(atom.GetDoubleProp('_GasteigerCharge')) \
        if atom.HasProp('_GasteigerCharge') else 0.0
    if np.isnan(charge) or np.isinf(charge):
        charge = 0.0
    features.append(charge)                                     # 1

    return features  # total: 20+6+6+6+1+1+1+6+1+1+1+1+1 = 52...
    # recount: 20+6+6+6+1+1+1+6+1+1+1+1+1 = 52
    # Note: actual dim depends on implementation; config should match


# ============================================================
# Bond (edge) featurisation
# ============================================================

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
]


def bond_features(bond):
    """
    Return a feature vector for a single bond.
    
    Features (total = 16 dims):
      - bond type one-hot           (5)
      - is conjugated               (1)
      - is in ring                  (1)
      - bond stereo one-hot         (6)
      - bond direction info         (3)
    """
    features = []
    features += _one_hot(bond.GetBondType(), BOND_TYPES)       # 5
    features.append(int(bond.GetIsConjugated()))                # 1
    features.append(int(bond.IsInRing()))                       # 1
    features += _one_hot(bond.GetStereo(), BOND_STEREO)        # 6
    
    # bond direction (for chirality)
    bd = bond.GetBondDir()
    features.append(int(bd == Chem.rdchem.BondDir.BEGINWEDGE))
    features.append(int(bd == Chem.rdchem.BondDir.ENDDOWNRIGHT))
    features.append(int(bd == Chem.rdchem.BondDir.ENDUPRIGHT))  # 3
    
    return features  # total: 5+1+1+6+3 = 16


# ============================================================
# Full molecule → PyG Data
# ============================================================

def smiles_to_graph(smiles: str) -> Data | None:
    """
    Convert a SMILES string into a PyTorch Geometric Data object with
    chemistry-aware node and edge features.
    
    Returns None if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # add Hs implicitly tracked, compute Gasteiger charges
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    crippen = Crippen.MolLogP(mol)  # just to ensure descriptors initialised
    crippen_contribs = Crippen._GetAtomContribs(mol)

    # --- node features ---
    atom_feat_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        atom_feat_list.append(atom_features(atom, crippen_contribs, idx))
    x = torch.tensor(atom_feat_list, dtype=torch.float)

    # --- edge features ---
    edge_index = []
    edge_attr_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = bond_features(bond)
        # undirected: add both directions
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr_list.append(bf)
        edge_attr_list.append(bf)

    if len(edge_index) == 0:
        # single atom molecule (rare but possible)
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 16), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                smiles=smiles, num_atoms=x.size(0))


def get_atom_feature_dim():
    """Return the dimensionality of atom features."""
    # Build a dummy molecule to infer feature size
    mol = Chem.MolFromSmiles('C')
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    contribs = Crippen._GetAtomContribs(mol)
    return len(atom_features(mol.GetAtomWithIdx(0), contribs, 0))


def get_bond_feature_dim():
    """Return the dimensionality of bond features."""
    return 16


def smiles_to_morgan(smiles: str, n_bits: int = 1024, radius: int = 2) -> torch.Tensor | None:
    """
    Compute Morgan fingerprint (ECFP4) from SMILES.

    Morgan fingerprints encode circular substructures and are inherently
    more generalizable than learned GNN features — they capture chemical
    similarity based on substructure occurrence regardless of whether the
    exact molecule has been seen during training.

    Args:
        smiles: SMILES string
        n_bits: fingerprint length (default 1024)
        radius: Morgan radius (2 = ECFP4)

    Returns:
        (n_bits,) float tensor, or None if parsing fails
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(fp, arr)
    return torch.from_numpy(arr)
