from rdkit import Chem
from data.ZINC_dataset import get_atom_features, get_bond_features
from torch_fgib.fgib_utils import detect_functional_group
from data.fgib_data import split_mol
import torch
import re
import numpy as np
from torch_fgib.fgib import FGIBModel
from argparse import Namespace
from typing import List, Tuple

def get_af_type(rel=False):
    if rel:
        relation = np.ones(10)
    else:
        relation = np.random.rand(10)
    return relation


class MolGraph():
    """
        A MolGraph represents the graph strcuture and featurization of a single molecule.
    """
    def __init__(self, mol: Chem.Mol, frag_emb: list = None, augmentation=False):
        """
        Computes the graph structure and featurization of a molecule.

        :param mol: A Chem.mol data.
        :param frag_emb: A list of frag_emb for molecule.
        :param augmentation: A boolean indicating whether to augment the molecule.
        """
        self.mol = mol
        self.smiles = Chem.MolToSmiles(mol)
        self.n_atoms = 0
        self.n_bonds = 0  # number of bonds
        self.n_frags = 0  # number of frags
        f_atoms = []  # mapping from atom index to atom features
        f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond

        self.frag_emb = frag_emb
        self.augmentation = augmentation
        self.edge_index = []
        atom_message = False

        # find the stereo of mol for atom featurization
        stereo = Chem.FindMolChiralCenters(mol)
        chiral_centers = [0] * mol.GetNumAtoms()
        for i in stereo:
            chiral_centers[i[0]] = i[1]
        
        if not self.augmentation:
            self.n_nodes = mol.GetNumAtoms()

            for i, atom_i in enumerate(mol.GetAtoms()):
                f_atoms.append(get_atom_features(atom_i, chiral_centers[i]))
            
            # initialization
            for _ in range(self.n_nodes):
                self.a2b.append([])
            
            for i, atom_i in enumerate(mol.GetAtoms()):
                    # get bond features
                    for j, atom_j in enumerate(mol.GetAtoms()):
                        bond_ij = mol.GetBondBetweenAtoms(i, j)
                        if bond_ij is not None:
                            bond_features_ij = get_bond_features(bond_ij)

                            if atom_message:
                                f_bonds.append(bond_features_ij)
                                f_bonds.append(bond_features_ij)
                            else:
                                f_bonds.append(list(f_atoms[i]) + list(bond_features_ij))
                                f_bonds.append(list(f_atoms[j]) + list(bond_features_ij))

                            b1 = self.n_bonds
                            b2 = b1 + 1
                            self.a2b[j].append(b1)  # b1 = a1 --> a2
                            self.b2a.append(i)
                            self.a2b[i].append(b2)  # b2 = a2 --> a1
                            self.b2a.append(j)
                            self.b2revb.append(b2)
                            self.b2revb.append(b1)
                            self.n_bonds += 2
                            self.edge_index.append([i, j])
            self.x = f_atoms
            self.edge_attr = f_bonds
        
        else:
            self.n_atoms = mol.GetNumAtoms()

            for i, atom_i in enumerate(mol.GetAtoms()):
                f_atoms.append(get_atom_features(atom_i, chiral_centers[i]))
            self.frags, self.node2frag, self.frag_i_edges, self.frag_f_edges, _ = get_frag_crem(mol, augmentation=self.augmentation)
            self.n_frags = len(self.frags)

            # initialization of all nodes
            for _ in range(self.n_atoms + self.n_frags):
                self.a2b.append([])

            # reinitialization of node2frag, frag_i_edges (+n_atoms)
            self.node2frag = [int(i + self.n_atoms) for i in self.node2frag]
            n_frag_edges = len(self.frag_i_edges)
            edge_i_add = [self.n_atoms] * n_frag_edges * 2
            self.frag_i_edges = [a + b for a, b in zip(edge_i_add, np.array(self.frag_i_edges).reshape(-1).tolist())]
            self.frag_i_edges = np.array(self.frag_i_edges).reshape(n_frag_edges, 2).tolist()

            f_frags = self.frag_emb
            f_atoms += f_frags
            self.n_nodes = self.n_atoms + self.n_frags

            # compose directed graph
            for i in range(self.n_nodes):
                for j in range(i+1, self.n_nodes):
                    if i < self.n_atoms and j < self.n_atoms:
                        # bond between atom and atom
                        bond_ij = mol.GetBondBetweenAtoms(i, j)
                        if bond_ij is not None:
                            bond_features_ij = np.array(get_bond_features(bond_ij)).astype(int)

                            if atom_message:
                                f_bonds.append(list(bond_features_ij))
                                f_bonds.append(list(bond_features_ij))
                            else:
                                f_bonds.append(list(f_atoms[i]) + list(bond_features_ij))
                                f_bonds.append(list(f_atoms[j]) + list(bond_features_ij))
                        
                        else:
                            continue
                    elif i >= self.n_atoms and j >= self.n_atoms:
                        # bond between frag and frag
                        if [i, j] in self.frag_i_edges or [j, i] in self.frag_i_edges:
                            index = self.frag_i_edges.index([i, j]) if [i, j] in self.frag_i_edges \
                                else self.frag_i_edges.index([j, i])
                            f_bond = np.array(self.frag_f_edges[index]).astype(int)
                            if atom_message:
                                f_bonds.append(list(f_bond))
                                f_bonds.append(list(f_bond))
                            else:
                                f_bonds.append(f_atoms[i] + list(f_bond))
                                f_bonds.append(f_atoms[j] + list(f_bond))
                        else:
                            continue
                    else:
                        # bond between frag and atom
                        index = [i, j]
                        index.sort()
                        if self.node2frag[index[0]] == index[1]:
                            relation = get_af_type(rel=True)
                        else:
                            relation = get_af_type(rel=False)
                        if atom_message:
                            f_bonds.append(relation)
                            f_bonds.append(relation)
                        else:
                            f_bonds.append(list(f_atoms[i]) + list(relation))
                            f_bonds.append(list(f_atoms[j]) + list(relation))
                    
                    # update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[j].append(b1) # b1 = a1 --> a2
                    self.b2a.append(i)
                    self.a2b[i].append(b2) # b2 = a2 --> a1
                    self.b2a.append(j)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2
                    self.edge_index.append([i, j])
            
            self.x = f_atoms
            self.edge_attr = f_bonds


def get_frag_crem(mol, get_frag_only=False, augmentation=None):

    detect_functional_group(mol)
    split_bonds = split_mol(mol)
    atoms_num = mol.GetNumAtoms()

    [atom.SetAtomMapNum(i + 1) for i, atom in enumerate(mol.GetAtoms())]
    bonds = []
    bonds_idx = []
    frag_f_edge = []
    for bond in split_bonds:
        x, y = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        bonds_idx.append((x, y))
        bonds.append(mol.GetBondBetweenAtoms(x, y).GetIdx())
        frag_f_edge.append(get_bond_features(bond))
    
    frag_bonds = bonds_idx
    frags = Chem.MolToSmiles(Chem.FragmentOnBonds(mol, bonds)).split('.')
    # not sort to match atoms in mol
    frags = [Chem.MolFromSmiles(frag) for frag in frags]

    frag_idx = []
    edge_idx = []
    for frag in frags:
        frag_list = []
        edge_list = []
        for atom in frag.GetAtoms():
            if atom.GetAtomMapNum() > 0:
                frag_list.append(atom.GetAtomMapNum() - 1)
                for atom_j in frag.GetAtoms():
                    if atom_j.GetAtomMapNum() > 0:
                        bond_ij = mol.GetBondBetweenAtoms(atom.GetAtomMapNum() - 1, atom_j.GetAtomMapNum() - 1)
                        if bond_ij is not None:
                            edge_list.append([atom.GetAtomMapNum() - 1, atom_j.GetAtomMapNum() - 1])
        frag_idx.append(frag_list)
        edge_idx.append(torch.tensor(edge_list, dtype=torch.long).T)
        [atom.SetAtomMapNum(0) for atom in frag.GetAtoms()]
    
    # create a edge_index for frag
    # frag_i_edges = [[0, 0]] * len(frag_bonds)
    frag_i_edges = []
    for _ in range(len(frag_bonds)):
        frag_i_edges.append([0, 0])
    for i, bond in enumerate(frag_bonds):
        for j, frag_j in enumerate(frag_idx):
            if bond[0] in frag_j:
                frag_i_edges[i][0] = j
            if bond[1] in frag_j:
                frag_i_edges[i][1] = j
    
    frags = [Chem.MolToSmiles(frag) for frag in frags]
    frags = [re.sub(r'\[[0-9]+\*\]', '*', frag) for frag in frags]
    frags = [re.sub(r'\*', '[*:1]', frag) for frag in frags]
    frags = [Chem.MolToSmiles(Chem.MolFromSmiles(frag), isomericSmiles=False) for frag in frags]

    if get_frag_only:
        return frags
    
    node2frag_batch = torch.zeros(mol.GetNumAtoms(), dtype=torch.int64)
    for batch_id, idx_list in enumerate(frag_idx):
        node2frag_batch[idx_list] = batch_id
    
    return frags, node2frag_batch, frag_i_edges, frag_f_edge, frag_idx


class BatchMolGraph():
    # add a 0 atom and bond to batch?
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule。 # could be replaced by a_size in lmdb
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs: list, args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = 130
        self.bond_fdim = 10 + (not args.atom_messages) * self.atom_fdim  # * 2

        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        batch = [0]
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        bonds = [[0, 0]]
        for i, mol_graph in enumerate(mol_graphs):
            batch.extend([i + 1] * mol_graph.n_nodes) # create mol_idx of f_atoms in batch
            f_atoms.extend(mol_graph.x)
            f_bonds.extend(mol_graph.edge_attr)

            for a in range(mol_graph.n_nodes): # n_atoms should be replaced by n_nodes (n_atoms + n_frags)
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]])  # if b!=-1 else 0

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
                bonds.append([b2a[-1],
                              self.n_atoms + mol_graph.b2a[mol_graph.b2revb[b]]])
            self.a_scope.append((self.n_atoms, mol_graph.n_nodes))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.n_atoms += mol_graph.n_nodes
            self.n_bonds += mol_graph.n_bonds

        bonds = np.array(bonds).transpose(1, 0)

        self.max_num_bonds = max(1, max(
            len(in_bonds) for in_bonds in a2b))  # max with 1 to fix a crash in rare case of all single-heavy-atom mols

        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.a2b = torch.LongTensor(
            [a2b[a][:self.max_num_bonds] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)]) # padding to a fixed num_bonds
        self.b2a = torch.LongTensor(b2a)
        self.bonds = torch.LongTensor(bonds)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages
        self.emb_init = args.BatchMolGraph.emb # args not have attribute emb
        self.batch = torch.tensor(batch, dtype=torch.int64)

    def get_components(self) -> [torch.FloatTensor, torch.FloatTensor,
    torch.LongTensor, torch.LongTensor, torch.LongTensor,
    List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.b_scope, self.bonds

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a

    def get_frag_emb(self):
        emb_model = FGIBModel(self.emb_init, out_dir=None)
        frag_emb = emb_model()
        frag_emb = []
        return frag_emb
    
def mol2graph(smiles_batch: List[str],
              args: Namespace, augmentation: bool, frag_emb=None, frag_batch=None) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :param augmentation: Whether to apply the augmentation or not.
    :param frag_emb: A batch of fragment embeddings.
    :param frag_batch: A batch of n_frags of molecule.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    n_frags = 0
    for i, smiles in enumerate(smiles_batch):
        if augmentation:
            frag_size = frag_batch.tolist()[i]
            mol_graph = MolGraph(smiles, frag_emb[n_frags:(n_frags + frag_size), :].tolist(), augmentation)
            n_frags += frag_size
            mol_graphs.append(mol_graph)
        else:
            mol_graph = MolGraph(smiles, None, augmentation)
            mol_graph.append(mol_graph)
    return BatchMolGraph(mol_graphs, args=args)