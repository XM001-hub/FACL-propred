from torch.utils.data import Dataset
import os
import lmdb
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
import pickle
from mordred import Calculator, descriptors
from rdkit.Chem import MACCSkeys
import torch
from rdkit.Chem import BRICS
from torch_geometric.data import Data, DataLoader
from rdkit.Chem import BondType
import re
import numpy as np
from torch_fgib.extract_vocab import detect_functional_group, extract_vocab
import chardet
from collections import defaultdict


class DatasetFrag(Dataset):
    def __init__(self, root, path_dict, split='random', transform=None, finetune=False):
        self.root = root
        self.split = split
        self.smiles_path = os.path.join(root, path_dict['smiles'])
        self.processed_path = os.path.join(root, path_dict['processed'])
        self.idx2mol_path = self.processed_path[:self.processed_path.find('.lmdb')] + '_idx2mol.pt'
        self.finetune = finetune
        self.labels, self.task_num = self.read_labels()
        self.smiles = self.read_smiles()
        
        self.labels = torch.tensor(self.labels, dtype=torch.float32)
        self.mean_l, self.std_l = self.labels.mean(), self.labels.std()

        if split == 'pre_split':
            split_path = os.path.join(root, path_dict['split'])
            self.split = self.get_split(split_path)

        self.transform = transform
        self.db = None
        self.keys = None

        if not os.path.exists(self.processed_path):
            self._process(finetune)
            self._test()
    
    def get_split(self, save_path, caco=False):
        if caco:
            split_path = '/home/rujinxiao/Desktop/GPSCL-main/dataset/ADMET/regression/caco2/scaffold.npy'
            data = np.load(split_path, allow_pickle=True)
            split = {'train': data[0].tolist(), 'valid': data[1].tolist(), 'test': data[2].tolist()}
            return split
        df = pd.read_csv(self.smiles_path)
        if 'scaffold_train_test_label' not in df.columns:
            raise ValueError(f'No scaffold column')
        train_idx = df.index[df['scaffold_train_test_label'].isin(['train', 'train_idx'])].tolist()
        test_idx = df.index[df['scaffold_train_test_label'].isin(['test', 'test_idx'])].tolist()
        split = defaultdict()
        split['train'] = train_idx
        split['test'] = test_idx
        if not os.path.exists(save_path):
            with open(save_path, 'wb') as f:
                pickle.dump(split, f)
        return split


    def read_smiles(self):
        df = pd.read_csv(self.smiles_path)
        smiles = df['smiles'].tolist()
        return smiles
    
    def read_labels(self):
        df = pd.read_csv(self.smiles_path)
        target_list = []
        for column in list(df):
            if column not in ['smiles', 'mol_idx', 'Drug_ID', 'scaffold_split', 'random_split',
                              'property', 'scaffold_train_test_label', 'random_train_test_label']:
                target_list.append(column)
        label_list = []
        for target in target_list:
            label_list.append(df[target].tolist())
        if len(label_list) == 1:
            label_list = label_list[0]
        return label_list, len(target_list)

    def _process(self, finetune=False):
        db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )

        df_smiles = pd.read_csv(self.smiles_path, index_col=0)

        num_skipped = 0
        num_processed = 0
        idx = 0

        with db.begin(write=True, buffers=True) as txn:
            if not finetune:
                for smiles in tqdm(df_smiles.iterrows(), total=len(df_smiles), desc='Processing SMILES'):
                    smiles = smiles[0]
                    # try:
                    mol = Chem.MolFromSmiles(smiles)
                    # MACCS
                    fp_maccs = MACCSkeys.GenMACCSKeys(mol)
                    # mordred_des
                    calc = Calculator(descriptors, ignore_3D=True)
                    descriptor = calc(mol)
                    raw_descriptors = descriptor.values()
                    final_descriptors = list()

                    for raw_descriptor in raw_descriptors:

                        if type(raw_descriptor) is float or type(raw_descriptor) is int:

                            final_descriptors.append(str(raw_descriptor))

                        else:

                            final_descriptors.append("")
                    # morgan_fp
                    morgan_fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=512,
                                                                                    useBondTypes=True)
                    # morgan = morgan_fp.ToList()
                    fingerprints = fp_maccs.ToList() + morgan_fp.ToList()
                    final_descriptors = [0 if x == "" else float(x) for x in final_descriptors]
                    fingerprints = fingerprints + final_descriptors
                    data = get_graph(smiles, fingerprints, num_processed)
                    # key = str(idx).encode()

                    txn.put(
                        key=str(num_processed).encode(),
                        value=pickle.dumps(data)
                    )
                    idx += 1
                    num_processed += 1
                    
                    '''except:
                        num_skipped += 1
                        idx += 1
                        print('Skipping (%d) Num: %s, %s' % (num_skipped, idx, smiles))
                        continue'''
            else:
                for smiles in tqdm(df_smiles.iterrows(), total=len(df_smiles), desc='Processing SMILES'):
                    label = smiles[1][0]
                    smiles = smiles[0]
                    try:
                        data = get_graph(smiles, label, num_processed)
                        txn.put(
                            key=str(num_processed).encode(),
                            value=pickle.dumps(data)
                        )
                        idx += 1
                        num_processed += 1
                    except:
                        num_skipped += 1
                        print('Skipping (%d) Num: %s, %s' % (num_skipped, idx, smiles))
                        continue
        db.close()
        print(self.__len__())
        print('Processed %d moleculed' % (len(df_smiles) - num_skipped), 'Skipped %d molecules' % num_skipped)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __getitem__(self, i):
        if self.db is None:
            self._connect_db()
        key = self.keys[i]
        data = pickle.loads(self.db.begin().get(key))
        if self.transform is not None:
            data = self.transform(data)
        return data

    def _connect_db(self):
        assert self.db is None, 'A connection has already been opened'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _test(self):
        if self.db is None:
            self._connect_db()
        for i in tqdm(range(self.__len__()), desc='Testing data'):
            key = self.keys[i]
            data = pickle.loads(self.db.begin().get(key))
            if data is None:
                print('Data is None')

def get_frag(mol, augmentation=None, get_frag_only=False):
    detect_functional_group(mol)
    split_bonds = split_mol(mol)
    # SetAtomMapNum for mol
    [atom.SetAtomMapNum(i + 1) for i, atom in enumerate(mol.GetAtoms())]
    bonds = []
    for bond in split_bonds:
        x, y = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        bonds.append(mol.GetBondBetweenAtoms(x, y).GetIdx())
    frags = Chem.MolToSmiles(Chem.FragmentOnBonds(mol, bonds)).split('.')
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
    
    frags = [Chem.MolToSmiles(frag) for frag in frags]
    frags = [re.sub(r'\[[0-9]+\*\]', '*', frag) for frag in frags]
    frags = [re.sub(r'\*', '[*:1]', frag) for frag in frags]
    frags = [Chem.MolToSmiles(Chem.MolFromSmiles(frag), isomericSmiles=False) for frag in frags]

    if get_frag_only:
        return frags

    node2frag_batch = torch.zeros(mol.GetNumAtoms(), dtype=torch.int64)
    for batch_id, idx_list in enumerate(frag_idx):
        node2frag_batch[idx_list] = batch_id

    edge2frag_size = []
    for i, frag in enumerate(edge_idx):
        if len(frag) == 0:
            edge2frag_size.append(1)
            edge_idx[i] = torch.tensor([[-1], [-1]], dtype=torch.int64)
        else:
            edge2frag_size.append(frag.shape[1])
    edge2frag_batch = torch.cat(edge_idx, dim=1)
    edge2frag_size = torch.tensor(edge2frag_size, dtype=torch.int64)
    frag2graph_batch = torch.zeros(len(frags), dtype=torch.int64)

    return frags, node2frag_batch, frag2graph_batch, edge2frag_batch, edge2frag_size


def get_graph(smiles, fingerprint, idx=0, split_ions=False, atom_message=True):

    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()

    stereo = Chem.FindMolChiralCenters(mol)
    chiral_centers = [0] * mol.GetNumAtoms()

    for i in stereo:
        chiral_centers[i[0]] = i[1]

    node_features = []
    edge_features = []
    bonds = []

    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)

        atom_i_features = get_atom_features(atom_i, chiral_centers[i])
        node_features.append(atom_i_features)

        for j in range(mol.GetNumAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)

    atom_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(bonds, dtype=torch.long).T
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    detect_functional_group(mol)
    split_bonds = split_mol(mol)
    frags, node2frag_batch, frag2graph_batch, edge2frag_batch, edge2frag_size = split_to_frags(mol, split_bonds)
    # frags, node2frag_batch, frag2graph_batch, edge2frag_batch, edge2frag_size = get_frag_batch_brics(mol)

    if frags is not None:
        return Data(x=atom_features, edge_index=edge_index, edge_attr=edge_features, y=fingerprint,
                    idx=idx, smiles=Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False),
                    frags=frags, node2frag_batch=node2frag_batch, frag2graph_batch=frag2graph_batch,
                    frags_e=edge2frag_batch.T, frags_e_size=edge2frag_size)
    

def replace_isform(frags):
    frags = [re.sub(r'@@', 'IS2', frag) for frag in frags]
    frags = [re.sub(r'@', 'IS1', frag) for frag in frags]
    return frags

def restore_isform(frags):
    frags = [re.sub(r'IS2', '@@', frag) for frag in frags]
    frags = [re.sub(r'IS1', '@', frag) for frag in frags]
    return frags
    

def split_to_frags(mol, split_bonds, get_frag_only=False):
    # SetAtomMapNum for mol
    [atom.SetAtomMapNum(i + 1) for i, atom in enumerate(mol.GetAtoms())]
    bonds = []
    for bond in split_bonds:
        x, y = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        bonds.append(mol.GetBondBetweenAtoms(x, y).GetIdx())
    frags = Chem.MolToSmiles(Chem.FragmentOnBonds(mol, bonds)).split('.')
    # sort to standradize frags_list
    
    # 替换异构符号防止出现异构错误
    frags = replace_isform(frags)
    frags.sort(key = lambda frag: len(frag), reverse=True)
    frags = restore_isform(frags)
    frags = [Chem.MolFromSmiles(frag) for frag in frags]
    '''frags_smiles = Chem.FragmentOnBonds(mol, split_bonds)
    for smiles in frags_smiles:
        frags = []
        frag = Chem.MolFromSmiles(smiles)
        frags.append(frag)'''

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

    frags = [Chem.MolToSmiles(frag) for frag in frags]
    frags = [re.sub(r'\[[0-9]+\*\]', '*', frag) for frag in frags]
    frags = [re.sub(r'\*', '[*:1]', frag) for frag in frags]
    frags = [Chem.MolToSmiles(Chem.MolFromSmiles(frag), isomericSmiles=False) for frag in frags]

    if get_frag_only:
        return frags

    node2frag_batch = torch.zeros(mol.GetNumAtoms(), dtype=torch.int64)
    for batch_id, idx_list in enumerate(frag_idx):
        node2frag_batch[idx_list] = batch_id

    edge2frag_size = []
    for i, frag in enumerate(edge_idx):
        if len(frag) == 0:
            edge2frag_size.append(1)
            edge_idx[i] = torch.tensor([[-1], [-1]], dtype=torch.int64)
        else:
            edge2frag_size.append(frag.shape[1])
    edge2frag_batch = torch.cat(edge_idx, dim=1)
    edge2frag_size = torch.tensor(edge2frag_size, dtype=torch.int64)
    frag2graph_batch = torch.zeros(len(frags), dtype=torch.int64)

    return frags, node2frag_batch, frag2graph_batch, edge2frag_batch, edge2frag_size
    

def split_mol(mol):
    rings = mol.GetRingInfo().AtomRings()
    split_bonds = set()
    for bond in mol.GetBonds():
        begin_atom, end_atom = bond.GetBeginAtom(), bond.GetEndAtom()
        begin_atom_prop = begin_atom.GetProp('FG')
        end_atom_prop = end_atom.GetProp('FG')
        begin_atom_symbol = begin_atom.GetSymbol()
        end_atom_symbol = end_atom.GetSymbol()

        if (begin_atom.IsInRing() and not end_atom.IsInRing()) or (not begin_atom.IsInRing() and end_atom.IsInRing()) or (begin_atom.IsInRing() and end_atom.IsInRing()): # anyone in ring
            flag = True
            for ring in rings:
                if begin_atom.GetIdx() in ring and end_atom.GetIdx() in ring:
                    flag = False
                    break
            if flag:
                split_bonds.add(bond) # one in ring and another not in ring
        else:
            if begin_atom_prop != end_atom_prop:
                split_bonds.add(bond)
            if begin_atom_prop == '' and end_atom_prop == '':
                if (begin_atom_symbol in ['C', '*'] and end_atom_symbol != 'C') or (begin_atom_symbol != 'C' and end_atom_symbol in ['C', '*']):
                    split_bonds.add(bond)
    split_bonds = list(split_bonds)
    return split_bonds


def get_bond_features(bond):
    bond_features = []
    bond_features += one_of_k_encoding_unk(bond.GetBondType(),
                                           [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC])
    bond_features += [bond.GetIsAromatic()]
    bond_features += [bond.GetIsConjugated()]
    bond_features += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return bond_features


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_frag_batch_brics(mol, get_frag_only=False):
    # smiles = Chem.MolToSmiles(mol)
    [atom.SetAtomMapNum(i + 1) for i, atom in enumerate(mol.GetAtoms())]
    # atom_map = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]

    bonds = [bond[0] for bond in list(BRICS.FindBRICSBonds(mol))] # bond_index should be cleared
    bonds = [mol.GetBondBetweenAtoms(x, y).GetIdx() for x, y in bonds]

    if not bonds:
        return None, None, None

    frags = Chem.MolToSmiles(Chem.FragmentOnBonds(mol, bonds)).split('.')
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

    frags = [Chem.MolToSmiles(frag) for frag in frags]
    frags = [re.sub(r'\[[0-9]+\*\]', '*', frag) for frag in frags]
    frags = [re.sub(r'\*', '[*:1]', frag) for frag in frags]
    frags = [Chem.MolToSmiles(Chem.MolFromSmiles(frag), isomericSmiles=False) for frag in frags]

    if get_frag_only:
        return frags

    node2frag_batch = torch.zeros(mol.GetNumAtoms(), dtype=torch.int64)
    for batch_id, idx_list in enumerate(frag_idx):
        node2frag_batch[idx_list] = batch_id

    edge2frag_size = []
    for i, frag in enumerate(edge_idx):
        if len(frag) == 0:
            edge2frag_size.append(1)
            edge_idx[i] = torch.tensor([[-1], [-1]], dtype=torch.int64)
        else:
            edge2frag_size.append(frag.shape[1])
    edge2frag_batch = torch.cat(edge_idx, dim=1)
    edge2frag_size = torch.tensor(edge2frag_size, dtype=torch.int64)
    frag2graph_batch = torch.zeros(len(frags), dtype=torch.int64)

    return frags, node2frag_batch, frag2graph_batch, edge2frag_batch, edge2frag_size


def get_atom_features(atom, stereo, explicit_hydrogen=False, reinterpret_aromatic=True, max_atomic_num: int = 100):
    """
    :param atom_idx: idx in molecule
    :param stereo:
    :param explicit_hydrogen: bool
    :param reinterpret_aromatic: bool
        Whether aromaticity should be determined from the created molecule,
        instead of taken from the SMILES string.
    :return: the features of an atom
    """
    # possible_atoms = ['C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I']
    atom_features = one_of_k_encoding_unk(atom.GetSymbol(), range(max_atomic_num))
    atom_features += one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3])
    atom_features += one_of_k_encoding_unk(atom.GetNumRadicalElectrons(), [0, 1])
    atom_features += one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6])
    atom_features += one_of_k_encoding_unk(atom.GetFormalCharge(), [-1, 0, 1])
    atom_features += one_of_k_encoding_unk(atom.GetHybridization(), [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D])
    atom_features += [atom.IsInRing()]

    if not explicit_hydrogen:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    try:
        atom_features += one_of_k_encoding_unk(stereo, ['R', 'S'])
        atom_features += [atom.HasProp('_ChiralityPossible')]
    except Exception as e:
        atom_features += [False, False] + [atom.HasProp('_ChiralityPossible')]
    return np.array(atom_features)


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def load_vocab_from_file(frag_vocab_path):
    with open(frag_vocab_path, 'rb') as fp:
        data = pickle.load(fp)
    vocab_size = len(data)
    print("vocab size: ", vocab_size)
    return data

'''if __name__ == "__main__":
    frag_vocab_path = '/home/rujinxiao/Desktop/GPSCL-main/dataset/ZINC_250k/vocab.pkl'
    data = load_vocab_from_file(frag_vocab_path)'''