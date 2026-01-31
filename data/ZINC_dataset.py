import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning import LightningDataModule
from tqdm import tqdm
from rdkit import Chem
import pickle
from torch_geometric.data import Data, DataLoader
import numpy as np
from rdkit.Chem import BondType
from rdkit.Chem import BRICS
import networkx as nx
import torch
from torch.utils.data import Dataset, random_split, Subset
import re
import json
import os
import lmdb
from rdkit.Chem import MACCSkeys
from mordred import Calculator, descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
from collections import defaultdict
import random
from pathlib import Path


class SmilesDataModule(LightningDataModule):
    def __init__(self, batch_size, root, split, split_path=None):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.split = split
        self.pre_split = False
        self.split_path = split_path
        self.train_data, self.val_data, self.test_data = None, None, None

    def setup(self, stage=None):
        dataset = SmilesDataset(self.root, self.pre_split, pretrain=True)
        if self.split == 'random':
            test_size = int(len(dataset) * 0.1)
            other_size = len(dataset) - test_size
            train_dataset, test_dataset = random_split(dataset, [other_size, test_size],
                                                       generator=torch.Generator().manual_seed(2024))
            val_size = int(len(dataset) * 0.1)
            train_size = other_size - val_size
            train_data, val_data = random_split(train_dataset, [train_size, val_size])

        elif self.split == 'scaffold':
            split = self.scaffold_split(dataset, ratio=[0.8, 0.1, 0.1], seed=2024)
            subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
            train_data, val_data, test_dataset = subsets['train'], subsets['val'], subsets['test']
            print('train: {}, val: {}, test: {}'.format(len(train_data), len(val_data), len(test_dataset)))
        else:
            raise NotImplementedError

        if stage == 'fit' or stage is None:
            self.train_data, self.val_data = train_data, val_data
        elif stage == 'test':
            self.test_data = test_dataset
        else:
            raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_data, self.batch_size, shuffle=False, num_workers=4)

    def scaffold_split(self, dataset, ratio: list, balanced=True, seed=0):
        train_size, val_size = int(len(dataset) * ratio[0]), int(len(dataset) * ratio[1])
        if not os.path.exists(Path(self.split_path)):
            train, val, test = [], [], []
            train_scaffold_count, val_scaffold_count, test_scaffold_count = 0, 0, 0
            scaffolds = self.get_scaffold(dataset)
            if balanced: # prevent big scaffold from val_data and test_data
                index_sets = list(scaffolds.values())
                big_index_sets = []
                small_index_sets = []
                for index_set in index_sets:
                    if len(index_set) > val_size / 2:
                        big_index_sets.append(index_set)
                    else:
                        small_index_sets.append(index_set)
                random.seed(seed)
                random.shuffle(big_index_sets)
                random.shuffle(small_index_sets)
                index_sets = big_index_sets + small_index_sets
            else: # sort from largest to smallest scaffold sets
                index_sets = sorted(list(scaffolds.values()),
                                    key=lambda index_set: len(index_set),
                                    reverse=True)

            for index_set in index_sets:
                if len(train) + len(index_set) <= train_size:
                    train += index_set
                    train_scaffold_count += 1
                elif len(val) + len(index_set) <= val_size:
                    val += index_set
                    val_scaffold_count += 1
                else:
                    test += index_set
                    test_scaffold_count += 1
            split = {'train': train, 'val': val, 'test': test}
            torch.save(split, Path(self.split_path))

        else:
            split = torch.load(Path(self.split_path))
        return split

    @staticmethod
    def get_scaffold(dataset, includeChirality=False, use_indices=True):
        scaffolds = defaultdict(set)
        for i in tqdm(range(len(dataset)), desc='reading scaffolds'):
            smiles = dataset[i]
            mol = Chem.MolFromSmiles(smiles)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=includeChirality)
            if use_indices:
                scaffolds[scaffold].add(i)
            else:
                scaffolds[scaffold].add(mol)
        return scaffolds


class SmilesDataset(Dataset):
    def __init__(self, root, pre_split, dataset_type, norm, pretrain):
        self.pretrain = pretrain
        if not pretrain:
            self.smiles_path = root
            self.labels, self.task_num = self.read_labels()
        else:
            self.smiles_path = os.path.join(root, 'smiles.csv')
        self.smiles = self.read_smiles()
        self.pre_split = pre_split
        self.dataset_type = dataset_type
        if self.pre_split:
            self.split = self.get_split()
        if dataset_type == 'regression':
            self.labels = torch.tensor(self.labels, dtype=torch.float32)
            if norm == 'zscore':
                self.mean_l, self.std_l = self.labels.mean(), self.labels.std()
                self.scaled_labels = ((self.labels - self.mean_l) / self.std_l).tolist()
            elif norm == 'minmax':
                min_l, max_l = self.labels.min(), self.labels.max()
                self.scaled_labels = ((self.labels - min_l) / (max_l - min_l)).tolist()

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        if self.pretrain:
            return self.smiles['smiles'][idx]
        else:
            if self.dataset_type == 'regression':
                return self.smiles['smiles'][idx], self.labels[0][idx], self.scaled_labels[0][idx]
            else:
                return self.smiles['smiles'][idx], self.labels[0][idx]

    def get_label(self, idx, target):
        return self.smiles[target][idx]

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
        return label_list, len(target_list)

    def read_smiles(self):
        smiles = pd.read_csv(self.smiles_path)
        return smiles

    def get_split(self):
        split = {'train': [], 'test': []}
        scaffold = self.smiles['scaffold_train_test_label'].tolist()
        for i in range(len(scaffold)):
            if scaffold[i] == 'train':
                split['train'].append(i)
            else:
                split['test'].append(i)
        return split


class DatasetFPModule(pl.LightningDataModule):
    def __init__(self, batch_size, root, path_dict, split):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.path_dict = path_dict
        self.split = split
        self.train_data, self.val_data, self.test_data = None, None, None

    def setup(self, stage=None):
        dataset = DatasetFP(self.root, self.path_dict, self.split)
        test_size = int(len(dataset) * 0.1)
        other_size = len(dataset) - test_size
        train_dataset, test_dataset = random_split(dataset, [other_size, test_size],
                                                   generator=torch.Generator().manual_seed(2024))
        if self.split == 'random':
            val_size = int(len(dataset) * 0.1)
            train_size = other_size - val_size
            train_data, val_data = random_split(train_dataset, [train_size, val_size])

        elif self.split == 'scaffold':
            split = self.scaffold_split(dataset, ratio=[0.8, 0.1, 0.1], seed=0)
            subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
            train_data, val_data, test_dataset = subsets['train'], subsets['val'], subsets['test']
        else:
            raise NotImplementedError

        if stage == 'fit' or stage is None:
            self.train_data, self.val_data = train_data, val_data
        elif stage == 'test':
            self.test_data = test_dataset
        else:
            raise NotImplementedError

        '''for i in range(len(dataset)):
            print(dataset[i].frag2graph_batch.max())'''
        '''if stage == 'fit' or stage is None:
            if self.split == 'random':
                # self.train_data, self.val_data = random_split(dataset, [253, 63])
                self.train_data, self.val_data = random_split(dataset, lengths=[197236, 49309])
            elif self.split == 'scaffold':
                split = self.scaffold_split(dataset, ratio=[0.8, 0.1, 0.1], seed=0)
                subsets = {k: Subset(dataset, indices=v) for k, v in split.items()}
                self.train_data, self.val_data = subsets['train'], subsets['val']
            else:
                self.train_data, self.val_data = dataset
        elif stage == 'test':
            self.test_data = dataset
        else:
            raise NotImplementedError'''

    def train_dataloader(self):
        return DataLoader(self.train_data, self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_data, self.batch_size, shuffle=False, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_data, self.batch_size, shuffle=False, num_workers=4)

    def scaffold_split(self, dataset, ratio: list, balanced=True, seed=0):
        train_size, val_size = int(len(dataset) * ratio[0]), int(len(dataset) * ratio[1])
        test_size = len(dataset) - train_size - val_size
        if not os.path.exists(os.path.join(self.root, self.path_dict['split'])):
            train, val = [], []
            train_scaffold_count, val_scaffold_count = 0, 0
            scaffolds = self.get_scaffold(dataset)
            if balanced: # prevent big scaffold from val_data
                index_sets = list(scaffolds.values())
                big_index_sets = []
                small_index_sets = []
                for index_set in index_sets:
                    if len(index_set) > val_size / 2:
                        big_index_sets.append(index_set)
                    else:
                        small_index_sets.append(index_set)
                random.seed(seed)
                random.shuffle(big_index_sets)
                random.shuffle(small_index_sets)
                index_sets = big_index_sets + small_index_sets
            else: # Sort from largest to smallest scaffold sets
                index_sets = sorted(list(scaffolds.values()),
                                    key=lambda index_set: len(index_set),
                                    reverse=True)

            for index_set in index_sets:
                if len(train) + len(index_set) <= train_size:
                    train += index_set
                    train_scaffold_count += 1
                else:
                    val += index_set
                    val_scaffold_count += 1
            split = {'train': train, 'val': val}
            torch.save(split, str(os.path.join(self.root, self.path_dict['split'])))

        else:
            split = torch.load(str(os.path.join(self.root, self.path_dict['split'])))
        return split

    @staticmethod
    def get_scaffold(dataset, includeChirality=False, use_indices=True):
        scaffolds = defaultdict()
        for i in range(len(dataset)):
            smiles = dataset[i].smiles
            mol = Chem.MolFromSmiles(smiles)
            scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=includeChirality)
            if use_indices:
                scaffolds[scaffold].add(i)
            else:
                scaffolds[scaffold].add(mol)
        return scaffolds

    '''def random_split(self, dataset, sizes, banlanced: bool = False, seed=0):
        assert sum(sizes) == 1

        # split
        train_size, val_size = sizes[0] * len(dataset), sizes[1] * len(dataset)
        train_idx, val_idx = [], []
        train_count, val_count = 0, 0'''


class DatasetFP(Dataset):
    def __init__(self, root, path_dict, split='random', transform=None):
        super().__init__()
        self.root = root
        self.split = split
        self.smiles_path = os.path.join(root, path_dict['smiles'])
        self.processed_path = os.path.join(root, path_dict['processed'])
        self.idx2mol_path = self.processed_path[:self.processed_path.find('.lmdb')]+'_idx2mol.pt'

        self.transform = transform
        self.db = None
        self.keys = None

        # if (not os.path.exists(self.processed_path)) or (not os.path.exists(self.idx2mol_path)):
        if not os.path.exists(self.processed_path):
            self._process()
            self._test()

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

    def _close_db(self):
        self.db.close()
        self.db = None
        self.keys = None

    def _process(self):
        db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )

        df_smiles = pd.read_csv(self.smiles_path, index_col=0)

        num_skipped = 0
        num_processed = 0
        idx = 0
        with db.begin(write=True, buffers=True) as txn:
            # for _, line in tqdm(df_smiles.iterrows(), total=len(df_smiles), desc='Preprocessing data'):
                # smiles_list = line['smiles']
            for smiles in tqdm(df_smiles.iterrows(), total=len(df_smiles), desc='Processing SMILES'):
                smiles = smiles[0]
                try:
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
                    morgan = morgan_fp.ToList()
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

                except:
                    num_skipped += 1
                    idx += 1
                    print('Skipping (%d) Num: %s, %s' % (num_skipped, idx, smiles))
                    continue

                '''for idx in range(len(df_smiles)):
                        smiles = smiles_list[idx]

                        mol = Chem.MolFromSmiles(smiles)
                        # MACCS
                        fp_maccs = MACCSkeys.GenMACCSkeys(mol)
                        # mordred_des
                        calc = Calculator(descriptors, ignore_3D=True)
                        descriptors = calc(mol)
                        raw_descriptors = descriptors.values()
                        final_descriptors = list()

                        for raw_descriptor in raw_descriptors:

                            if type(raw_descriptor) is float or type(raw_descriptor) is int:

                                final_descriptors.append(str(raw_descriptor))

                            else:

                                final_descriptors.append("")
                        # morgan_fp
                        morgan_fp = Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius=4, nBits=512,
                                                                                 useBondTypes=True)
                        fingerprints = fp_maccs.ToList() + morgan_fp.ToList()
                        final_descriptors = [0 if x == "" else float(x) for x in final_descriptors]
                        fingerprints = fingerprints + final_descriptors
                        data = get_graph(smiles, fingerprints, idx)

                        txn.put(
                            key=idx,
                            value=pickle.dumps(data)
                        )'''
        db.close()
        print(self.__len__())
        print('Processed %d moleculed' % (len(df_smiles) - num_skipped), 'Skipped %d molecules' % num_skipped)

    def __len__(self):
        if self.db is None:
            self._connect_db()
        return len(self.keys)
    
    # reprocess idx of molecule (need to def)
    '''def redefine_idx(self):
        idx2mol = {}
        for i in tqdm(range(self.__len__()), desc='Redefining index'):
            try:
                data = self.__getitem__(i)
            except AssertionError as e:
                print(i, e)
                continue
            idx = data.idx
            idx2mol[idx] = i
        torch.save(idx2mol, self.idx2mol_path)'''

    def __getitem__(self, i):
        if self.db is None:
            self._connect_db()
        key = self.keys[i]
        data = pickle.loads(self.db.begin().get(key))
        if self.transform is not None:
            data = self.transform(data)
        return data

    def _test(self):
        if self.db is None:
            self._connect_db()
        for i in tqdm(range(self.__len__()), desc='Testing data'):
            key = self.keys[i]
            data = pickle.loads(self.db.begin().get(key))
            if data is None:
                print('Data is None')


def build_datasets(df):
    processed = []
    for idx in tqdm(range(len(df))):
        smiles = df.iloc[idx]['smiles']
        graph = get_graph(smiles, idx)

        if graph is not None:
            processed.append((graph, idx))
    return processed


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))    
    return list(map(lambda s: x == s, allowable_set))


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
    atom_sym = atom.GetSymbol()
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
    # atom_features += [atom.GetMass() * 0.01]

    if not explicit_hydrogen:
        atom_features += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])

    try:
        atom_features += one_of_k_encoding_unk(stereo, ['R', 'S'])
        atom_features += [atom.HasProp('_ChiralityPossible')]
    except Exception as e:
        atom_features += [False, False] + [atom.HasProp('_ChiralityPossible')]
    return list(np.array(atom_features))


def get_bond_features(bond):
    bond_features = []
    bond_features += one_of_k_encoding_unk(bond.GetBondType(),
                                           [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC])
    bond_features += [bond.GetIsAromatic()]
    bond_features += [bond.GetIsConjugated()]
    bond_features += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return bond_features


def get_frag_batch_brics(mol, get_frag_only=False, augmentation=False):
    # smiles = Chem.MolToSmiles(mol)
    [atom.SetAtomMapNum(i + 1) for i, atom in enumerate(mol.GetAtoms())]
    # atom_map = [atom.GetAtomMapNum() for atom in mol.GetAtoms()]

    bonds = [bond[0] for bond in list(BRICS.FindBRICSBonds(mol))] # bond_index should be cleared
    frag_bonds = bonds
    frag_f_edge = []
    for x, y in bonds:
        bond = mol.GetBondBetweenAtoms(x, y)
        f_edge = get_bond_features(bond)
        frag_f_edge.append(f_edge)
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

    if augmentation:
        return frags, node2frag_batch, frag_i_edges, frag_f_edge, frag_idx
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

    return frags, node2frag_batch, frag2graph_batch, edge2frag_batch, edge2frag_size, frag_idx


'''def get_he_graph(smiles, fingerprint=None, idx=0, split_ions=False, atom_message=True, aug_model=None):

    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()

    stereo = Chem.FindMolChiralCenters(mol)
    chiral_centers = [0] * mol.GetNumAtoms()

    node_features = []
    edge_features = []
    bonds = []
    bond_count = 0

    if not aug_model:
        for i, atom_i in enumerate(mol.GetAtoms()):
            atom_features = get_atom_features(atom_i, chiral_centers[i])
            node_features.append(atom_features)

            for j, atom_j in enumerate(mol.GetAtoms()):
                bond_ij = mol.GetBondBetweenAtoms(i, j)
                if bond_ij is not None:
                    bond_features_ij = get_bond_features(bond_ij)

                    if atom_message:
                        edge_features.append(bond_features_ij)
                        edge_features.append(bond_features_ij)
                    else:
                        edge_features.append(get_atom_features(atom_i, chiral_centers[i]) + bond_features_ij)
                        edge_features.append(get_atom_features(atom_j, chiral_centers[j]) + bond_features_ij)


    else:
        aug_model.eval()
        n_real_atoms = n_atoms

        for i, atom_i in enumerate(mol.GetAtoms()):
            atom_features = get_atom_features(atom_i, chiral_centers[i])
            node_features.append(atom_features)

            for j, atom_j in enumerate(mol.GetAtoms()):
                bond_ij = mol.GetBondBetweenAtoms(i, j)
                if bond_ij is not None:
                    bond_features_ij = get_bond_features(bond_ij)

                    if atom_message:
                        edge_features.append(bond_features_ij)
                        edge_features.append(bond_features_ij)
                    else:
                        edge_features.append(get_atom_features(atom_i, chiral_centers[i]) + bond_features_ij)
                        edge_features.append(get_atom_features(atom_j, chiral_centers[j]) + bond_features_ij)


    return n_atoms'''


def get_graph(smiles, fingerprint=None, idx=0, split_ions=False, atom_message=True):

    mol = Chem.MolFromSmiles(smiles)
    n_atoms = mol.GetNumAtoms()

    stereo = Chem.FindMolChiralCenters(mol)
    chiral_centers = [0] * mol.GetNumAtoms()

    for i in stereo:
        chiral_centers[i[0]] = i[1]

    node_features = []
    edge_features = []
    bonds = []
    bond_count = 0

    # featurization for c-mpnn
    a2b, b2a, b2revb = [], [], []
    for _ in range(n_atoms):
        a2b.append([])

    for i, atom_i in enumerate(mol.GetAtoms()):
        atom_features = get_atom_features(atom_i, chiral_centers[i])
        node_features.append(atom_features)

        for j, atom_j in enumerate(mol.GetAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bond_features_ij = get_bond_features(bond_ij)

                if atom_message:
                    edge_features.append(bond_features_ij)
                    edge_features.append(bond_features_ij)
                else:
                    edge_features.append(get_atom_features(atom_i, chiral_centers[i]) + bond_features_ij)
                    edge_features.append(get_atom_features(atom_j, chiral_centers[j]) + bond_features_ij)

                # update index mappings
                b1 = bond_count
                b2 = b1 + 1
                a2b[j].append(b1) # b1 = i --> j, j is coming from b1
                b2a.append(i)# b1 is coming from i
                a2b[i].append(b2) # b2 = j --> i
                b2a.append(j)
                b2revb.append(b1)
                b2revb.append(b2)
                bond_count += 2
                bonds.append([i, j])

    b2a = torch.tensor(b2a, dtype=torch.int64)
    b2revb = torch.tensor(b2revb, dtype=torch.int64)
    a2b_size = []
    a2b_new = []
    for a_b in a2b:
        a2b_size.append(len(a_b))
        a_b = torch.tensor(a_b, dtype=torch.int64)
        a2b_new.append(a_b)
    a2b = torch.cat(a2b_new, dim=0)
    a2b = torch.tensor(a2b, dtype=torch.int64)
    a2b_size = torch.tensor(a2b_size, dtype=torch.int64)

    # padding
    '''atom_fdim = len(node_features[0])
    bond_fdim = len(edge_features[0])

    # Start n_atoms and n_bonds at 1 b/c zero padding
    atom_start = 1
    bond_start = 1
    a_scope = [] # list of tuples indicating (start_atom_index, num_atoms) for each molecule
    b_scope = [] # list of tuples indicating (start_bond_index, num_bonds) for each molecule
    
    f_atoms = [[0] * atom_fdim]
    f_bonds = [[0] * bond_fdim]'''

    '''for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)

        atom_i_features = get_atom_features(atom_i, chiral_centers[i])
        node_features.append(atom_i_features)

        for j in range(mol.GetNumAtoms()):
            bond_ij = mol.GetBondBetweenAtoms(i, j)
            if bond_ij is not None:
                bonds.append([i, j])
                bond_features_ij = get_bond_features(bond_ij)
                edge_features.append(bond_features_ij)'''

    atom_features = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(bonds, dtype=torch.long).T
    edge_features = torch.tensor(edge_features, dtype=torch.float)

    frags, node2frag_batch, frag2graph_batch, edge2frag_batch, edge2frag_size = get_frag_batch_brics(mol, augmentation=False)

    if frags is not None:
        return Data(x=atom_features, edge_index=edge_index, edge_attr=edge_features, y=fingerprint,
                    idx=idx, smiles=Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False),
                    frags=frags, node2frag_batch=node2frag_batch, frag2graph_batch=frag2graph_batch,
                    frags_e=edge2frag_batch.T, frags_e_size=edge2frag_size, a2b=a2b, a2b_size=a2b_size,
                    b2a=b2a, b2revb=b2revb)
    

class MoleculeDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        with open(os.path.join(self.data_dir, 'ZINC_250k/data_shard_ids_0.pkl'), 'rb') as f: # ZINC_250k/data_shard_ids_0.pkl
            self.data_shard_ids = pickle.load(f)
        self.keys = list(self.data_shard_ids.keys())

    
    def get_process_data(self, key):
        lmdb_path = os.path.join(self.data_dir, 'precompute_old1/triplet_{}.lmdb'.format(self.data_shard_ids[key]))
        read_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)

        with read_env.begin(write=False) as txn:
            try:
                processed_data = txn.get(key.encode())
                processed_data = json.loads(processed_data.decode())
            except:
                # blank data
                print(key)
                processed_data = {'a': 'CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1', 'n': 'CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1', 
                                  's': 'CC1(C)CN(C(=O)Nc2cc3ccccc3nn2)C[C@@]2(CCOC2)O1', 'n_w': 0.9635903239250183, 's_w': 0.021186748519539833}
        
        return processed_data
    
    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]

        pdata = self.get_process_data(key)

        smi = pdata['a']
        neg_smi = pdata['n']
        smi_smi = pdata['s']
        '''weight1 = pdata['n_w']
        weight2 = pdata['s_w']'''
        weight1 = pdata['wei1']
        weight2 = pdata['wei2']

        return smi, neg_smi, smi_smi, weight2, weight1


        print(pdata)



