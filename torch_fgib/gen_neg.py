import argparse
from argparse import Namespace
from collections import Counter, defaultdict
import datetime
import sys
sys.path.append('/home/rujinxiao/Desktop/GPSCL-main/torch_fgib')
from fgib import FGIBModel
from pretrain_gpscl import get_args
from data.fgib_data import DatasetFrag, split_mol
from torch_geometric.data import DataLoader
from torch_fgib.fgib_utils import detect_functional_group
from itertools import combinations, chain
from rdkit import Chem
import sqlite3
import re
import numpy as np
import itertools


def gen_neg_dataset():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='/home/rujinxiao/Desktop/GPSCL-main/dataset/ZINC_250k')
    parser.add_argument("--path_dict", type=dict, default={'smiles': 'smiles.csv', 'processed': 'processed.lmdb', 'split': 'scaffold.pkl'})
    args1 = parser.parse_args()
    args2 = get_args()

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    emb_model = FGIBModel.load_from_checkpoint(args2.model_path, args=args2, out_dir=None)
    dataset = DatasetFrag(args1.dataset, args1.path_dict, split='random')
    dataloader = DataLoader(dataset, batch_size=args2.batch_size, shuffle=True)


def gen_neg(smiles, protected_ids, replace_id, replace_frag, stmmetry_fixed=False, index=True, db_path=None):
    """
    smiles: smiles of augmented mol;
    replace_id: id of replaced fragment in mol; should be converted to atom id?
    replace_frag: smiles pf replaced fragment;
    output: augmented molecule;
    """
    mol = Chem.MolFromSmiles(smiles)
    # init index prop
    if index:
        for atom in mol.GetAtoms():
            atom.SetIntProp("Index", atom.GetIdx())
    
    # need frag_id and frag_smiles to avoiding same frag in one molecule
    replace_atom_id = get_atom_id(mol, replace_id, replace_frag)
    neg_sample = mutate_mol(mol, replace_atom_id, replace_frag, db_path)

    return neg_sample
    
    

def get_atom_id(mol, replace_id, replace_frag):
    detect_functional_group(mol)
    split_bonds = split_mol(mol)
    [atom.SetAtomMapNum(i + 1) for i, atom in enumerate(mol.GetAtoms())]
    
    bonds = []
    for bond in split_bonds:
        x, y = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        bonds.append(mol.GetBondBetweenAtoms(x, y).GetIdx())
    frags = Chem.MolToSmiles(Chem.FragmentOnBonds(mol, bonds)).split('.')
    frags.sort(key = lambda frag: (len(frag), str.casefold), reverse=True)
    frags = [Chem.MolFromSmiles(frag) for frag in frags]

    # get atom_ids before validing to avoid atom_map_num lossing
    atom_ids = []
    for atom in frags[int(replace_id)].GetAtoms():
        atom_id = atom.GetAtomMapNum() - 1
        if atom_id != -1:
            atom_ids.append(atom_id)
    frags_test = frags.copy()
    frags_smiles = standard_smiles(frags_test)
    
    assert frags_smiles[int(replace_id)] == replace_frag
    
    return atom_ids

def standard_smiles(frags):
    for frag in frags:
        [atom.SetAtomMapNum(0) for atom in frag.GetAtoms()]

    # standard smiles of frags of mol
    frags_smiles = [Chem.MolToSmiles(frag) for frag in frags]
    frags_smiles = [re.sub(r'\[[0-9]+\*\]', '*', frag) for frag in frags_smiles]
    frags_smiles = [re.sub(r'\*', '[*:1]', frag) for frag in frags_smiles]
    frags_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(frag), isomericSmiles=False) for frag in frags_smiles]

    return frags_smiles


def mutate_mol(mol, replace_ids, replace_frag, db_path, return_frag_smi_only=False):

    replace_ids = set(replace_ids) if replace_ids else set()
    replace_ids = sorted(replace_ids)

    # split to replaced_frag and other components
    mol = Chem.RemoveAllHs(mol)
    mol_smiles = Chem.MolToSmiles(mol)
    bonds_list = []
    for index in replace_ids:
        atom = mol.GetAtomWithIdx(index)
        for nei in atom.GetNeighbors():
            if nei.GetAtomMapNum() - 1 not in replace_ids:
                bonds_list.append(mol.GetBondBetweenAtoms(atom.GetIdx(), nei.GetIdx()).GetIdx())
            else:
                continue

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    frags = Chem.MolToSmiles(Chem.FragmentOnBonds(mol, bonds_list)).split('.')
    frags.sort(key = lambda frag: len(frag), reverse=True)
    # transform * to [0*], if first bond between atom0 and atom1
    frags = [re.sub(r'(?<!\d)\*', '[0*]', frag) for frag in frags]

    
    # obtain atom map to link core and env  
    frags = standardize_attach(frags)
    # test proper atom map number for attach point
    for frag in frags:
        for atom in Chem.MolFromSmiles(frag).GetAtoms():
            if atom.GetAtomMapNum():
                print(atom.GetSymbol())

    # standard smiles of frags of mol
    frags = [re.sub(r'\[[0-9]+\*\]', '*', frag) for frag in frags]
    # frags = [re.sub(r'\*', '[*:1]', frag) for frag in frags]
    frags = [re.sub(r'\d+\*', '*', frag) for frag in frags]
    frags = [Chem.MolToSmiles(Chem.MolFromSmiles(frag), isomericSmiles=False) for frag in frags]
    env = [frag for frag in frags if re.sub(r'\*+\:+\d', '*', frag) != re.sub(r'\*+\:+\d', '*', replace_frag)]
    attn_num = len(env)

    # cut env to a certain radius
    env = get_context_env(env, radius=3)
    env = Chem.MolToSmiles(env)

    # obtain atom map to link core and env
    standardize_att_by_env(env, replace_frag)

    mol_hac = mol.GetNumHeavyAtoms() # orgin molecule
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    # mol_smi = combine_core_env_to_smi(replace_frag, env)
    # row_ids = get_replacements_rowids(cur, env, attn_num)
    new_frag_id = get_replacements_rowid(cur, attn_num)
    
    res = get_replacements(cur, new_frag_id, radius=3)
    # new_mol = link_mol(env, replace_frag)

    '''for rowid, core_smi, core_sma, freq in res:
        if core_smi != replace_frag:
            if return_frag_smi_only:
                yield core_smi
            else:
                yield core_sma, freq'''

    '''detect_functional_group(mol)
    split_bonds = split_mol(mol)
    [atom.SetAtomMapNum(i + 1) for i, atom in enumerate(mol.GetAtoms())]
    bonds = []
    for bond in split_bonds:
        x, y = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        bonds.append(mol.GetBondBetweenAtoms(x, y).GetIdx())
    frags = Chem.MolToSmiles(Chem.FragmentOnBonds(mol, bonds)).split('.')
    frags = [Chem.MolFromSmiles(frag) for frag in frags]'''

    return mol

def get_replacements_rowid(cur_db, attn_num, env, radius, min_freq=0, min_atoms=0, max_atoms=4):
    
    '''sql = f"""SELECT rowid
              From radius{radius}
              WHERE env = '{env}' AND
                    freq >= {min_freq} AND
                    core_num_atoms BETWEEN {min_atoms} AND {max_atoms}"""'''
    sql = f"""SELECT frag
              FROM attn_num{attn_num}
              WHERE core_num_atoms BETWEEN {min_atoms} AND {max_atoms}
              ORDER BY RANDOM() limit 1"""
    
    cur_db.exexute(sql)

    return set(i[0] for i in cur_db.fetchall())

def get_replacements(cur_db, row_ids, radius=0):

    sql = f"""SELECT rowid, core_smi, core_sma, freq
             FROM radius{radius}
             WHERE rowid IN ({','.join(map(str, row_ids))})"""
    cur_db.execute(sql)
    return cur_db.fetchall()

def link_mol(env, core):

    if isinstance(env, str):
        m_env = Chem.MolFromSmiles(env)
    if isinstance(core, str):
        m_core = Chem.MolFromSmiles(core)
    
    m_env.UpdatePropertyCache()
    m = Chem.RWMol(Chem.CombineMols(m_env, m_core))

    links = defaultdict(list)
    att_to_remove = [] # if env is cutted
    for atom in m.GetAtoms():
        if atom.GetAtomMapNum():
            links[atom.GetAtomMapNum()].append(atom.GetIdx())
    
    for i, j in links.values():
        m.AddBond(i, j, Chem.BondType.SINGLE)
    com_smi = mol_to_smi(m)

def mol_to_smi(mol):
    mol = Chem.Mol(mol)
    mol.UpdatePropertyCache()


def standardize_attach(frags_smiles):
    frags = [Chem.MolFromSmiles(frag) for frag in frags_smiles]
    num_frags = len(frags)
    all_number = []
    number_map = defaultdict(list)

    for i, frag_smiles in enumerate(frags_smiles):
        attach_num = find_numbers(frag_smiles)
        number_map[i].extend(attach_num) 
        all_number.extend(attach_num)
    all_number.sort()

    bond_index = get_bond_index(all_number, num_frags, number_map)
    for i in range(len(bond_index)):
        bond = bond_index[i]
        begin_idx = list(get_key(number_map, bond[0]))
        end_idx = list(get_key(number_map, bond[1]))
        frags = set_attach(frags, i, begin_idx, end_idx)
    frags = [Chem.MolToSmiles(frag) for frag in frags]

    return frags

def find_numbers(str):
    return [int(num) for num in re.findall(r'(\d+)\*', str)]

def get_bond_index(attach_list, num_frags, number_map):

    if num_frags == 2:
        array = np.array(attach_list).reshape(-1, 2)
        bond_list = list(array)
    else:
        core_idx = max(number_map, key=lambda x: len(number_map[x]))
        core_number = list(number_map[core_idx])
        other_number = list(chain(*(number_map[key] for key in number_map.keys() if key != core_idx)))
        core_number.sort()
        other_number.sort()
        bond_list = list(zip(*[core_number, other_number]))
        # bond_list = list(map(list, zip([core_number.sort(), other_number.sort()])))

    return bond_list

def get_bond_index_old(attach_list, num_frags, number_map):
    array = np.array(attach_list).reshape(-1, 2)
    bond_list = list(array)

    for i in range(len(bond_list)):
        if bond_list[i][0] == bond_list[i][1]: # need to devide bonds all from one atom
            core_number = [attach_list[i][0]]
            core_number, count = Counter(attach_list).most_common(1)[0]
            if count == num_frags: # all need from the same atom
                other_number = [number for number in attach_list if number != core_number]
                bond_iter = itertools.product(core_number, other_number)
                bond_list = [elem for elem in bond_iter]
            elif count < num_frags: # not all need from the same atom
                core_idx = max(number_map, key=lambda x: len(number_map[x]))
                core_number = number_map[core_idx]
                # considering easy method to create bond_index
    return bond_list

def get_key(dict, value):
    return set([k for k, v in dict.items() if value in v])

# set appoach to sovle idx is not len() == 1
def set_attach(frags, bond_idx, idx1, idx2):
    for idx in idx1:
        for atom in frags[idx].GetAtoms():
            if atom.GetSymbol() == '*' and not atom.GetAtomMapNum():
                atom.SetAtomMapNum(bond_idx + 1)
                break
        else:
            continue
        break
    
    for idx in idx2:
        for atom in frags[idx].GetAtoms():
            if atom.GetSymbol() == '*' and not atom.GetAtomMapNum():
                atom.SetAtomMapNum(bond_idx + 1)
                break

        else:
            continue
        break
    
    for frag in frags:
        print(Chem.MolToSmiles(frag))

    return frags
    
    
    '''attach_map = {}
    # obtain attach_point map and nei atom
    if len(frags) == 2:
        for frag in frags:
            for atom in Chem.MolFromSmiles(frag).GetAtoms():
                if atom.GetSymbol() == '*':
                    atom.SetIntProp('attach', 1)
                else:
                    atom.SetIntProp("attach", 0)
    else:
        att_num = len(frags - 1)
        all_nei_atom = []
        for i, frag in enumerate(frags):
            attach_map[str(i)] = []
            for atom in Chem.MolFromSiles(frag).GetAtoms():
                if atom.GetSymbol() == '*':
                    neighbor = atom.GetNeighbors()[0]
                    all_nei_atom.append(neighbor.GetAtomMapNum())
                    attach_map[str(i)].append(neighbor.GetAtomMapNum)
        
        # has no repeat 
        if len(set(all_nei_atom)) == len(all_nei_atom):
            all_nei_atom.sort()
            attach_map = {value:key for key, value in attach_map.items()}
            for i in range(len(all_nei_atom)):
                frag_index = int(attach_map[all_nei_atom[i]])
                for atom in Chem.MolFromSiles(frags[frag_index]).GetAtoms():
                    if atom.GetSymbol() == '*':
                        atom.SetIntProp('attach', i+1)'''

    '''for i, frag in enumerate(frags):
        attach_map[str(i)] = []
        nei_map[str(i)] = []
        for atom in Chem.MolFromSmiles(frag):
            if atom.GetSymbol() == '*':
                attach_map_num = atom.GetAtomMapNum()
                attach_map[str(i)].append(attach_map_num)
                if len(atom.GetNeighbors()) == 1:
                    nei_map_num = atom.GetNeighbors().GetAtomMapNum()
                    nei_map[str(i)].append(nei_map_num)
                else:
                    raise('The number of neighbors is too large')'''
                    

def combine_core_env_to_smi(core, env):
    if isinstance(env, list):
        if len(env) == 1:
            env = env[0]
        else:
            env = '.'.join(env)
    m_env = Chem.MolFromSmiles(env, sanitize=False)
    m_frag = Chem.MolFromSmiles(core, sanitize=False)

    m_env.UpdatePropertyCache()
    m = Chem.RWMol(Chem.CombineMols(m_env, m_frag))

    links = defaultdict(list)
    for atom in m.GetAtoms():
        if atom in m.GetAtoms():
            links[atom.GetAtomMapNum()].append(atom.GetIdx())
    
    for i, j in links.values():
        m.AddBond(i, j, Chem.BondType.SINGLE)
    com_smi = mol_to_smi(m)

    return com_smi

def get_context_env(env, radius=3):
    # convert smiles_list to a smiles
    if len(env) != 1:
        env = '.'.join(env)
    else:
        env = env[0]
    env = Chem.RemoveHs(Chem.MolFromSmiles(env))
    env = Chem.RWMol(env)

    bonds_ids = set()
    for a in env.GetAtoms():
        if a.GetSymbol() == '*':
            i = radius
            bond = Chem.FindAtomEnvironmentOfRadiusN(env, i, a.GetIdx())
            while not bond and i > 0:
                i -= 1
                bond = Chem.FindAtomEnvironmentOfRadiusN(env, i, a.GetIdx())
            bonds_ids.update(bond)
    
    atom_ids = set(bonds_to_atoms(env, bonds_ids))

    dummy_atoms = []

    # growing for cutting atom
    for a in env.GetAtoms():
        if a.GetIdx() not in atom_ids:
            nei_ids = set(na.GetIdx() for na in a.GetNeighbors())
            intersect = nei_ids & atom_ids
            if intersect:
                dummy_atom_bonds = []
                for ai in intersect:
                    dummy_atom_bonds.append((ai, env.GetBondBetweenAtoms(a.GetIdx(), ai).GetBondType()))
                dummy_atoms.append(dummy_atom_bonds)
    # replace cutting atom-atom with atom-H
    for data in dummy_atoms:
        dummy_id = env.AddAtom(Chem.Atom(0))
        for atom_id, bond_type in data:
            env.AddBond(dummy_id, atom_id, bond_type)
        atom_ids.add(dummy_id)
    # cut env to obtain submol in radius
    env = get_submol(mol=env, atom_ids=atom_ids)

    return env

def get_submol(mol, atom_ids):
    bond_ids = []
    # combinations: combine each two (排列组合)
    for pair in combinations(atom_ids, 2):
        b = mol.GetBondBetweenAtoms(*pair)
        if b:
            bond_ids.append(b.GetIdx())
    m = Chem.PathToSubmol(mol, bond_ids) # obtain structures between breakpoint
    m.UpdatePropertyCache()
    return m

def bonds_to_atoms(mol, bond_ids):
    output = []
    for i in bond_ids:
        b = mol.GetBondWithIdx(i)
        output.append(b.GetBeginAtom().GetIdx())
        output.append(b.GetEndAtom().GetIdx())
    return tuple(set(output))

def standardize_att_by_env(env, core):
    maps, ranks = get_maps_and_ranks(env)


def get_maps_and_ranks(env, keep_stereo=False):
    """
    Return the list of attachment point map numbers and
    the list of canonical SMILES without mapped attachment points (ranks)
    """
    tmp_mol = Chem.Mol(env)
    maps = []
    ranks = []
    for comp in Chem.GetMolFargs(tmp_mol, asMols=True, sanitizeFrags=False):
        for a in comp.GetAtoms():
            atom_num = a.GetAtomMapNum()
            if atom_num:
                maps.append(atom_num)
                a.SetAtomMapNum(0)
                break
        ranks.append(Chem.MolToSmiles(comp, isomericSmiles=keep_stereo))
    return maps, ranks


