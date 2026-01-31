import os
import lmdb
import numpy as np
import argparse
from tqdm import tqdm
from torch_facl.facl_utils import MolData, BatchMolData
import yaml
import warnings
from argparse import Namespace
import torch
from rdkit import Chem
from typing import DefaultDict
from torch_fgib.fgib_utils import detect_functional_group
from data.fgib_data import split_mol, replace_isform, restore_isform
from rdkit import DataStructs
from create_frag_env_db import standardize_attach, standard_env, _permute_att, _standardize_smiles_with_att_points, _get_att_permutations
import re
from context_utils import combine_core_env_to_rxn_smarts
from collections import defaultdict
from itertools import combinations, chain
from rdkit.Chem import AllChem
import sys
import sqlite3
import json
import pickle
import random



def merge_args_and_yaml(args, config_dict):
    arg_dict = args.__dict__
    for key, value in config_dict.items():
        if key in arg_dict:
            warnings.warn(f"Command line argument '{key}' (value: "
                          f"{arg_dict[key]}) will be overwritten with value "
                          f"{value} provided in the config file.")
        if isinstance(value, dict):
            arg_dict[key] = Namespace(**value)
        else:
            arg_dict[key] = value
    return args


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/gpscl_config.yml')
    parser.add_argument('--resume', type=str, default=None)
    # Trainer
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--enable_progress_bar', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--augmentation', type=bool, default=True)
    # emb_model
    parser.add_argument('--model_path', type=str,
                        default='./checkpoints/fgib/best_emb_checkpoint.pt')
    parser.add_argument('--step', type=str, default='pretrain', choices=['pretrain', 'finetune'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    args = merge_args_and_yaml(args, config)
    return args


def get_smi_weight(neg_smiles, smi_smiles):

    fps1 = [Chem.RDKFingerprint(Chem.MolFromSmiles(x)) for x in neg_smiles]
    fps2 = [Chem.RDKFingerprint(Chem.MolFromSmiles(x)) for x in smi_smiles]

    weights =[]
    for i in range(len(fps1)):
        weight = DataStructs.FingerprintSimilarity(fps1[i], fps2[i])
        weights.append(weight)

    return weights


def compute(smiles_list, emb_model, db_path=None, shard_id=None):
    """
        smiles_list: input smiles of anchor;
        emb_model: frag_emb model;
        db_path: path of lmdb file of fragment;
    """
    mol_datas = []
    for smiles in smiles_list:
        mol_data = MolData(Chem.MolFromSmiles(smiles))
        mol_datas.append(mol_data)
    batch = BatchMolData(mol_datas)

    # obtain frah_emb using frag_emb model trained
    frag_emb, frag_size, frag_weight = emb_model(batch, augmentation=True)
    pos_index, neg_index, pos_frags, neg_frags, max_weis, min_weis, smiles_list = batch.get_aug_frags(frag_weight)
    # mol_triplet contains all sample smiles of anchor
    mol_triplet = DefaultDict(list)
    n_frags = 0

    for i, smiles in enumerate(smiles_list):
        mol_triplet['a'].append(smiles)
        n_frags = frag_size[i]

        protected_index = set(range(n_frags))

        # obtain negative sample of anchor
        for ind in range(len(pos_index[i])):
            replace_id = pos_index[i][ind]
            replace_frag = pos_frags[i][ind]
            protected_neg = protected_index.remove(int(replace_id))
            neg_sample = gen_sample(smiles, protected_ids=protected_neg, replace_frag=replace_frag, replace_id=replace_id, db_path=db_path)
            neg_wei = max_weis[i][ind]
            if neg_sample:
                break

        if not neg_sample:
            print(smiles, 'neg_sample failed in {}'.format(shard_id))
            mol_triplet['n'].append(smiles)
        else:
            mol_triplet['n'].append(neg_sample)
            mol_triplet['n_w'].append(neg_wei)

        # obtain same sample of anchor
        for ind in range(len(neg_index[i])):
            replace_id_smi = neg_index[i][ind]
            replace_frag_smi = neg_frags[i][ind]
            protected_smi = protected_index.remove(int(replace_id_smi))
            smi_sample = gen_sample(smiles, protected_ids=protected_smi, replace_frag=replace_frag_smi, replace_id=replace_id_smi, db_path=db_path)
            sim_wei = min_weis[i][ind]
            if smi_sample:
                break
        
        if not smi_sample:
            print(smiles, 'smi_sample failed in {}'.format(shard_id))
            mol_triplet['s'].append(smiles)
        else:
            mol_triplet['s'].append(smi_sample)
            mol_triplet['s_w'].append(sim_wei)

    '''smi_weights = get_smi_weight(mol_triplet['n'], mol_triplet['s'])
    neg_weights = get_smi_weight(mol_triplet['n'], mol_triplet['a'])
    mol_triplet['weight1'].extend(smi_weights)
    mol_triplet['weight2'].extend(neg_weights)'''

    return mol_triplet

def gen_sample(smiles, protected_ids, replace_id, replace_frag, stmmetry_fixed=False, index=True, db_path=None):
    """
    smiles: smiles of augmented mol;
    replace_id: id of replaced fragment in mol; should be converted to atom id?
    replace_frag: smiles pf replaced fragment;
    output: augmented molecule;
    """
    mol = Chem.MolFromSmiles(smiles)
    if index:
        for atom in mol.GetAtoms():
            atom.SetIntProp("Index", atom.GetIdx())
    
    replace_atom_id = get_atom_id(mol, replace_id, replace_frag)

    try:
        new_sample = mutate_mol(mol, replace_atom_id, replace_frag, db_path)
    except:
        if replace_frag != 'N#[*:1]':
            print("Smiles %s couldn't be mutated because %s" %(smiles, replace_frag))
        return None

    return new_sample


def get_atom_id(mol, replace_id, replace_frag):

    '''if replace_frag == 'c1nc([*:1])[nH]c1[*:1]':
        print(replace_frag)'''
    detect_functional_group(mol)
    split_bonds = split_mol(mol)
    [atom.SetAtomMapNum(i + 1) for i, atom in enumerate(mol.GetAtoms())]
    
    bonds = []
    for bond in split_bonds:
        x, y = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        bonds.append(mol.GetBondBetweenAtoms(x, y).GetIdx())
    frags = Chem.MolToSmiles(Chem.FragmentOnBonds(mol, bonds)).split('.')
    frags = replace_isform(frags)
    frags.sort(key = lambda frag: len(frag), reverse=True)
    frags = restore_isform(frags)
    frags = [Chem.MolFromSmiles(frag) for frag in frags]

    # get atom_ids before validing to avoid atom_map_num lossing
    atom_ids = []
    for atom in frags[int(replace_id)].GetAtoms():
        atom_id = atom.GetAtomMapNum() - 1
        if atom_id != -1:
            atom_ids.append(atom_id)
    frags_test = frags.copy()
    frags_smiles = standard_smiles(frags_test)
    
    '''if frags_smiles[int(replace_id)] != replace_frag:
        print(replace_frag)'''


    assert frags_smiles[int(replace_id)] == replace_frag
    
    return atom_ids

def mutate_mol(mol, replace_ids, replace_frag, db_path, radius=3, min_inc=-2, dist=None,
               max_inc=2, sample_func=None, keep_stereo=False, return_frag_smi_only=False):

    output = []
    replace_ids = set(replace_ids) if replace_ids else set()
    replace_ids = sorted(replace_ids)
    
    # split to replaced_frag and other components

    mol = Chem.RemoveAllHs(mol)
    mol_smiles = Chem.MolToSmiles(mol)
    mol_hac = mol.GetNumHeavyAtoms()
    bonds_list = []
    for index in replace_ids:
        atom = mol.GetAtomWithIdx(index)
        for nei in atom.GetNeighbors():
            if nei.GetAtomMapNum() - 1 not in replace_ids:
                bonds_list.append(mol.GetBondBetweenAtoms(atom.GetIdx(), nei.GetIdx()).GetIdx())
            else:
                continue
    
    core_env = Chem.MolToSmiles(Chem.FragmentOnBonds(mol, bonds_list)).split('.')
    env_ori, core = standardize_attach(core_env, replace_ids)
    att_num = core.count('*')

    while len(output) == 0:

        env = env_ori.copy()
        env = standard_env(env, radius, att_num)
        if radius == 0:
            env_smi = env
        else:
            env_smi = Chem.MolToSmiles(env, isomericSmiles=keep_stereo, allBondsExplicit=True)
        core = Chem.MolFromSmiles(core)
        # num_heavy_atoms = core.GetNumHeavyAtoms()
        for a in core.GetAtoms():
            if a.GetSymbol() != '*':
                a.SetAtomMapNum(0)

        if att_num == 1 or radius == 0:
            core = _standardize_smiles_with_att_points(core, keep_stereo)

        else:
            res = []
            p = _get_att_permutations(env)

            if len(p) > 1:
                for d in p:
                    c = _permute_att(core, d)
                    res.append(c)
            
            else:
                res.append(core)

            core = sorted(tuple(set(_standardize_smiles_with_att_points(m, keep_stereo) for m in res)))[0]
        
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        replacements = dict()
        returned_values = 0
        preliminary_return = 1

        num_heavy_atoms = Chem.MolFromSmiles(core).GetNumHeavyAtoms()
        hac_ratio = num_heavy_atoms / mol_hac
        min_atoms = num_heavy_atoms + min_inc
        max_atoms = num_heavy_atoms + max_inc

        '''if env_smi == '3':
            print(env_smi)'''

        if radius == 0:

            row_ids = _get_replacements_rowids(cur, env_smi, core, min_atoms, max_atoms, radius, dist=None)

            res = _get_replacements(cur, radius, row_ids)

            n = min(len(row_ids), preliminary_return)

            if sample_func is not None:
                selected_row_ids = sample_func(list(row_ids), cur, radius, n)
            else:
                selected_row_ids = random.sample(list(row_ids), n)

            row_ids.difference_update(selected_row_ids)
            res = _get_replacements(cur, radius, selected_row_ids)

            output = link_mol(env_ori, res[0][1], att_num)

        else:

            frag_sma = combine_core_env_to_rxn_smarts(core, env_smi)

            row_ids = _get_replacements_rowids(cur, env_smi, core, min_atoms, max_atoms, radius, dist=dist)

            res = _get_replacements(cur, radius, row_ids)
            n = min(len(row_ids), preliminary_return)

            if sample_func is not None:
                selected_row_ids = sample_func(list(row_ids), cur, radius, n)
            else:
                selected_row_ids = random.sample(list(row_ids), n)
            row_ids.difference_update(selected_row_ids)
            res = _get_replacements(cur, radius, selected_row_ids)

            '''for row_id, core_smi, core_sma in res:
                for smi, m, rxn in _frag_replace(mol, frag_sma, core_sma, radius, replace_ids):
                    output.append(smi)'''
            if res:
                output = link_mol(env_ori, res[0][1], att_num)
            
            radius -= 1

    output = output[0]

    return output


    '''# old 
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
    if len(frags) == 2:
        env = [frag for frag in frags if re.sub(r'\*+\:+\d', '*', frag) != re.sub(r'\*+\:+\d', '*', replace_frag)]
    else:
        frag_att_num = [len(re.findall(r'\*', frag)) for frag in frags]
        frags.pop(frag_att_num.index(max(frag_att_num)))
        env = frags
    attn_num = len(env)

    con = sqlite3.connect(db_path)
    cur = con.cursor()

    if attn_num == 1:
        env = env[0]
    else:
        env = '.'.join(env)
        
    for i in range(10):
        frag, num_atoms, att_num = get_replacements_row(cur, attn_num)

        sample = link_mol(frag, env, att_num)
        
        if sample is not None:
            break

    return sample'''


def _frag_replace(mol, frag_sma, replace_sma, radius, frag_ids=None):

    def set_protected_atoms(mol, ids, radius):

        def extend_ids(mol, atom_id, r, ids):
            if r:
                for a in mol.GetAtomWithIdx(atom_id).GetNeighbors():
                    a_id = a.GetIdx()
                    if a_id not in ids:
                        ids.add(a_id)
                    extend_ids(mol, a_id, r-1, ids)

        if ids:
            ids_ext = set(ids)
            # extend atom ids on neighbour atoms
            for i in ids:
                extend_ids(mol, i, radius + 1, ids_ext)
            # protect untouched atoms
            for a in mol.GetAtoms():
                if a.GetAtomicNum() > 1 and a.GetIdx() not in ids_ext:
                    a.SetProp('_protected', '1')
                else:
                    a.ClearProp('_protected')


    frag_sma = frag_sma.replace('*', '!#1')

    rxn_sma = "%s>>%s" % (frag_sma, replace_sma)
    rxn = AllChem.ReactionFromSmarts(rxn_sma)

    set_protected_atoms(mol, frag_ids, radius)
    products = set()
    mol_smiles = Chem.MolToSmiles(mol)
    reactants = [mol]
    ps = rxn.RunReactants(reactants)
    for y in ps:
        for p in y:
            e = Chem.SanitizeMol(p, catchErrors=True)
            if e:
                sys.stderr.write("Molecule %s caused sanitization error %i" % (Chem.MolToSmiles(p, isomericSmiles=True), e))
                sys.stderr.flush()
            else:
                smi = Chem.MolToSmiles(Chem.RemoveHs(p), isomericSmiles=True)
                if smi not in products:
                    products.add(smi)
                    yield smi, p, rxn_sma


def _get_replacements_rowids(db_cur, env, core, min_atoms, max_atoms, radius, min_freq=0, dist=None, **kwargs):
    sql = f"""SELECT rowid
              FROM radius{radius}
              WHERE env = '{env}' AND
                    core_smi != '{core}' AND
                    core_num_atoms BETWEEN {min_atoms} AND {max_atoms}"""
    if isinstance(dist, int):
        sql += f" AND dist2 = {dist}"
    elif isinstance(dist, tuple) and len(dist) == 2:
        sql += f" AND dist2 BETWEEN {dist[0]} AND {dist[1]}"
    for k, v in kwargs.items():
        if isinstance(v, tuple) and len(v) == 2:
            sql += f" AND {k} BETWEEN {v[0]} AND {v[1]}"
        else:
            sql += f" AND {k} = {v}"
    db_cur.execute(sql)
    return set(i[0] for i in db_cur.fetchall())


def get_replacements_row(cur_db, attn_num):

    table_name = 'att_num%i' % attn_num
    query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 1"
    cur_db.execute(query)

    row = cur_db.fetchone()
    
    frag, num_atoms, att_num = row

    return frag, num_atoms, att_num

def _get_replacements(db_cur, radius, row_ids):
    if radius == 0:
        sql = f"""SELECT rowid, core_smi
                FROM radius{radius}
                WHERE rowid IN ({','.join(map(str, row_ids))})"""
    else:
        sql = f"""SELECT rowid, core_smi, core_sma
                FROM radius{radius}
                WHERE rowid IN ({','.join(map(str, row_ids))})"""
    db_cur.execute(sql)
    return db_cur.fetchall()


def find_numbers(str):
    all_attach_indices = [i for i, char in enumerate(str) if char == '*']
    assert all_attach_indices
    
    att_num_map = {}
    pattern = r'(\d+)\s*\*'

    for match in re.finditer(pattern, str):
        num = int(match.group(1))
        att_index = match.end() - 1
        att_num_map[att_index] = num
    # numbers = [int(num) for num in re.findall(r'(\d+)\*', str)]
    return [att_num_map.get(index, 0) for index in all_attach_indices]


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

def link_mol(replacement, core, att_num):
    """link new fragment and core of anchor"""
    
    if len(replacement) == 1:
        env = replacement[0]
    else:
        env = '.'.join(replacement)

    if isinstance(env, str):
        m_env = Chem.MolFromSmiles(env)
    if isinstance(core, str):
        m_core = Chem.MolFromSmiles(core)

    for atom in m_env.GetAtoms():
        atom_sym = atom.GetSymbol()
        atom_map = atom.GetAtomMapNum()
        if atom.GetSymbol() != '*':
            atom.SetAtomMapNum(0)
    for atom in m_core.GetAtoms():
        if atom.GetSymbol() == '*':
            atom.SetAtomMapNum(0)
    
    # standardize attachment of replacement(all attachment of replacement is 1).
    '''map_ind = 1
    if att_num > 1:
        for atom in m_env.GetAtoms():
            if atom.GetSymbol() =='*':
                atom.SetAtomMapNum(map_ind)
                map_ind += 1
            if map_ind == att_num + 1:
                break'''

    mol = Chem.RWMol(Chem.CombineMols(m_env, m_core))
    test_smiles = Chem.MolToSmiles(mol)
    att_to_remove = []
    core_att, env_att = [], []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == '*':
            nei_id = atom.GetNeighbors()[0].GetIdx()
            if atom.GetAtomMapNum():
                env_att.append((nei_id, mol.GetBondBetweenAtoms(atom.GetIdx(), nei_id).GetBondType()))
            else:
                core_att.append((nei_id, mol.GetBondBetweenAtoms(atom.GetIdx(), nei_id).GetBondType()))
            att_to_remove.append(atom.GetIdx())
    
    assert len(env_att) == len(core_att)
    for i in range(len(env_att)):
        for j in range(len(core_att)):
            if env_att[i][1] == core_att[j][1]:
                mol.AddBond(env_att[i][0], core_att[j][0], env_att[i][1])
                core_att.pop(j)
                break
            else:
                continue
    
    
    '''links = defaultdict(list)
    att_to_remove = []

    for atom in mol.GetAtoms():
        if atom.GetAtomMapNum():
            # map = atom.GetAtomMapNum()
            # symbol = atom.GetSymbol()
            nei_id = atom.GetNeighbors()[0].GetIdx()
            links[atom.GetAtomMapNum()].append(
                (nei_id, mol.GetBondBetweenAtoms(atom.GetIdx(), nei_id).GetBondType()))
            att_to_remove.append(atom.GetIdx())
    
    for i, j in links.values():
        if i[1] == j[1]:
            mol.AddBond(i[0], j[0], i[1])
        else:
            return None'''

    for i in sorted(att_to_remove, reverse=True):
        mol.RemoveAtom(i)

    sample_smi = mol_to_smi(mol)

    return [sample_smi]

def mol_to_smi(rwmol):
    mol = Chem.Mol(rwmol)
    mol.UpdatePropertyCache()
    smi = Chem.MolToSmiles(mol, isomericSmiles=True, allBondsExplicit=True)
    return smi


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


def standard_smiles(frags):
    for frag in frags:
        [atom.SetAtomMapNum(0) for atom in frag.GetAtoms()]

    # standard smiles of frags of mol
    frags_smiles = [Chem.MolToSmiles(frag) for frag in frags]
    frags_smiles = [re.sub(r'\[[0-9]+\*\]', '*', frag) for frag in frags_smiles]
    frags_smiles = [re.sub(r'\*', '[*:1]', frag) for frag in frags_smiles]
    frags_smiles = [Chem.MolToSmiles(Chem.MolFromSmiles(frag), isomericSmiles=False) for frag in frags_smiles]

    return frags_smiles


def attach_mol(core, frag):
    if isinstance(frag, str):
        m_frag = Chem.MolFromSmiles(frag, sanitize=False)
    if isinstance(core, str):
        m_core = Chem.MolFromSmiles(core, sanitize=False)

    m = Chem.RWMol(Chem.CombineMols(m_core, m_frag))
    m_smiles = Chem.MolToSmiles(m)

    links = defaultdict(list) 
    att_to_remove = []
    

def main(args):
    ids, smiles, is_perturb_valid = [], [], []

    # get emb_model
    args_emb = get_args()
    emb_model = torch.load(args_emb.model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    emb_model.to(device)

    with open(args.smiles_path, 'r') as f:
        for i, line in enumerate(f.readlines()):
            if i == 0:
                continue
            smiles.append(line.strip())
            ids.append('z{}'.format(i))
    
    key2shard_id = {}
    chunk_size = 16
    for shard_id in tqdm(range(0, len(smiles)//chunk_size + 1)):
        smiles_shard = smiles[shard_id*chunk_size:(shard_id+1)*chunk_size]
        smiles_ids = ids[shard_id*chunk_size:(shard_id+1)*chunk_size]
        os.makedirs(os.path.join(args.save_dir, 'precompute_new'), exist_ok=True)
        env = lmdb.open(os.path.join(args.save_dir, 'precompute_new/triplet_{}.lmdb'.format(shard_id)), map_size=1099511627776)


        with env.begin(write=True) as txn:
            output = compute(smiles_shard, emb_model, db_path=args.fragment_path, shard_id=shard_id)
            mol_data = {'anchor': smiles_shard, 'neg': output['n'], 'smi': output['s'], 
                        'ids': smiles_ids, 'n_w': output['n_w'], 's_w': output['s_w']}
            for i in range(chunk_size):
                key = mol_data['ids'][i].encode()
                try:
                    txn.put(key, json.dumps({'a': mol_data['anchor'][i], 'n': mol_data['neg'][i],
                                             's': mol_data['smi'][i], 'n_w': mol_data['n_w'][i], 's_w':mol_data['s_w'][i]}).encode())
                    key2shard_id[key.decode()] = shard_id
                except:
                    print('[warning] ignoring smiles {}.'.format(key.decode()))
                
                '''for i in range(chunk_size):
                    try:
                        key = smiles_ids[i].encode()
                        key2shard_id[key.decode()] = shard_id
                    except:
                        break'''
                
    
    with open(os.path.join(args.save_dir, 'data_shard_ids.pkl'), 'wb') as f:
        pickle.dump(key2shard_id, f)

class FilterFunc():
    def __init__(self, core_smi):
        self.core = core_smi
    
    def forward(row_ids, cur, radius):
        return row_ids
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_path', type=str, default='./dataset/ZINC_250k/smiles.csv')
    parser.add_argument('--fragment_path', type=str, default='./dataset/ZINC_250k/core_fragments.db',
                        help='path of fragment database')
    parser.add_argument('--save_dir', type=str, default='./dataset/', 
                        help='dir to save')
    args = parser.parse_args()
    main(args)




































