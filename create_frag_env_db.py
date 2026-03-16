import argparse
import pandas as pd
import numpy as np
import re
import sqlite3
from rdkit import Chem
from torch_fgib.fgib_utils import detect_functional_group
from data.fgib_data import split_mol
from torch_fgib.context_utils import combine_core_env_to_rxn_smarts
from tqdm import tqdm
from itertools import product, permutations, combinations, chain
from collections import defaultdict



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smiles_path', type=str, required=True, default='/Path/to/smiles.csv')
    parser.add_argument('--out_txt', type=str, default='/Path/to/logdir/frag_fgib_inmap.txt')
    parser.add_argument('--mode', type=int, choices=[0, 1, 2], default=0, help='0 -- all atoms constitute a fragment, 1 -- only heavy atoms, 2 -- only Hs')
    parser.add_argument('--db_path', type=str, required=True, default='/Path/to/core_fragments.db')
    parser.add_argument('--radius', type=int, required=True, default=0, help='radius of context to generate')
    args = parser.parse_args()

    df_smiles = pd.read_csv(args.smiles_path)

    create_frag_env_db(df_smiles, args.db_path, args.radius)


def _get_submol(mol, atom_ids):
    bond_ids = []
    # combinations: combine each two (排列组合)
    for pair in combinations(atom_ids, 2):
        b = mol.GetBondBetweenAtoms(*pair)
        if b:
            bond_ids.append(b.GetIdx())
    m = Chem.PathToSubmol(mol, bond_ids) # obtain structures between breakpoint
    m.UpdatePropertyCache()
    return m

def __standardize_att_by_env(env, core, keep_stereo=False):
    """
    Set attachment point numbers in core and context according to canonical ranks of attachment points in context
    Ties are broken
    Makes changes in place
    """
    maps, ranks = __get_maps_and_ranks(env, keep_stereo)
    new_att = {m: i+1 for i, (r, m) in enumerate(sorted(zip(ranks, maps)))}
    __replace_att(core, new_att)
    __replace_att(env, new_att)


def __replace_att(mol, repl_dict):
    for a in mol.GetAtoms():
        map_num = a.GetAtomMapNum()
        if map_num in repl_dict:
            a.SetAtomMapNum(repl_dict[map_num])


def __get_maps_and_ranks(env, keep_stereo=False):
    """
    Return the list of attachment point map numbers and
    the list of canonical SMILES without mapped attachment points (ranks)
    """
    tmp_mol = Chem.Mol(env)
    env_test = Chem.MolToSmiles(env)
    maps = []
    ranks = []
    '''for atom in env.GetAtoms():
        atom_map = atom.GetAtomMapNum()
        atom_idx = atom.GetIdx()
        atom_sym = atom.GetSymbol()'''
    for comp in Chem.GetMolFrags(tmp_mol, asMols=True, sanitizeFrags=False):
        for a in comp.GetAtoms():
            atom_num = a.GetAtomMapNum()
            atom_sym = a.GetSymbol()
            if atom_num and atom_sym == '*':
                maps.append(atom_num)
                a.SetAtomMapNum(0)
                break
        ranks.append(Chem.MolToSmiles(comp, isomericSmiles=keep_stereo))
    return maps, ranks


def __bonds_to_atoms(mol, bond_ids):
    output = []
    for i in bond_ids:
        b = mol.GetBondWithIdx(i)
        output.append(b.GetBeginAtom().GetIdx())
        Beginsym = b.GetBeginAtom().GetSymbol()
        output.append(b.GetEndAtom().GetIdx())
        Endsym = b.GetEndAtom().GetSymbol()
    return tuple(set(output))


def standard_env(env, radius, attn_num, keep_stereo=False):
    if len(env) == 1:
        env = env[0]
    else:
        env = '.'.join(env)

    rw_env = Chem.RWMol(Chem.RemoveAllHs(Chem.MolFromSmiles(env)))

    bond_ids = set()
    bonds_type = []

    if radius == 0:
        for atom in rw_env.GetAtoms():
            if atom.GetSymbol() == '*':
                nei = atom.GetNeighbors()[0]
                bond = rw_env.GetBondBetweenAtoms(atom.GetIdx(), nei.GetIdx())
                bond_type = str(int(bond.GetBondType()))
                bonds_type.append(bond_type)
        if len(bonds_type) == 1:
            env = bonds_type[0]
        else:
            env = '.'.join(bonds_type)
    else:

        for atom in rw_env.GetAtoms():
            if atom.GetSymbol() == '*':
                i = radius
                b = Chem.FindAtomEnvironmentOfRadiusN(rw_env, i, atom.GetIdx())
                while not b and i > 0:
                    i -= 1
                    b = Chem.FindAtomEnvironmentOfRadiusN(rw_env, i, atom.GetIdx())
                bond_ids.update(b)
        
        atom_ids = set(__bonds_to_atoms(rw_env, bond_ids))

        dummy_atoms = []

        for a in rw_env.GetAtoms():
            if a.GetIdx() not in atom_ids:
                nei_ids = set(na.GetIdx() for na in a.GetNeighbors())
                intersect = nei_ids & atom_ids
                if intersect:
                    dummy_atom_bonds = []
                    for ai in intersect:
                        dummy_atom_bonds.append((ai, rw_env.GetBondBetweenAtoms(a.GetIdx(), ai).GetBondType()))
                    dummy_atoms.append(dummy_atom_bonds)
        
        for data in dummy_atoms:
            dummy_id = rw_env.AddAtom(Chem.Atom(0))
            for atom_id, bond_type in data:
                rw_env.AddBond(dummy_id, atom_id, bond_type)
            atom_ids.add(dummy_id)

        # test for error
        # env_smi_test = Chem.MolToSmiles(_get_submol(rw_env, atom_ids))
        env = Chem.RemoveAllHs(_get_submol(rw_env, atom_ids), sanitize=False)

        # conert map of heavy atom to 0 
        for a in env.GetAtoms():
            if a.GetSymbol() != '*':
                a.SetAtomMapNum(0)
        
    return env

def create_frag_env_db(df_smiles, out_path, radius, keep_stereo=False, counts=False):
    smiles_list = list(df_smiles['smiles'])
    
    table_name = 'radius%i' % radius

    with sqlite3.connect(out_path) as conn:
        cur = conn.cursor()

        cur.execute("DROP TABLE IF EXISTS %s" % table_name)
        if counts and radius != 0:
            cur.execute("CREATE TABLE %s("
                        "env TEXT NOT NULL, "
                        "core_smi TEXT NOT NULL, "
                        "core_num_atoms INTEGER NOT NULL, "
                        "core_sma TEXT NOT NULL, "
                        "dist2 INTEGER NOT NULL, "
                        "freq INTEGER NOT NULL)" % table_name)
        elif radius !=0:
            cur.execute("CREATE TABLE %s("
                        "env TEXT NOT NULL, "
                        "core_smi TEXT NOT NULL, "
                        "core_num_atoms INTEGER NOT NULL, "
                        "core_sma TEXT NOT NULL,"
                        "dist2 INTEGER NOT NULL)" % table_name)
        else:
            cur.execute("CREATE TABLE %s("
                       "env TEXT NOT NULL, "
                       "core_smi TEXT NOT NULL, "
                        "core_num_atoms INTEGER NOT NULL)" % table_name)
        conn.commit()

        buf = []
        for i, smiles in enumerate(tqdm(smiles_list)):
            '''if i == 159:
                frag = None'''
            mol_env_core = fragment_mol(smiles, radius)
            if mol_env_core:
                for env_core in mol_env_core:
                    env, core, num_heavy_atoms = env_core
                    for c in core:
                        buf.append((env, c, num_heavy_atoms))
                if (i + 1) % 100 == 0:
                    if radius != 0:
                        adata = _get_additional_data((items[:2] for items in buf))
                        buf = [a + b for a, b in zip(buf, adata)]
                        cur.executemany("INSERT INTO %s VALUES (?, ?, ?, ?, ?)" % table_name, buf)
                        conn.commit()
                        buf = []
                    else:
                        cur.executemany("INSERT INTO %s VALUES (?, ?, ?)" % table_name, buf)
                        conn.commit()
                        buf = []

        if buf:
            if radius != 0:
                adata = _get_additional_data((items[:2] for items in buf))
                buf = [a + b for a, b in zip(buf, adata)]
                cur.executemany("INSERT INTO %s VALUES (?, ?, ?, ?, ?)" % table_name, buf)
            else:
                cur.executemany("INSERT INTO %s VALUES (?, ?, ?)" % table_name, buf)
        conn.commit()

        idx_name = "%s_env_idx" % table_name
        cur.execute("DROP INDEX IF EXISTS %s" % idx_name)
        cur.execute("CREATE INDEX %s ON %s (env)" % (idx_name, table_name))
        conn.commit()


def _get_additional_data(data):
    res = [__calc(*items) for items in data]
    return res

def __calc(env, core):
    sma = combine_core_env_to_rxn_smarts(core, env, False)
    if core.count('*') == 2:
        mol = Chem.MolFromSmiles(core, sanitize=False)
        mat = Chem.GetDistanceMatrix(mol)
        ids = []
        for a in mol.GetAtoms():
            if not a.GetAtomicNum():
                ids.append(a.GetIdx())
        dist2 = mat[ids[0], ids[1]]
    else:
        dist2 = 0
    return sma, dist2
        

def fragment_mol(smiles, radius, keep_stereo=False):
    mol = Chem.MolFromSmiles(smiles)
    detect_functional_group(mol)
    split_bonds = split_mol(mol)
    [atom.SetAtomMapNum(i + 1) for i, atom in enumerate(mol.GetAtoms())]
    
    bonds = []
    for bond in split_bonds:
        x, y = bond.GetBeginAtom().GetIdx(), bond.GetEndAtom().GetIdx()
        bonds.append(mol.GetBondBetweenAtoms(x, y).GetIdx())
    # mol = Chem.MolFromSmiles(smiles)
    frags_org = Chem.MolToSmiles(Chem.FragmentOnBonds(mol, bonds)).split('.')
    frags_org.sort(key = lambda frag: (len(frag), str.casefold), reverse=True)
    all_core_env = []
        
    # split for each frag
    for i in range(len(frags_org)):
        frags = frags_org.copy()
        frags = [Chem.MolFromSmiles(frag) for frag in frags]
        core = frags[i]

        # obtain atom_ids of core 
        core_ids = []
        for atom in core.GetAtoms():
            atom_id = atom.GetAtomMapNum() - 1
            if atom_id != -1:
                core_ids.append(atom_id)
        # frags_test = frags.copy()
        # frags_smiles = standard_smiles(frags_test)

        # assert frags_smiles[i] == core

        core_ids = sorted(set(core_ids))
        # split mol to core and other
        bonds_list = []
        bonds_type = []
        for index in core_ids:
            atom = mol.GetAtomWithIdx(index)
            for nei in atom.GetNeighbors():
                if nei.GetAtomMapNum() - 1 not in core_ids:
                    bonds_list.append(mol.GetBondBetweenAtoms(atom.GetIdx(), nei.GetIdx()).GetIdx())
                else:
                    continue

        # get env of mol
        core_env = Chem.MolToSmiles(Chem.FragmentOnBonds(mol, bonds_list)).split('.')
        env, core = standardize_attach(core_env, core_ids)
        att_num = core.count('*')
        env_smi = standard_env(env, radius, att_num)

        core = Chem.MolFromSmiles(core)
        num_heavy_atoms = core.GetNumHeavyAtoms()
        for a in core.GetAtoms():
            if a.GetSymbol() != '*':
                a.SetAtomMapNum(0)
        # __standardize_att_by_env(env, core, keep_stereo) 
                
    
        if att_num == 1 or radius == 0:
            core = _standardize_smiles_with_att_points(core, keep_stereo)

            all_core_env.append((env_smi, (core, ), num_heavy_atoms))
                
        else:
            res = []

            p = _get_att_permutations(Chem.MolFromSmiles(env_smi))

            if len(p) > 1:
                for d in p:
                    c = _permute_att(core, d)
                    res.append(c)
            else:
                res.append(core)

            d = tuple(set(_standardize_smiles_with_att_points(m, keep_stereo) for m in res))

            all_core_env.append((env_smi, d, num_heavy_atoms))
            # core_smi = _standardize_smiles_with_att_points(core, keep_stereo)
            '''core_env.sort(key = lambda frag: len(frag), reverse=True)
            if len(core_env) == 2:
                env = [frag for frag in core_env if re.sub(r'\*+\:+\d', '*', frag) != re.sub(r'\*+\:+\d', '*', core)]
            else:
                frag_att_num = [len(re.findall(r'\*', frag)) for frag in core_env]
                core_env.pop(frag_att_num.index(max(frag_att_num)))
                env = core_env'''
            # env = standard_env(env, radius)
    return all_core_env


def standardize_attach(frags_smiles, core_ids):

    frags = [Chem.MolFromSmiles(frag) for frag in frags_smiles]
    num_frags = len(frags)
    all_number = []
    number_map = defaultdict(list)

    for i, smiles in enumerate(frags_smiles):
        attach_num = find_numbers(smiles)
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

    # detect core of mol
    for frag in frags:
        for atom in Chem.MolFromSmiles(frag).GetAtoms():
            if atom.GetAtomMapNum() - 1 in core_ids and atom.GetSymbol() != '*':
                core = frag
                frags.pop(frags.index(frag))
                break
    
    return frags, core

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
    
    '''for frag in frags:
        print(Chem.MolToSmiles(frag))'''

    return frags


def get_key(dict, value):
    return set([k for k, v in dict.items() if value in v])

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


def _permute_att(mol, d):
    new_mol = Chem.Mol(mol)
    for a in new_mol.GetAtoms():
        i = a.GetAtomMapNum()
        if i in d:
            a.SetAtomMapNum(d[i])
    return new_mol


def _get_att_permutations(env):
    """
    Return possible permutations of attachment point map numbers as a tuple of dicts,
    where each dict: key - old number, value - new number
    """
    maps, ranks = __get_maps_and_ranks(env)

    d = defaultdict(list)
    for rank, att in zip(ranks, maps): # define to determine which frag will attach to bond
        d[rank].append(att)

    c = []
    for v in d.values():
        c.append([dict(zip(v, x)) for x in permutations(v, len(v))])

    return tuple(__merge_dicts(*item) for item in product(*c))


def __merge_dicts(*dicts):
    res = dicts[0].copy()
    for item in dicts[1:]:
        res.update(item)
    return res


def _standardize_smiles_with_att_points(mol, keep_stereo=False):
    """
    to avoid different order of atoms in SMILES with different map number of attachment points

    smi = ["ClC1=C([*:1])C(=S)C([*:2])=C([*:3])N1",
           "ClC1=C([*:1])C(=S)C([*:3])=C([*:2])N1",
           "ClC1=C([*:2])C(=S)C([*:1])=C([*:3])N1",
           "ClC1=C([*:2])C(=S)C([*:3])=C([*:1])N1",
           "ClC1=C([*:3])C(=S)C([*:1])=C([*:2])N1",
           "ClC1=C([*:3])C(=S)C([*:2])=C([*:1])N1"]

    these will produce different output with RDKit MolToSmiles():
        S=c1c([*:1])c(Cl)[nH]c([*:3])c1[*:2]
        S=c1c([*:1])c(Cl)[nH]c([*:2])c1[*:3]
        S=c1c([*:1])c([*:3])[nH]c(Cl)c1[*:2]
        S=c1c([*:2])c(Cl)[nH]c([*:1])c1[*:3]
        S=c1c([*:1])c([*:2])[nH]c(Cl)c1[*:3]
        S=c1c([*:2])c([*:1])[nH]c(Cl)c1[*:3]

    output of this function
        S=c1c([*:2])c([*:3])[nH]c(Br)c1[*:1]
        S=c1c([*:3])c([*:2])[nH]c(Br)c1[*:1]
        S=c1c([*:1])c([*:3])[nH]c(Br)c1[*:2]
        S=c1c([*:3])c([*:1])[nH]c(Br)c1[*:2]
        S=c1c([*:1])c([*:2])[nH]c(Br)c1[*:3]
        S=c1c([*:2])c([*:1])[nH]c(Br)c1[*:3]

    https://sourceforge.net/p/rdkit/mailman/message/35862258/
    """

    # update property cache if needed
    if mol.NeedsUpdatePropertyCache():
        mol.UpdatePropertyCache()

    # store original maps and remove map numbers from mol
    backup_atom_map = "backupAtomMap"
    for a in mol.GetAtoms():
        atom_map = a.GetAtomMapNum()
        if atom_map:
            a.SetIntProp(backup_atom_map, atom_map)
            a.SetAtomMapNum(0)

    # get canonical ranks for atoms for a mol without maps
    atoms = list(zip(list(Chem.CanonicalRankAtoms(mol)), [a.GetIdx() for a in mol.GetAtoms()]))
    atoms.sort()

    # set new atom maps based on canonical order
    rep = {}
    atom_map = 1
    for pos, atom_idx in atoms:
        a = mol.GetAtomWithIdx(atom_idx)
        if a.HasProp(backup_atom_map):
            a.SetAtomMapNum(atom_map)
            rep["[*:%i]" % atom_map] = "[*:%i]" % a.GetIntProp(backup_atom_map)
            atom_map += 1

    # get SMILES and relabel with original map numbers
    s = Chem.MolToSmiles(mol, isomericSmiles=keep_stereo)
    rep = dict((re.escape(k), v) for k, v in rep.items())
    patt = re.compile("|".join(rep.keys()))
    s = patt.sub(lambda m: rep[re.escape(m.group(0))], s)

    return s


if __name__ == '__main__':
    main()

