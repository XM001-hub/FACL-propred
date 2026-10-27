from rdkit import Chem
import re
from collections import defaultdict


patt_remove_h = re.compile("(?<!\[)H[1-9]*(?=:[0-9])") 

def combine_core_env_to_rxn_smarts(core, env, keep_h=True):

    if isinstance(env, str):
        # env = env[2:]
        m_env = Chem.MolFromSmiles(env, sanitize=False)
    if isinstance(core, str):
        m_frag = Chem.MolFromSmiles(core, sanitize=False)

    backup_atom_map = "backupAtomMap"

    # put all atom maps to atom property and remove them
    for a in m_env.GetAtoms():
        atom_map = a.GetAtomMapNum()
        if atom_map:
            a.SetIntProp(backup_atom_map, atom_map)
            a.SetAtomMapNum(0)

    for a in m_frag.GetAtoms():
        atom_map = a.GetAtomMapNum()
        if atom_map:
            a.SetIntProp(backup_atom_map, atom_map)
            a.SetAtomMapNum(0)

    # set canonical ranks for atoms in env without maps
    m_env.UpdatePropertyCache()
    list_test = list(Chem.CanonicalRankAtoms(m_env))
    for atom_id, rank in zip([a.GetIdx() for a in m_env.GetAtoms()], list(Chem.CanonicalRankAtoms(m_env))):
        a = m_env.GetAtomWithIdx(atom_id)
        if not a.HasProp(backup_atom_map):
            a.SetAtomMapNum(rank + 1)  # because ranks start from 0

    m = Chem.RWMol(Chem.CombineMols(m_frag, m_env))
    m_smiles = Chem.MolToSmiles(m)

    links = defaultdict(list)  # pairs of atom ids to create bonds
    att_to_remove = []  # ids of att points to remove
    for a in m.GetAtoms():
        if a.HasProp(backup_atom_map):
            i = a.GetIntProp(backup_atom_map)
            links[i].append(a.GetNeighbors()[0].GetIdx())
            att_to_remove.append(a.GetIdx())

    for i, j in links.values():
        m.AddBond(i, j, Chem.BondType.SINGLE)

    for i in sorted(att_to_remove, reverse=True):
        m.RemoveAtom(i)

    comb_sma = mol_to_smarts(m, keep_h)
    if not keep_h:  # remove H only in mapped env part
        comb_sma = patt_remove_h.sub('', comb_sma)
    return comb_sma


def mol_to_smarts(mol, keep_h=True):
    # keep_h - will increase the count of H atoms for atoms with attached hydrogens to create a valid smarts
    # e.g. [H]-[CH2]-[*] -> [H]-[CH3]-[*]

    mol = Chem.Mol(mol)
    mol.UpdatePropertyCache()

    # change the isotope to 42
    for atom in mol.GetAtoms():
        if keep_h:
            s = sum(na.GetAtomicNum() == 1 for na in atom.GetNeighbors())
            if s:
                atom.SetNumExplicitHs(atom.GetTotalNumHs() + s)
        atom.SetIsotope(42)

    # print out the smiles - all the atom attributes will be fully specified
    smarts = Chem.MolToSmiles(mol, isomericSmiles=True, allBondsExplicit=True)
    # remove the 42 isotope labels
    smarts = re.sub(r'\[42', "[", smarts)

    return smarts