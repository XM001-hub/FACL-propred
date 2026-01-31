import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import BondType, AllChem
from torch_fgib.gen_neg import gen_neg
from data.Molgraph import MolGraph, BatchMolGraph
from typing import DefaultDict, List, Union
from torch_fgib.fgib import MolData, BatchMolData
import math
from sklearn.metrics import auc, mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,\
    roc_auc_score, accuracy_score, log_loss
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from typing import Any
import torch.nn as nn
import torch.nn.functional as F


def attention(query, key, value, mask, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn



class AttentionLayer(nn.Module):
    def __init__(self, args):
        super(AttentionLayer, self).__init__()
        self.hidden_size = args.hidden_size
        self.w_q = nn.Linear(133, 32)
        self.w_k = nn.Linear(133, 32)
        self.w_v = nn.Linear(133, 32)
        
        self.dense = nn.Linear(32, 133)
        self.LayerNorm = nn.LayerNorm(133, eps=1e-6)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self,fg_hiddens, init_hiddens):
        query = self.w_q(fg_hiddens)
        key = self.w_k(fg_hiddens)
        value = self.w_v(fg_hiddens)

        padding_mask = (init_hiddens != 0) + 0.0
        mask = torch.matmul(padding_mask, padding_mask.transpose(-2, -1))
        x, attn = attention(query, key, value, mask)

        hidden_states = self.dense(x)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + fg_hiddens)
        
        return hidden_states


class Prompt_generator(nn.Module):
    def __init__(self, args):
        super(Prompt_generator, self).__init__()
        self.hidden_size = args.hidden_size
        self.alpha = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.alpha.data.fill_(0.1)
        self.cls = nn.Parameter(torch.randn(1,133), requires_grad=True)
        self.linear = nn.Linear(133, self.hidden_size)
        self.attention_layer_1 = AttentionLayer(args)
        self.attention_layer_2 = AttentionLayer(args)
        self.norm = nn.LayerNorm(args.hidden_size)
        
    def forward(self, atom_hiddens: torch.Tensor, fg_states: torch.Tensor, atom_num, fg_indexs):
        for i in range(len(fg_indexs)):
            fg_states.scatter_(0, fg_indexs[i:i+1], self.cls)
        
        hidden_states = self.attention_layer_1(fg_states, fg_states)
        hidden_states = self.attention_layer_2(hidden_states, fg_states)
        fg_out = torch.zeros(1, self.hidden_size).cuda()
        cls_hiddens = torch.gather(hidden_states, 0, fg_indexs)
        cls_hiddens = self.linear(cls_hiddens)
        fg_hiddens = torch.repeat_interleave(cls_hiddens, torch.tensor(atom_num).cuda(), dim=0)
        fg_out = torch.cat((fg_out, fg_hiddens), 0)

        fg_out = self.norm(fg_out)
        return atom_hiddens + self.alpha * fg_out




def smiles2triplet(data, emb_model, augmentation=False, convert_neg=False, frag_batch=None, db_path=None):
    """
        data: data of smiles;
        emb_model: FGIB_model to obtion frag weight for property predicting;
        db_path: database path of frags;

        output(batch_triplet): a dict contains 's'(smi_sample), 'p'(pos_sample), 'a'(anchor) and 'n'(neg_sample)
    """
    # get batch of MolData from smiles_batch
    mol_datas = []
    for smiles in data:
        mol_data = MolData(Chem.MolFromSmiles(smiles))
        mol_datas.append(mol_data)
    batch = BatchMolData(mol_datas)

    # obtain frah_emb using frag_emb model trained
    frag_emb, frag_size, frag_weight = emb_model(batch, augmentation=True)
    pos_index, neg_index, pos_frags, neg_frags, smiles_list = batch.get_aug_frags(frag_weight)
    # neg_sample, smi_sample = batch.get_aug_mol(pos_index, neg_index)
    mol_triplet = DefaultDict(list)
    n_frags = 0
    for i, smiles in enumerate(smiles_list):
        mol_triplet['a'].append(Chem.MolFromSmiles(smiles))
        n_frags = frag_size[i]
        # product = {Chem.MolToSmiles(Chem.MolFromSmiles(smiles))}

        protected_index = set(range(n_frags))

        # obtain negative sample of anchor
        replace_id = pos_index[i]
        replace_frag = pos_frags[i]
        protected_neg = protected_index.remove(int(replace_id))
        neg_sample = gen_neg(smiles, protected_ids=protected_neg, replace_frag=replace_frag, replace_id=replace_id, db_path=db_path)
        neg_sample = MolGraph(neg_sample, augmentation=False)
        mol_triplet['n'].append(neg_sample)

        # obtain same sample of anchor
        replace_id_smi = neg_index[i]
        replace_frag_smi = neg_frags[i]
        protected_smi = protected_index.remove(int(replace_id_smi))
        smi_sample = gen_neg(smiles, protected_ids=protected_smi, replace_frag=replace_frag_smi)
        smi_sample = MolGraph(smi_sample,augmentation=False)
        mol_triplet['s'].append(smi_sample)

        # get augmented sample
        pos_sample = MolGraph(Chem.MolFromSmiles(smiles), frag_emb=frag_emb, augmentation=True)
        mol_triplet['p'].append(pos_sample)

    # batch mol graph
    batch_triplet = {key: BatchMolGraph(mol_triplet[key]) for key in mol_triplet.keys()}
    

    return batch_triplet

def rmse(targets: List[float], preds: List[float]) -> float:
    """
    Computes the root mean squared error.

    :param targets: A list of targets.
    :param preds: A list of predictions.
    :return: The computed rmse.
    """
    return math.sqrt(mean_squared_error(targets, preds))


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


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def get_bond_features(bond):
    bond_features = []
    bond_features += one_of_k_encoding_unk(bond.GetBondType(),
                                           [BondType.SINGLE, BondType.DOUBLE, BondType.TRIPLE, BondType.AROMATIC])
    bond_features += [bond.GetIsAromatic()]
    bond_features += [bond.GetIsConjugated()]
    bond_features += one_of_k_encoding_unk(str(bond.GetStereo()), ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])

    return bond_features


def mol2graph(smiles_batch, augmentation: bool, frag_emb=None, frag_batch=None, args=None):
    
    mol_datas = []
    n_frags = 0
    for i, smiles in enumerate(smiles_batch):
        if augmentation:
            frag_size = frag_batch.tolist()[i]
            mol_data = MolGraph(Chem.MolFromSmiles(smiles), frag_emb[n_frags:(n_frags + frag_size), :].tolist(), augmentation)
            n_frags += frag_size
            mol_datas.append(mol_data)
        else:
            mol_data = MolGraph(Chem.MolFromSmiles(smiles), None, augmentation)
            mol_datas.append(mol_data)
    return BatchMolGraph(mol_datas, args)



def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target[index == 0] = 0
    return target


class NoamLR(_LRScheduler):
    """
    Noam learning rate scheduler with piecewise linear increase and exponential decay.

    The learning rate increases linearly from init_lr to max_lr over the course of
    the first warmup_steps (where warmup_steps = warmup_epochs * steps_per_epoch).
    Then the learning rate decreases exponentially from max_lr to final_lr over the
    course of the remaining total_steps - warmup_steps (where total_steps =
    total_epochs * steps_per_epoch). This is roughly based on the learning rate
    schedule from Attention is All You Need, section 5.3 (https://arxiv.org/abs/1706.03762).
    """
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_epochs: List[Union[float, int]],
                 total_epochs: List[int],
                 steps_per_epoch: int,
                 init_lr: List[float],
                 max_lr: List[float],
                 final_lr: List[float]):
        """
        Initializes the learning rate scheduler.

        :param optimizer: A PyTorch optimizer.
        :param warmup_epochs: The number of epochs during which to linearly increase the learning rate.
        :param total_epochs: The total number of epochs.
        :param steps_per_epoch: The number of steps (batches) per epoch.
        :param init_lr: The initial learning rate.
        :param max_lr: The maximum learning rate (achieved after warmup_epochs).
        :param final_lr: The final learning rate (achieved after total_epochs).
        """
        assert len(optimizer.param_groups) == len(warmup_epochs) == len(total_epochs) == len(init_lr) == \
               len(max_lr) == len(final_lr)

        self.num_lrs = len(optimizer.param_groups)

        self.optimizer = optimizer
        self.warmup_epochs = np.array(warmup_epochs).reshape(-1)
        self.total_epochs = np.array(total_epochs).reshape(-1)
        self.steps_per_epoch = steps_per_epoch
        self.init_lr = np.array(init_lr).reshape(-1)
        self.max_lr = np.array(max_lr).reshape(-1)
        self.final_lr = np.array(final_lr).reshape(-1)

        self.current_step = 0
        self.lr = init_lr
        self.warmup_steps = (self.warmup_epochs * self.steps_per_epoch).astype(int)
        self.total_steps = self.total_epochs * self.steps_per_epoch
        self.linear_increment = (self.max_lr - self.init_lr) / self.warmup_steps

        self.exponential_gamma = (self.final_lr / self.max_lr) ** (1 / (self.total_steps - self.warmup_steps))

        super(NoamLR, self).__init__(optimizer)

    def get_lr(self) -> List[float]:
        """Gets a list of the current learning rates."""
        return list(self.lr)

    def step(self, current_step: int = None):
        """
        Updates the learning rate by taking a step.

        :param current_step: Optionally specify what step to set the learning rate to.
        If None, current_step = self.current_step + 1.
        """
        if current_step is not None:
            self.current_step = current_step
        else:
            self.current_step += 1

        for i in range(self.num_lrs):
            if self.current_step <= self.warmup_steps[i]:
                self.lr[i] = self.init_lr[i] + self.current_step * self.linear_increment[i]
            elif self.current_step <= self.total_steps[i]:
                self.lr[i] = self.max_lr[i] * (self.exponential_gamma[i] ** (self.current_step - self.warmup_steps[i]))
            else:  # theoretically this case should never be reached since training should stop at total_steps
                self.lr[i] = self.final_lr[i]

            self.optimizer.param_groups[i]['lr'] = self.lr[i]


class StandardScaler:
    """A StandardScaler normalizes a dataset.

    When fit on a dataset, the StandardScaler learns the mean and standard deviation across the 0th axis.
    When transforming a dataset, the StandardScaler subtracts the means and divides by the standard deviations.
    """

    def __init__(self, means: np.ndarray = None, stds: np.ndarray = None, replace_nan_token: Any = None):
        """
        Initialize StandardScaler, optionally with means and standard deviations precomputed.

        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: The token to use in place of nans.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X: List[List[float]]) -> 'StandardScaler':
        """
        Learns means and standard deviations across the 0th axis.

        :param X: A list of lists of floats.
        :return: The fitted StandardScaler.
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means), np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds), np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(self.stds.shape), self.stds)

        return self

    def transform(self, X: List[List[float]]):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.

        :param X: A list of lists of floats.
        :return: The transformed data.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X: List[List[float]]):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.

        :param X: A list of lists of floats.
        :return: The inverse transformed data.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none
        
        
        

