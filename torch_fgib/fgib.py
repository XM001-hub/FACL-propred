from torch import nn
from argparse import Namespace
from torch_geometric.nn import NNConv, global_mean_pool
import torch
import torch.nn.functional as F
from torch import Tensor
# from model.gpscl_model import get_activation_function
from torch_scatter import scatter_mean, scatter_add, scatter_std
from sklearn.metrics import mean_absolute_error
from typing import List
import time
from rdkit import Chem
from torch_fgib.fgib_utils import detect_functional_group, extract_vocab
from Data.fgib_data import split_mol, split_to_frags
from Data.featurize import get_atom_features, get_top_indices, get_bond_features
import numpy as np



# from torch_geometric.nn import gnn


class FGIBModel(nn.Module):
    def __init__(self, args: Namespace):
        super(FGIBModel, self).__init__()

        self.MP_layer = MPModel(args.node_input_dim, args.node_hidden_dim,
                                args.edge_input_dim, args.edge_hidden_dim,
                                args.MP_steps, args.MP_dropout)

        self.compressor = nn.Sequential(
            nn.Linear(args.node_hidden_dim, args.node_hidden_dim),
            nn.BatchNorm1d(args.node_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.node_hidden_dim, 1),
            nn.Sigmoid()
        )

        '''self.predictor_LR = nn.Sequential(
            nn.Linear(args.node_hidden_dim, 256),
            nn.LeakyReLU(0.01),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128)
        )'''
        self.predictor = nn.Sequential(
            nn.Linear(args.node_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.fingerprint = nn.Linear(128, args.fingerprint_bits)
        self.regression = nn.Linear(128, args.regression)
        self.beta = args.beta
        self.frag = args.frag
        self.hidden_dims = 128
        self.fingerprint_bits = args.fingerprint_bits
        # self.frag_extractor = FragExtractor()

        # reconstruct fingerprint loss when training
        self.fp_loss_fn = nn.BCEWithLogitsLoss()
        self.re_loss_fn = DESLossFunc(beta=0.5, reduction='mean')

        # finetune
        self.f_predictor = None
        self.label_cl_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.label_re_loss = nn.MSELoss(reduction='none')

        self.init_model()

    def init_model(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def create_predictor(self, output_size, num_layers=3, dropout=0, activation='LeakyReLU'):
        # args: including dropout, output_size(num_tasks), activation("ReLU"), num_layers
        dropout = nn.Dropout(dropout)
        activation = get_activation_function(activation)
        if num_layers == 1:
            '''f_predictor = [
                activation,
                nn.LayerNorm(self.hidden_dims),
                dropout,
                nn.Linear(self.hidden_dims, output_size),
                # nn.BatchNorm1d(output_size)
            ]'''
            f_predictor = [
                nn.BatchNorm1d(self.hidden_dims),
                # nn.LayerNorm(self.hidden_dims),
                nn.Linear(self.hidden_dims, self.hidden_dims),
                nn.BatchNorm1d(self.hidden_dims),
                # nn.LayerNorm(self.hidden_dims),
                nn.LeakyReLU(0.02),
                nn.Linear(self.hidden_dims, output_size)]
        else:
            f_predictor = [
                dropout,
                nn.Linear(self.hidden_dims, self.hidden_dims)
            ]
            for i in range(num_layers - 1):
                f_predictor.extend([
                    activation,
                    dropout,
                    nn.Linear(self.hidden_dims, self.hidden_dims)])
            f_predictor.extend([
                activation,
                # nn.LayerNorm(self.hidden_dims),
                nn.Linear(self.hidden_dims, output_size)
                # nn.BatchNorm1d(output_size)
            ])
        self.f_predictor = nn.Sequential(*f_predictor)
        
        # self.f_predictor = f_predictor
        self.label_cl_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.label_re_loss = nn.MSELoss(reduction='none')
        self.f_predictor.cuda()

    def forward(self, batch, get_w=False, augmentation=False, device=None, finetune=False, reg_norm=None, type='classification'):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        start_time = time.time()
        # batch.to(device)
        if finetune:
            batch.to(device)

        lambda_pos, frag_features = self.get_lambda_pos(batch)
        w = lambda_pos

        if get_w:
            return w.cpu()
        lambda_pos = lambda_pos.reshape(-1, 1)
        lambda_neg = 1 - lambda_pos
        preserve_rate = (w > 0.5).float().mean()
        preserve_std = lambda_pos.std()

        static_feature = frag_features.clone()
        frag_feature_mean = scatter_mean(static_feature, batch.frag2graph_batch, dim=0)[batch.frag2graph_batch]
        frag_feature_std = scatter_std(static_feature, batch.frag2graph_batch, dim=0)[batch.frag2graph_batch]

        noisy_frag_feature_mean = lambda_pos * frag_features + lambda_neg * frag_feature_mean
        noisy_frag_feature_std = lambda_neg * frag_feature_std

        noisy_frag_feature = noisy_frag_feature_mean + torch.randn_like(
            noisy_frag_feature_mean) * noisy_frag_feature_std
        
        # return frag_emb when augmentation is True for mol_graph
        if augmentation:
            return noisy_frag_feature, batch.frag_sizes, w.cpu()
        noisy_subgraph = global_mean_pool(noisy_frag_feature, batch.frag2graph_batch)
        graph_emb = self.predictor(noisy_subgraph)

        if not finetune:
            fp_pred = self.fingerprint(graph_emb)
            re_pred = self.regression(graph_emb)
            y_pred = torch.cat((fp_pred, re_pred), dim=1)
            target = torch.tensor(batch.y, dtype=torch.float32, device=device)

            regression_loss = self.re_loss_fn(target, y_pred)
            # fp_loss = self.fp_loss(data.y, y_pred)
            re_mae = self.regression_metrics(target, y_pred)

            epsilon = 1e-7
            KL_tensor = 0.5 * scatter_add(((noisy_frag_feature_std ** 2) / (frag_feature_std + epsilon) ** 2).mean(dim=1),
                                        batch.frag2graph_batch).reshape(-1, 1) + \
                        scatter_add((((noisy_frag_feature_mean - frag_feature_mean) / (frag_feature_std + epsilon)) ** 2),
                                    batch.frag2graph_batch, dim=0)
            KL_loss = KL_tensor.mean()
            end_time = time.time()
            training_time = end_time - start_time
            print('Training time is %.2f s' % training_time)

            return regression_loss, KL_loss, preserve_rate, preserve_std, re_mae
        else:

            predicted_f = self.f_predictor(graph_emb)
            if reg_norm:
                test_output = predicted_f * reg_norm[1] + reg_norm[0]
            else:
                sigmoid = nn.Sigmoid()
                test_output = sigmoid(predicted_f)
            # loss = self.regression_metrics(predicted_f, torch.tensor(batch.y, device=torch.device('cuda:0'), dtype=torch.float32).view(-1, 1))
            labels = batch.y
            label_var = torch.var(labels, dim=0, unbiased=False).mean()
            pred_var = torch.var(test_output, dim=0, unbiased=False).mean()
            var_penalty = torch.relu(0.1 - pred_var)
            loss = self.re_loss_fn(test_output, labels) + var_penalty
            # loss = mean_absolute_error(test_output.cpu().detach().numpy(), labels.cpu().detach().numpy())
            '''if type == 'classification':
                loss = self.label_cl_loss(predicted_f, torch.tensor(batch.y, device=torch.device('cuda:0'), dtype=torch.float32).view(-1, 1))
            else:
                loss = self.label_re_loss(predicted_f, batch.y)'''
            epsilon = 1e-7
            KL_tensor = 0.5 * scatter_add(((noisy_frag_feature_std ** 2) / (frag_feature_std + epsilon) ** 2).mean(dim=1),
                                        batch.frag2graph_batch).reshape(-1, 1) + \
                        scatter_add((((noisy_frag_feature_mean - frag_feature_mean) / (frag_feature_std + epsilon)) ** 2),
                                    batch.frag2graph_batch, dim=0)
            KL_loss = KL_tensor.mean()

            return loss, KL_loss
    
    def frag_augment_batch(self, batch, get_w=False, augmentation=True):
        batch = smiles2batch(batch)
        frag_emb, frag_size, _ = self.forward(batch, get_w, augmentation)
        return frag_emb, frag_size
    

    def get_lambda_pos(self, batch):
        node_features = F.normalize(self.MP_layer(batch.x, batch.edge_attr, batch.edge_index))
        if self.frag == 'pool':
            frag_features = global_mean_pool(node_features, batch.node2frag_batch)
            # print(data.node2frag_batch)
        else:
            frag_features = self.frag_extractor(node_features, batch.node2frag_batch,
                                                batch.edge_attr, batch.edge2frag_batch)
        lambda_pos = self.compressor(frag_features).squeeze()
        return lambda_pos, frag_features


    def finetune(self, data):
        node_features = F.normalize(self.MP_layer(data.x, data.edge_attr, data.edge_index), dim=1)
        if self.frag == 'pool':
            frag_features = global_mean_pool(node_features, data.node2frag_batch)
            # print(data.node2frag_batch)
        else:
            frag_features = self.frag_MP_layer(node_features, data.node2frag_batch,
                                               data.edge_attr,
                                               data.edge2frag_batch)  # edge2frag_batch should be created in ZINC_dataset
        lambda_pos = self.compressor(frag_features).squeeze()
        lambda_pos = lambda_pos.reshape(-1, 1)
        print('[Lambda_pos std]: {:.4f}'.format(lambda_pos.std()))

        lambda_neg = 1 - lambda_pos
        # preserve_rate = (w > 0.5).float().mean()

        static_feature = frag_features.clone().detach()
        frag_feature_mean = scatter_mean(static_feature, data.frag2graph_batch, dim=0)[data.frag2graph_batch]
        frag_feature_std = scatter_std(static_feature, data.frag2graph_batch, dim=0)[data.frag2graph_batch]

        noisy_frag_feature_mean = lambda_pos * frag_features + lambda_neg * frag_feature_mean
        noisy_frag_feature_std = lambda_neg * frag_feature_std

        noisy_frag_feature = noisy_frag_feature_mean + torch.randn_like(
            noisy_frag_feature_mean) * noisy_frag_feature_std

        noisy_subgraph = global_mean_pool(noisy_frag_feature, data.frag2graph_batch)
        graph_emb = self.predictor(noisy_subgraph)

        pred_label = self.f_predictor(graph_emb).reshape(-1)
        true_label = torch.tensor(data.y, device=self.gpu, dtype=torch.float32)  # should be replaced by scaled_label

        return pred_label, true_label

    def regression_metrics(self, y_true, y_pred: torch.Tensor):
        fp_bits = self.fingerprint_bits
        y_pred = y_pred.cpu().detach().numpy()
        return mean_absolute_error(y_true[:, fp_bits:].cpu(), y_pred[:, fp_bits:])

class ResidualBlock(nn.Module):
    def __init__(self, input_dims, hidden_dims=128):
        super().__init__()
        self.linear1 = nn.Linear(input_dims, hidden_dims)



# MP + Multi-head attention ( similiar to GPS) for extracting fragment features
class FragExtractor(nn.Module):
    def __init__(self, embed_dim, num_layers=3, batch_norm=True):
        super(FragExtractor, self).__init__()
        self.num_layers = num_layers
        self.relu = nn.ReLU()

        if batch_norm:
            self.bn = nn.BatchNorm1d(embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        layers = []
        '''for _ in range(num_layers):
            layers.append(get_gnn(frag='gin'))'''

    def forward(self, node_features, node2frag_batch, edge_attr, edge2frag_batch):
        return node_features + edge2frag_batch


class MPModel(nn.Module):
    def __init__(self, node_input_dim, node_hidden_dim, edge_input_dim,
                 edge_hidden_dim, num_steps, dropout=0.0):
        super(MPModel, self).__init__()
        self.num_steps = num_steps
        self.mp_input = nn.Linear(node_input_dim, node_hidden_dim)
        self.message_layer = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim)
        )

        self.conv = NNConv(in_channels=node_hidden_dim,
                           out_channels=node_hidden_dim,
                           nn=edge_network,
                           aggr='add',
                           root_weight=True)
        self.dropout = dropout

    def forward(self, node_features, edge_attr, edge_index):
        init_features = node_features.clone()
        activation = nn.ReLU()
        out = activation(self.mp_input(node_features))
        for i in range(self.num_steps):
            if len(edge_attr) != 0:
                m = torch.relu(self.conv(out, edge_index, edge_attr))
            else:
                m = torch.relu(self.conv.bias + out)
            out = self.message_layer(torch.cat([m, out], dim=1))
        return out + init_features


class DESLossFunc(nn.Module):
    def __init__(self, beta, reduction: str):
        super().__init__()
        self.beta = torch.tensor(beta, dtype=torch.float)
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        diff = torch.abs(input - target)
        cond = diff <= self.beta
        re_loss = torch.where(cond, 0.5 * diff ** 2, self.beta * diff - 0.5 * self.beta ** 2)
        if self.reduction == 'mean':
            loss = torch.mean(re_loss)
        elif self.reduction == 'sum':
            loss = torch.sum(re_loss)
        else:
            raise ValueError('reduction must be "mean" or "sum"')
        return loss
    

'''def get_gnn(gnn_type, dim):
    if gnn_type == 'graph':
        return gnn.GraphConv(dim, dim)
    elif gnn_type == 'gcn':
        return gnn.GCNConv(dim, dim)
    elif gnn_type == 'gin':
        mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(True),
            nn.Linear(dim, dim),
        )
        return gnn.GINConv(mlp, train_eps=True)'''


def get_activation_function(activation):
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == 'GELU':
        return nn.GELU()
    else:
        raise ValueError(f'Activation "{activation}" not supported.')


def smiles2batch(batch: List[str]):
    mol_datas = []
    for smiles in batch:
        mol_data = MolData(Chem.MolFromSmiles(smiles))
        mol_datas.append(mol_data)
    return BatchMolData(mol_datas)


class MolData():
    """
        A MolData class represents the graph structure and featuyization of a single molecule for fgib.

        computing the following attributes:
        - x: A mapping from an atom index to a list atom features.
        - edge_index: edge_index of mol graph (shape: [2, n_edges]).
        - edge_attr: edge attribute of mol graph (shape: [n_edges, 10]

        """
    def __init__(self, mol : Chem.Mol, augmentation=False):
        n_atoms = mol.GetNumAtoms()

        stereo = Chem.FindMolChiralCenters(mol)
        chiral_centers = [0] * mol.GetNumAtoms()

        for i in stereo:
            chiral_centers[i[0]] = i[1]

        node_features = []
        edge_features = []
        bonds = []

        # if not augmentation:
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

        self.atom_features = node_features
        self.edge_index = torch.tensor(bonds, dtype=torch.long).T
        self.edge_features = edge_features

        detect_functional_group(mol)
        split_bonds = split_mol(mol)
        self.frags, self.node2frag_batch, self.frag2graph_batch, self.edge2frag_batch, self.edge2frag_size = split_to_frags(mol, split_bonds)

        '''self.frags, self.node2frag_batch, self.frag2graph_batch, self.edge2frag_batch, self.edge2frag_size = (
                get_frag_batch_brics(mol))'''
        self.smiles = Chem.MolToSmiles(mol)
        self.n_atoms = n_atoms
        '''else:

            for i in range(mol.GetNumAtoms()):
                atom_i = mol.GetAtomWithIdx(i)
                atom_i_features = get_atom_features(atom_i, chiral_centers[i])
                node_features.append(atom_i_features)

            detect_functional_group(mol)'''



class BatchMolData():
    """
    A BatchMolData class represents the graph structure and featurization of a batch of molecules.
    """
    def __init__(self, mol_datas: list):
        self.smiles_batch = [mol_data.smiles for mol_data in mol_datas]
        self.n_mols = len(self.smiles_batch)
        self.atom_fdim = 130
        self.bond_fdim = 10

        # Start n_atoms, n_bonds and n_frags at 0
        self.n_atoms = 0
        self.n_bonds = 0
        self.n_frags = 0
        self.n_mol = 0
        frag_sizes, edge_attr = [], []
        edge_index = torch.tensor([[], []], dtype=torch.int64)
        frag2graph_batch = torch.tensor([], dtype=torch.int64)
        x = []
        node2frag_batch = []
        frag_smiles = []

        # create data_batch for batch
        for mol_data in mol_datas:
            # some components about atoms
            x.extend(mol_data.atom_features)
            edge_attr.extend(mol_data.edge_features)
            mol_edge_index = mol_data.edge_index + self.n_atoms
            edge_index = torch.cat((edge_index, mol_edge_index), dim=1)
            self.n_atoms += mol_data.n_atoms
            frag_smiles.extend(mol_data.frags)

            # some components about frags
            node2frag_batch.extend((mol_data.node2frag_batch + self.n_frags).tolist())
            self.n_frags += len(mol_data.frags)
            frag_sizes.append(len(mol_data.frags))
            frag2graph_batch = torch.cat((frag2graph_batch, mol_data.frag2graph_batch + self.n_mol), dim=0)
            self.n_mol += 1

        # Convert attributes of BatchMolData to cuda
        self.edge_index, self.frag2graph_batch = edge_index.cuda(), frag2graph_batch.cuda()
        self.x = torch.tensor(x, dtype=torch.float, device='cuda:0')
        self.edge_attr = torch.tensor(edge_attr, dtype=torch.float, device='cuda:0')
        self.node2frag_batch = torch.tensor(node2frag_batch, dtype=torch.int64, device='cuda:0')
        self.frag_sizes = torch.tensor(frag_sizes, dtype=torch.int64, device='cuda:0')
        self.frag_attr = torch.zeros([sum(self.frag_sizes)])
        self.frag_smiles = frag_smiles

    def get_aug_frags(self, frag_weights):
        # get tuple of frag_weights for each mol in batch
        frag_weights = torch.split(frag_weights, list(self.frag_sizes))
        frag_smiles_index = torch.split(torch.tensor(range(len(self.frag_smiles))), list(self.frag_sizes))

        # get index of key and component of each mol
        max_inds, min_inds = [], []
        max_weis, min_weis = [], []
        for mol_frags in frag_weights:
            max_wei, max_ind = get_top_indices(mol_frags)
            min_wei, min_ind = get_top_indices(mol_frags, top=False)
            max_inds.append(max_ind)
            min_inds.append(min_ind)
            max_weis.append(max_wei)
            min_weis.append(min_wei)

        # get smiles of frag based on pos/neg_inds
        neg_frags, sim_frags = [], []
        for i, mol_smiles in enumerate(frag_smiles_index):
            neg_frag, sim_frag = [], []
            for ind in max_inds[i]:
                neg_frag.append(self.frag_smiles[mol_smiles[ind]])
            for ind in min_inds[i]:
                sim_frag.append(self.frag_smiles[mol_smiles[ind]])
            '''pos_frag = self.frag_smiles[mol_smiles[pos_inds[i]]]
            neg_frag = self.frag_smiles[mol_smiles[neg_inds[i]]]'''
            neg_frags.append(tuple(neg_frag))
            sim_frags.append(tuple(sim_frag))

        return max_inds, min_inds, neg_frags, sim_frags, max_weis, min_weis, self.smiles_batch
    
    def to_cuda(self):
        for attr_name in vars(self):
            attr_value = getattr(self, attr_name)
            if isinstance(attr_value, torch.Tensor):
                setattr(self, attr_name, attr_value.cuda())
        return self


