import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace
from torch_facl.facl_utils import smiles2triplet, mol2graph, index_select_ND
from torch_geometric.utils import to_dense_batch
import torch_geometric.nn as pygnn
import math
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
from data.Molgraph import BatchMolGraph
from typing import Union, List, Tuple, DefaultDict
import wandb
import sys
from fcal_utils import Prompt_generator



class FACL(nn.Module):
    def __init__(self, args: Namespace, emb_model, out_dir, pretrain=None, struct_attention=False):
        super().__init__()
        self.out_dir = out_dir
        self.args = args
        self.classification = args.classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = args.multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)
        self.pretrain = pretrain

        self.emb_model = emb_model
        self.triplet_input = False
        self.graph_input = False
        self.struct_attention = struct_attention
        self.encoder1 = GPSModel(args.gps)
        self.encoder2 = GPSModel(args.gps)
        self.criterion1 = ContrastiveLoss(loss_computer='nce_softmax',
                                          temperature=args.loss.temperature, args=args.loss).cuda()
        self.alpha = float(args.alpha)
        self.beta = float(args.beta)
        self.multiloss = MultiLossLayer(num_loss=3)

    def forward(self, data, augmentation=False, db_path=None, embedding=False):
        """
        :param batch: A batch of smiles
        """

        anchor_fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), radius=2, nBits=512) for s in data[0]]

        mol_dist = 1 - torch.Tensor(
            [DataStructs.BulkTanimotoSimilarity(anchor_fps[i], anchor_fps) for i in range(len(anchor_fps))]
        )
        graph_reps, aug_rep = self.get_representation(data, self.emb_model, self.args)
        reps_dict = {}

        assert len(graph_reps) == 3
        sim_dist = data[-2]
        neg_dist = data[-1]
        for i, reps in enumerate(graph_reps):
            reps_dict[i] = reps

        output_ori = DefaultDict(list)
        for key in reps_dict.keys():
            output_ori[key] = self.encoder1.forward(reps_dict[key])
        output_aug = self.encoder2.forward(aug_rep)
        
        if embedding:
            return output_ori, output_aug
        '''if not self.graph_input:
            batch = mol2graph(data, self.emb_model) 
        if not self.triplet_input:
            triplet_batch = smiles2triplet(data, self.emb_model, db_path=db_path)'''
        
        '''smi_dist = compute_dist(triplet_batch['s'], triplet_batch['n'])
        output = self.encoder.forward(triplet_batch)'''

        # loss = self.compute_loss_test(output_ori, output_aug, mol_dist)
        loss = self.compute_loss(output_ori, output_aug, sim_dist, neg_dist, mol_dist)
        
        return loss

    def finetune(self, smiles, labels, emb_model, args, prompt=False):

        # self.prompt = Prompt_generator(args)
        graph_rep, aug_rep = self.get_representation(smiles, emb_model, self.args) 

        output_ori = self.encoder1.forward(graph_rep)
        output_aug = self.encoder2.forward(aug_rep)
        com_rep = torch.cat((output_aug, output_ori), dim=-1)
        preds = self.predictor(com_rep)
        return preds

    def get_encoder(self):
        encoder = self.encoder1
        encoder.cuda()
        return encoder
    
    def create_predictor(self, args, output_size):

        linear_dims = 300
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        if args.use_input_features:
            linear_dims += args.feature_dims

        if args.num_layers == 1:
            predictor = [
                nn.BatchNorm1d(linear_dims * 2),
                nn.Linear(linear_dims * 2, linear_dims),
                nn.BatchNorm1d(linear_dims),
                activation,
                dropout,
                nn.Linear(linear_dims, output_size)
            ]
        else:
            predictor = [
                nn.BatchNorm1d(linear_dims * 2),
                activation,
                dropout,
                nn.Linear(linear_dims * 2, args.hidden_dims)
            ]
            for _ in range(args.num_layers - 2):
                predictor.extend([
                    activation,
                    dropout,
                    nn.Linear(args.hidden_dims, args.hidden_dims)
                ])
            predictor.extend([
                activation,
                nn.LayerNorm(args.hidden_dims),
                dropout,
                nn.Linear(args.hidden_dims, output_size)
            ])
        self.predictor = nn.Sequential(*predictor)
        self.predictor.cuda()

    def compute_loss_old(self, output):
        rep_keys = output.keys()
        for rep in rep_keys:
            assert rep in ['s', 'n', 'a', 'p']
        loss = self.triplet_loss(output)
        return loss
    
    def compute_trip_loss(self, ori_reps, mol_dist, neg_dist=None):
        device = ori_reps[0][0].device
        batch_size = ori_reps[0].shape[0]
        if neg_dist is not None:
            neg_dist = neg_dist.float()
            neg_dist = (1 - neg_dist).repeat_interleave(batch_size, dim=-1).view(batch_size, batch_size)
            mol_dist = mol_dist * neg_dist
            anchor, mol_neg, mol_sim = ori_reps[0], ori_reps[1], ori_reps[2]
            neg_sim = self.distance(anchor, mol_neg)
            neg_sim = neg_sim.repeat_interleave(batch_size - 1, dim=0).view(batch_size, batch_size - 1)
            sim_sim = self.distance(anchor, mol_sim)
            sim_sim = sim_sim.repeat_interleave(batch_size - 1, dim=0).view(batch_size, batch_size - 1)
            diag_idx = torch.eye(batch_size, dtype=torch.bool)
            mol_dist = mol_dist[~diag_idx].view(batch_size, batch_size - 1)
            mol_dist_edge = torch.ones((batch_size, batch_size - 1), device=device)
            mol_dist_edge[mol_dist == 0] = 0

            loss = torch.maximum(torch.tensor(0).to(device),
                             mol_dist_edge * (mol_dist + sim_sim - neg_sim)).mean()

        else:
            anchor, mol_neg = ori_reps[0], ori_reps[1]
            neg_sim = self.distance(anchor, mol_neg)
            neg_sim = neg_sim.repeat_interleave(batch_size - 1, dim=0).view(batch_size, batch_size - 1)

            interleave = anchor.repeat_interleave(batch_size, dim=0).view(batch_size, batch_size, -1)
            repeat = anchor.repeat(batch_size, 1).view(batch_size, batch_size, -1)
            neg_matrix = self.distance(interleave, repeat)
            diag_idx = torch.eye(batch_size, dtype=torch.bool)
            neg_matrix = neg_matrix[~diag_idx].view(batch_size, batch_size - 1)
            mol_dist = mol_dist[~diag_idx].view(batch_size, batch_size - 1)
            mol_dist_edge = torch.ones((batch_size, batch_size - 1), device=device)
            mol_dist_edge[mol_dist == 0] = 0

            loss = torch.maximum(torch.tensor(0).to(device),
                                mol_dist_edge * (mol_dist + neg_sim - neg_matrix)).mean()
        
        return loss


    def compute_trip_loss_827(self, aug_rep, ori_reps, sim_diat, neg_dist, mol_dist):
        device = ori_reps[0][0].device
        mol_dist = mol_dist.to(device)
        batch_size = aug_rep.shape[0]
        
        mol_anchor, mol_neg, mol_sim = ori_reps[0], ori_reps[1], ori_reps[2]

        sim_sim = self.distance(mol_anchor, mol_sim)
        sim_sim = sim_sim.repeat_interleave(batch_size - 1, dim=0).view(batch_size, batch_size - 1)
        neg_sim = self.distance(mol_anchor, mol_neg)
        neg_sim = neg_sim.repeat_interleave(batch_size - 1, dim=0).view(batch_size, batch_size - 1)
        diag_idx = torch.eye(batch_size, dtype=torch.bool)
        mol_dist = mol_dist[~diag_idx].view(batch_size, batch_size - 1)
        mol_dist_edge = torch.ones((batch_size, batch_size - 1), device=device)
        mol_dist_edge[mol_dist == 0] = 0

        loss = torch.maximum(torch.tensor(0).to(device),
                            mol_dist_edge * (mol_dist + sim_sim - neg_sim)).mean()

        return loss
    

    def compute_loss(self, ori_reps, aug_rep, sim_dist, neg_dist, mol_dist):
        tem = 0.1
        device = ori_reps[0][0].device
        # rep_keys = ori_reps.keys()
        sim_dist = sim_dist.to(device)
        neg_dist = neg_dist.to(device)
        mol_dist = mol_dist.to(device)
        # loss1 = self.criterion1(aug_rep, ori_reps[0])
        # pos_matrix = self.distance(ori_reps[0], aug_rep, metric='l2norm')
        # neg_matrix = self.neg_matrix(ori_reps, sim_dist, mol_dist)
        loss1 = self.compute_trip_loss(ori_reps, mol_dist)
        # loss2 = self.compute_simcl_loss(aug_rep, ori_reps, sim_dist)
        loss2 = self.distance(ori_reps[0], aug_rep, metric='l2norm').mean()
        # loss2 = self.criterion1(aug_rep, ori_reps[0])
        loss3 = self.compute_trip_loss(ori_reps, mol_dist, neg_dist)
        # loss3 = self.compute_sim_loss(ori_reps, neg_dist, sim_dist)
        # loss3 = self.compute_trip_loss(aug_rep, ori_reps[0], neg_matrix, mol_dist)
        # loss3 = 1 - loss1 / (tri_loss * neg_dist) 
        print('loss1 is {}, loss2 is {}, loss3 is {}'.format(float(loss1), float(loss2), float(loss3)))
        '''wandb.log({'loss1/step': loss1.item()})
        wandb.log({'loss2/step': loss2.item()})
        wandb.log({'loss3/strp': loss3.item()})'''
        # weight_loss = self.multiloss(torch.tensor([loss1, loss2, loss3], device=device))
        # factor = get_weighted_loss(torch.tensor([loss1, loss2], device=device))
        # loss = self.alpha * loss1 + loss2
        # factor = get_weighted_loss(torch.tensor([loss1, loss2], device=device))
        
        # loss = loss1 + 0.05 * loss2 + 0.25 * loss3

        loss = loss1 + 0.25 * loss3
        return loss
        # return torch.maximum(torch.tensor(0).to(device), loss1 - loss2)
    

    def compute_sim_loss(self, ori_reps, neg_dist, sim_dist):

        eps = 1e-10
        sim_matrix = self.distance(ori_reps[0], ori_reps[2])
        neg_matrix = self.distance(ori_reps[0], ori_reps[1])

        coeff_in = (neg_dist / sim_dist).float()
        coeff = torch.where(coeff_in > 10, torch.log(coeff_in), torch.log(coeff_in) + 1)
        coeff = torch.where(coeff > 10, torch.ones_like(coeff) * 10, coeff)

        matrix_coeff = neg_matrix / (sim_matrix + eps)
        mask = torch.where(matrix_coeff > 100000, torch.zeros_like(matrix_coeff), torch.ones_like(matrix_coeff))
        mask = torch.where(matrix_coeff == 0, torch.zeros_like(matrix_coeff), mask)
        matrix_coeff = torch.where(matrix_coeff > 100000, 0, matrix_coeff)
        loss = self.normalized_rmse_loss(matrix_coeff, coeff, mask)
        '''ratio_1 = torch.log(matrix_coeff / coeff).mean()
        ratio_2 = torch.log(coeff / matrix_coeff).mean()'''

        return loss.float()

    def normalized_rmse_loss(self, pred, target, mask):
        assert pred.shape == target.shape
        eps = 1e-8
        pred_norm = (pred - pred.mean()) / (pred.std() + eps)
        target_norm = (target - target.mean()) / (target.std() + eps)

        loss = self.mse_loss(pred_norm, target_norm, mask)
        return torch.sqrt(loss)
    
    def mse_loss(self, pred, target, mask):
        squared_diff = (pred - target) ** 2
        return (squared_diff * mask).sum() / mask.sum()
    
    def compute_simcl_loss(self, aug_sample, ori_reps, sim_dist):
        device = aug_sample.device
        sim_matrix = self.distance(ori_reps[0], ori_reps[2], metric='cossim')
        pos_matrix = self.distance(ori_reps[0], aug_sample, metric='cossim')
        return torch.maximum(torch.tensor(0).to(device), 
                             (sim_dist + pos_matrix - sim_matrix)).mean()

    '''def compute_trip_loss(self, aug_sample, anchor, neg_matrix, mol_dist):
        pos_matrix = self.distance(anchor, aug_sample)
        batch_size = len(pos_matrix)
        device = pos_matrix.device
        pos_matrix = pos_matrix.repeat_interleave(batch_size - 1, dim=0).view(batch_size, batch_size - 1)
        neg_matrix = neg_matrix.repeat_interleave(batch_size - 1, dim=0).view(batch_size, batch_size - 1)

        diag_idx = torch.eye(batch_size, dtype=torch.bool)
        mol_dist = mol_dist[~diag_idx].view(batch_size, batch_size - 1)
        mol_edge = torch.ones((batch_size, batch_size - 1), device=mol_dist.device)
        mol_edge[mol_dist == 0] = 0

        return torch.maximum(torch.tensor(0).to(device), 
                             mol_edge * (mol_dist + pos_matrix - neg_matrix)).mean()'''


    def compute_trip_loss_old(self, aug_sample, output, sim_dist, neg_dist, mol_dist):
        mol_anchor, mol_neg, mol_sim = output[0], output[1], output[2]
        device = mol_anchor.device
        batch_size = mol_anchor.shape[0]

        diag_idx = torch.eye(batch_size, dtype=torch.bool)
        margin_factor = mol_dist[~diag_idx].view(batch_size, batch_size - 1)
        sim_dist = sim_dist.repeat_interleave(batch_size - 1).view(batch_size, batch_size - 1).to(device)
        neg_dist = neg_dist.repeat_interleave(batch_size - 1).view(batch_size, batch_size - 1).to(device)
        sim_matrix = self.distance(mol_anchor, mol_sim, metric='l2norm')
        sim_matrix = sim_matrix.repeat_interleave(batch_size - 1).view(batch_size, batch_size - 1)
        '''pos_matrix = self.distance(mol_anchor, aug_sample, metric='l2norm')
        pos_matrix = pos_matrix.repeat_interleave(batch_size - 1).view(batch_size, batch_size - 1)'''
        neg_matrix = self.distance(mol_anchor, mol_neg, metric='l2norm')
        neg_matrix = neg_matrix.repeat_interleave(batch_size - 1).view(batch_size, batch_size - 1)
        # pos_matrix = pos_matrix.repeat_interleave(batch_size - 1, dim=0).view(batch_size, batch_size - 1)
        # neg_matrix = neg_matrix.repeat_interleave(batch_size - 1, dim=0).view(batch_size, batch_size - 1)

        '''diag_idx = torch.eye(batch_size, dtype=torch.bool)
        mol_dist = mol_dist[~diag_idx].view(batch_size, batch_size - 1)
        mol_edge = torch.ones((batch_size, batch_size - 1), device=mol_dist.device)
        mol_edge[mol_dist == 0] = 0'''

        return torch.maximum(torch.tensor(0).to(device), 
                             (margin_factor + sim_matrix - neg_matrix)).mean()
    
    def distance(self, tensor_a, tensor_b, metric='l2norm'):
        if metric == 'cossim':
            return 1 - F.cosine_similarity(tensor_a, tensor_b, dim=-1)
        elif metric == 'l2norm':
            return (tensor_a - tensor_b).norm(dim=-1)

    def neg_matrix(self, output, sim_dist, mol_dist):
        
        mol_anchor, mol_neg, mol_sim = output[0], output[1], output[2]
        device = mol_anchor.device
        sim_dist = sim_dist.to(device)
        batch_size = len(mol_anchor)

        sim_matrix = self.distance(mol_anchor, mol_sim, metric='l2norm') # (B, dim)
        # smi_matrix = self.distance(output['a'], output['s']) # (B, dim)
        neg_matrix = self.distance(mol_anchor, mol_neg, metric='l2norm') # (B, dim)
        '''if mol_dist is not None:
            # create margin_factor
            diag_idx = torch.eye(batch_size, dtype=bool)
            margin_factor = mol_dist[~diag_idx].view(batch_size, -1)
            margin_factor_edge = torch.ones((batch_size, -1), device=margin_factor.device)
            margin_factor_edge[margin_factor == 0] = 0

            margin_loss = torch.maximum(torch.tensor(0).to(device),
                                    margin_factor_edge * (margin_factor + sim_matrix * sim_dist + neg_matrix)).mean()
        else:
            margin_loss = torch.maximum(torch.tensor(0).to(device),
                                    sim_matrix * sim_dist + neg_matrix).mean()'''
        
        return (sim_matrix * sim_dist + neg_matrix) / 2
    
    '''def neg_matrix(self, output, sim_dist, mol_dist):
        mol_anchor, mol_neg, mol_sim = output[0], output[1], output[2]
        device = mol_anchor.device
        sim_dist = sim_dist.to(device)

        sim_matrix = self.criterion1(mol_anchor, mol_sim)
        neg_matrix = self.criterion1(mol_anchor, mol_neg)

        return (sim_matrix * sim_dist + neg_matrix) / 2'''
    
    def get_representation(self, batch, emb_model, args):
        # graph_reps = [anchor_batch, pos_batch, neg_batch, sim_batch]
        if isinstance(batch, list):
            graph_reps = []

            for i, smiles_list in enumerate(batch):
                if isinstance(smiles_list, tuple):
                    sample_rep = mol2graph(smiles_list, augmentation=False, args=args)
                    graph_reps.append(sample_rep)
                if not i:
                    frag_emb, frag_size = self.emb_model.frag_augment_batch(smiles_list)
                    sample_rep = mol2graph(smiles_list, augmentation=True, frag_emb=frag_emb, frag_batch=frag_size, args=args)
                    aug_rep = sample_rep
        else:
            graph_reps = mol2graph(batch, augmentation=False, args=args)
            frag_emb, frag_size = emb_model.frag_augment_batch(batch)
            aug_rep = mol2graph(batch, augmentation=True, frag_emb=frag_emb, frag_batch=frag_size, args=args)
        return graph_reps, aug_rep


class ContrastiveLoss(nn.Module):
    def __init__(self, loss_computer: str, temperature: float, args) -> None:
        super().__init__()
        self.device = args.device

        if loss_computer == 'nce_softmax':
            self.loss_computer = NCESoftmaxLoss(self.device)
        else:
            raise NotImplementedError(f"Loss Computer {loss_computer} not Support!")
        self.temperature = temperature

    def forward(self, z_i, z_j):
        # SimCSE
        batch_size = z_i.size(0)

        emb = F.normalize(torch.cat([z_i, z_j]))

        similarity = torch.matmul(emb, emb.t()) - torch.eye(batch_size * 2).to(self.device) * 1e12
        # similarity = similarity * 20
        loss = self.loss_computer(similarity)

        return loss
    

class NCESoftmaxLoss(nn.Module):
    def __init__(self, device) -> None:
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, similarity):
        batch_size = similarity.size(0) // 2
        label = torch.tensor([(batch_size + i) % (batch_size * 2) for i in range(batch_size * 2)]).to(
            self.device).long()
        loss = self.criterion(similarity, label)
        return loss


# New Attention Layer for GPS Model
class StructAttenLayer(nn.Module):
    def __init__(self, embed_dim, num_heads=8, bias=False):
        super(StructAttenLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.bias = bias

        # frag_features extractor
        self.frag_extractor = FragExtractor(embed_dim, num_layers=3)

    def forward(self, node_feats, edge_index, frag_index=None):
        
        frag_features = self.frag_extractor(node_feats, edge_index, frag_index)
        
        frag_features = frag_features.reshape(-1)
        

class FragExtractor(nn.Module):
    def __init__(self, embed_dim, num_layers):
        super(FragExtractor, self).__init__()


class GPSModel(nn.Module):
    def __init__(self, args: Namespace):
        super().__init__()

        self.atom_dims = args.input_atom_dims
        self.bond_dims = args.input_bond_dims
        self.hidden_dims = args.hidden_dims

        self.num_heads = args.num_heads
        self.layer_norm = args.layer_norm
        self.batch_norm = args.batch_norm
        self.activation = get_activation_function(args.activation)
        self.attn_dropout = args.attn_dropout
        self.num_classes = None
        self.predictor = None
        self.projection = nn.Linear(args.hidden_dims, args.project_dims)
        

        '''self.attn = torch.nn.MultiheadAttention(
            self.hidden_dims, num_heads=self.num_heads, dropout=self.attn_dropout, batch_first=True
        )'''
        self.attn = None

        if args.mpnn == 'CMPNN':
            if not args.atom_messages:
                self.bond_dims = args.input_bond_dims + self.atom_dims
            self.mpnn = CMPN(args, self.atom_dims, self.bond_dims)
        elif args.mpnn == 'MPNN':
            self.mpnn = MPN()
        elif args.mpnn == 'GIN':
            gin_nn = nn.Sequential(
                pygnn.Linear(self.hidden_dims, self.hidden_dims),
                self.activation(),
                pygnn.Linear(self.hidden_dims, self.hidden_dims)
            )
            self.mpnn = pygnn.GINConv(gin_nn)



        self.gt_dropout = nn.Dropout(args.dropout)
        self.mp_dropout = nn.Dropout(args.dropout)
        self.ff_linear1 = nn.Linear(self.hidden_dims, self.hidden_dims * 2)
        self.ff_linear2 = nn.Linear(self.hidden_dims * 2, self.hidden_dims)
        self.act_fn_ff = get_activation_function(args.activation)
        self.ff_dropout = nn.Dropout(args.dropout)

        if self.layer_norm:
            self.mp1_norm = nn.LayerNorm(self.hidden_dims)
            self.gt1_norm = nn.LayerNorm(self.hidden_dims)
            self.norm2 = nn.LayerNorm(self.hidden_dims)
        if self.batch_norm:
            self.mp1_norm = nn.BatchNorm1d(self.hidden_dims)
            self.gt1_norm = nn.BatchNorm1d(self.hidden_dims)
            self.norm2 = nn.BatchNorm1d(self.hidden_dims)
        self.log_attn_weights = args.log_attn_weights

    def create_predictor(self, args):
        """
        Creates the predictor layer for predicting task.
        :param args: Arguments object.
        :return: predictor
        """
        '''if args.dataset_type == 'muticlass':
            self.num_classes = args.muticlass_num_classes'''
        linear_dims = self.hidden_dims
        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)
        if args.use_input_features:
            linear_dims += args.feature_dims
        # Create predictor
        if args.num_layers == 1:
            predictor = [
                nn.LayerNorm(linear_dims),
                dropout,
                nn.Linear(linear_dims, args.output_size)
            ]
        else:
            predictor = [
                nn.LayerNorm(linear_dims),
                dropout,
                nn.Linear(linear_dims, args.hidden_dims)
            ]
            for _ in range(args.num_layers - 2):
                predictor.extend([
                    activation,
                    dropout,
                    nn.Linear(args.hidden_dims, args.hidden_dims * 2),
                    dropout,
                    nn.Linear(args.hidden_dims * 2, args.hidden_dims)
                ])
            predictor.extend([
                activation,
                nn.LayerNorm(args.hidden_dims),
                dropout,
                nn.Linear(args.hidden_dims, args.output_size),
            ])
        
        self.predictor = nn.Sequential(*predictor)
        self.predictor.cuda()

    def forward(self, data, pretrain=True):
        
        f_res = data.f_atoms
        f_res = f_res.cuda() # (2842, 130)
        f_out_list = []
        if self.mpnn is not None:
            mp_out = self.mpnn(data) # Nan
            out_sim1 = self.pairwise_cosine_similarity(mp_out)
            mp_out = f_res + mp_out

            if self.layer_norm:
                mp_out = self.mp1_norm(mp_out) # add data.batch
            if self.batch_norm:
                mp_out = self.mp1_norm(mp_out)
            out_sim1 = self.pairwise_cosine_similarity(mp_out)
            f_out_list.append(mp_out)

            # muiti-head attention, not including edge_attr
            if self.attn is not None:
                f_atoms_dense, mask = to_dense_batch(data.f_atoms, data.batch) # padding to make dims of each mol equal
                f_atoms_attn = self._sa_block(f_atoms_dense, None, ~mask)[mask]

                f_atoms_attn = self.gt_dropout(f_atoms_attn)
                attn_out = f_res + f_atoms_attn

                if self.layer_norm:
                    attn_out = self.gt1_norm(attn_out)
                if self.batch_norm:
                    attn_out = self.gt1_norm(attn_out)
                f_out_list.append(attn_out)
                attn_sim1 = self.pairwise_cosine_similarity(attn_out)
                
            # Combine mpnn and attn outputs.
            f_out = sum(f_out_list)
            fout_sim1 = self.pairwise_cosine_similarity(f_out)

            # Feed forward block
            f_out = f_out + self._ff_block(f_out)
            if self.layer_norm:
                f_out = self.norm2(f_out)
            if self.batch_norm:
                f_out = self.norm2(f_out)
            fout_sim2 = self.pairwise_cosine_similarity(f_out)
                
            # Read out for each molecule
            mol_out = []
            for i, (a_start, a_size) in enumerate(data.a_scope):
                if a_size == 0:
                    assert 0
                cur_hidden = f_out.narrow(0, a_start, a_size)
                mol_out.append(cur_hidden.mean(0))
            mol_out = torch.stack(mol_out, dim=0)
            similarity = self.pairwise_cosine_similarity(mol_out)
            sim_mean = similarity.mean()
            if pretrain:
                return self.projection(mol_out)
            else:
                return self.predictor(mol_out) # self.predictor(mol_out)'''

    def pairwise_cosine_similarity(self, embedding):
        embedding = F.normalize(embedding, p=2, dim=1)
        cosine_similarity = torch.mm(embedding, embedding.t())
        return cosine_similarity
    
    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        # x = torch.tensor(x, dtype=torch.float16)
        x = x.cuda()
        key_padding_mask = key_padding_mask.cuda()
        if not self.log_attn_weights:
            x = self.attn(x, x, x, attn_mask=attn_mask,
                          key_padding_mask=key_padding_mask,
                          need_weights=False)[0]
        else:
            x, A = self.attn(x, x, x,
                             attn_mask=attn_mask,
                             key_padding_mask=key_padding_mask,
                             need_weights=True,
                             average_attn_weights=False
                             )
            self.attn_weights = A.detach().cpu()
        return x
    
    def _ff_block(self, f_input):
        """Feed-forward block"""
        f_out = self.ff_dropout(self.act_fn_ff(self.ff_linear1(f_input)))
        return self.ff_dropout(self.ff_linear2(f_out))
    
class CMPN(nn.Module):
    def __init__(self, args: Namespace, atom_dims: int, bond_dims: int):
        super(CMPN, self).__init__()
        self.atom_dims = atom_dims
        self.bond_dims = bond_dims
        self.hidden_dims = args.hidden_dims
        self.bias = args.mp_bias
        self.depth = args.mp_depth
        self.dropout = args.dropout
        self.layers_per_message = 1
        self.undirected = args.undirected
        self.atom_messages = args.atom_messages
        self.features_only = args.features_only
        self.use_input_features = args.use_input_features
        self.args = args

        # Dropout
        self.dropout = nn.Dropout(p=self.dropout)

        self.act_func = get_activation_function(args.activation)

        # Input
        self.W_i_atom = nn.Linear(self.atom_dims, self.hidden_dims, bias=self.bias)
        self.W_i_bond = nn.Linear(self.bond_dims, self.hidden_dims, bias=self.bias)

        w_h_input_size_atom = self.hidden_dims + self.bond_dims
        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_dims, bias=self.bias)

        w_h_input_size_bond = self.hidden_dims
        for depth in range(self.depth - 1):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_dims, bias=self.bias)

        self.W_o = nn.Linear(self.hidden_dims * 2, self.hidden_dims)
        self.gru = BatchGRU(self.hidden_dims)
        self.lr = nn.Linear(self.hidden_dims * 3, self.hidden_dims, bias=self.bias)

    def forward(self, batch: Union[List[str], BatchMolGraph]):

        # Input
        f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = batch.get_components()
        if self.args.cuda or next(self.parameters()).is_cuda:
            f_atoms, f_bonds, a2b, b2a, b2revb = (
                    f_atoms.cuda(), f_bonds.cuda(),
                    a2b.cuda(), b2a.cuda(), b2revb.cuda())
        input_atom = self.W_i_atom(f_atoms)
        input_atom = self.act_func(input_atom)
        message_atom = input_atom.clone()

        input_bond = self.W_i_bond(f_bonds)
        input_bond = self.act_func(input_bond)
        message_bond = input_bond.clone()

        # Message passing
        for depth in range(self.depth - 1):
            agg_message = index_select_ND(message_bond, a2b)
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
            message_atom = message_atom + agg_message


            # directed graph
            rev_message = message_bond[b2revb]  # num_bonds x hidden
            message_bond = message_atom[b2a] - rev_message  # num_bonds x hidden

            message_bond = self._modules[f'W_h_{depth}'](message_bond)
            message_bond = self.dropout(self.act_func(input_bond + message_bond))
        
        agg_message = index_select_ND(message_bond, a2b)
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]
        agg_message = self.lr(torch.cat([agg_message, message_atom, input_atom], 1))
        agg_message = self.gru(agg_message, a_scope)

        atom_hiddens = self.act_func(self.W_o(agg_message))  # num_atoms x hidden
        atom_hiddens = self.dropout(atom_hiddens)  # num_atoms x hidden [n_atoms, 130]


        return atom_hiddens



class MPN():
    def __init__(self, d):
        self.d = d


class BatchGRU(nn.Module):
    def __init__(self, hidden_size: int):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                          bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size),
                                1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))

            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len - cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)

        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2 * self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                             cur_message_unpadding], 0)
        return message


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
    
def compute_dist(input1, input2):
    dist = torch.Tensor([
        DataStructs.BulkAllBitSimilarity(input1, input2)
    ])
    return dist


class MultiLossLayer(nn.Module):
    """
        implementation of "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
        """
    
    def __init__(self, num_loss):
        super(MultiLossLayer, self).__init__()
        self.sigmas_dota = nn.Parameter(nn.init.uniform(torch.empty(num_loss), a=0.1, b=0.2), requires_grad=True)

    def forward(self, loss_list):
        factor = torch.div(1.0, torch.mul(2.0, self.sigmas_dota))
        loss_part = torch.sum(torch.mul(factor, loss_list))
        regular_part = torch.sum(torch.log(self.sigmas_dota))
        print('factor: {}, {}, {}'.format(factor[0], factor[1], factor[2]))
        loss = loss_part + regular_part
    
        return loss
    
def get_weighted_loss(loss_list):
    factor = list(torch.div(loss_list, torch.sum(loss_list)))
    for i in range(len(loss_list)):
        if loss_list[i] < 1e-3:
            factor[i] = 0
    return factor