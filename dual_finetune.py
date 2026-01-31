import torch
import argparse
import pandas as pd
import os
import torch.nn.functional as F
import math
import yaml
import warnings
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from data.fgib_data import DatasetFrag
from torch.utils.data import random_split, Subset, Sampler
from torch_geometric.data import DataLoader
from pretrain import get_args
from torch_fgib.fgib import FGIBModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch.optim import Adam, AdamW
from torch_facl.facl_model import FACL
import random
import pickle
import webcolors as bcolors
from torch_facl.facl_utils import rmse, NoamLR, mol2graph
from rdkit import RDLogger
from argparse import Namespace
from logger_utils import initialize_exp
from splitter import scaffold_class_split, scaffold_split
from tdc.single_pred import ADME
from collections import defaultdict



def init_frag_emb_model(args, model_args, reg_norm=None, model_path=None, finetune=True, emb_args=None):
    # init emb_model and finetune model using labeled data
    if finetune:
        if model_path:
            emb_model = torch.load(model_path)
        else:
            emb_model = torch.load(model_args.model_path)
    else:
        emb_model = FGIBModel(emb_args.model)
    return emb_model


def print_grad(model):
    v_n, v_v, v_g = [], [], []
    for name, parameter in model.named_parameters():
        v_n.append(name)
        v_v.append(parameter.detach().cpu().numpy() if parameter is not None else [0])
        v_g.append(parameter.grad.detach().cpu().numpy() if parameter.grad is not None else [0])
    for i in range(len(v_n)):
        print('value %s: %.3e ~ %.3e' % (v_n[i], np.min(v_v[i]).item(), np.max(v_v[i]).item()))
        print('grad  %s: %.3e ~ %.3e' % (v_n[i], np.min(v_g[i]).item(), np.max(v_g[i]).item()))


def finetune_emb(args, emb_model, model_args, dataset, emb_args):

    if args.regression_norm == 'zscore' and args.dataset_type == 'regression':
        reg_norm = [dataset.mean_l, dataset.std_l]
    else:
        reg_norm = None
    
    # split
    if args.split_type == 'random':
        val_size = int(len(dataset) * 0.1)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                                    generator=torch.Generator().manual_seed(args.seed))
    
    elif args.split_type == 'scaffold':
        train_idx, val_idx = scaffold_split(dataset.smiles, frac_valid=0.2, test=False)
        train_size = len(train_idx)
        train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)
    
    elif args.split_type == 'pre_split':
        train_idx, val_idx = dataset.split['train'], dataset.split['test']
        train_size = len(train_idx)
        train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)

    else:
        train_idx, val_idx = scaffold_class_split(dataset.smiles, dataset.labels, frac_valid=0.1, frac_test=0, balanced=False, tested=True, only_val=True)
        train_size = len(train_idx)
        train_dataset, val_dataset = Subset(dataset, train_idx), Subset(dataset, val_idx)
    args.output_size = dataset.task_num

    # determine metric type
    if args.metric is None:
        if args.dataset_type == 'classification' and args.output_size == 1:
            args.metric = 'auc'
        elif args.dataset_type == 'regression':
            args.metric = 'rmse'
        else:
            args.metric = 'cross_entropy'
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # emb_model = init_frag_emb_model(args, model_args, reg_norm=reg_norm, emb_args=emb_args)

    # finetune emb_model
    emb_model.create_predictor(args.output_size, num_layers=1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emb_model.to(device)
    parameters = emb_model.parameters()
    total_trainable_params = sum(p.numel() for p in emb_model.parameters())

    train_name, pretrain_name, freeze_name, mintor_name = [], [], [], []
    '''for name, param in emb_model.named_parameters():
        train_name.append(name)'''
    for name, param in emb_model.named_parameters():
        if 'f_predictor' in name or 'compressor' in name:
            train_name.append(name)
        else:
            # param.requires_grad = False
            pretrain_name.append(name) 
        
        if 'f_predictor' or 'predictor' in name:
                mintor_name.append(name)
        '''elif 'compressor' in name:
            freeze_name.append(name)'''
    
    train_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in train_name, emb_model.named_parameters()))))
    pretrain_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in pretrain_name, emb_model.named_parameters()))))
    mintor_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in mintor_name, emb_model.named_parameters()))))


    optimizer = torch.optim.AdamW([
        {'params': train_params},
        {'params': pretrain_params, 'lr': args.pretrain_lr}],
        lr=args.init_lr, weight_decay=1e-8)
    optimizer = torch.optim.Adam(train_params, lr=args.init_lr, weight_decay=1e-6)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, verbose=True)
    '''scheduler = NoamLR(optimizer, [args.warmup_epochs] * 2,
                       total_epochs=[args.epochs] * 2, steps_per_epoch = train_size / args.batch_size,
                       init_lr = [args.init_lr, args.pretrain_lr], max_lr=[args.max_lr, args.max_lr / 5], final_lr=[args.final_lr, args.final_lr / 5])'''
    best_val_loss = float('inf')
    for _ in range(args.epochs):
        emb_model.train()
        iter_count_train = 0
        epoch_label_loss = []
        epoch_train_loss = []
        for batch_idx, batch in enumerate(tqdm(train_loader, desc='Emb epoch {}'.format(_))):
            loss, KL_loss = emb_model(batch, finetune=True, reg_norm=reg_norm)
            # loss = loss.sum() / args.batch_size
            train_loss = loss + KL_loss
            optimizer.zero_grad()
            train_loss.backward()
            # torch.nn.utils.clip_grad_norm_(emb_model.parameters(), 5)
            optimizer.step()
            if isinstance(scheduler, NoamLR):
                scheduler.step()
            epoch_label_loss.append(loss.item())
            epoch_train_loss.append(train_loss.item())
            
            iter_count_train += 1
        label_loss_epoch = sum(epoch_label_loss) / iter_count_train
        train_loss_epoch = sum(epoch_train_loss) / iter_count_train
        print('Epoch %d: Label_loss %.4f' % (_, label_loss_epoch))
        print('Epoch %d: Loss %.4f' % (_, train_loss_epoch))


        # val
        emb_model.eval()
        iter_count_val = 0
        epoch_val_label_loss = []
        epoch_val_loss = []
        for batch_idx, batch in enumerate(tqdm(val_loader, desc= 'Emb val epoch {}'.format(_))):
            loss, KL_loss = emb_model(batch, finetune=True, reg_norm=reg_norm)
            # loss = loss.sum() / args.batch_size
            train_loss = loss + KL_loss
            epoch_val_label_loss.append(loss.item())
            epoch_val_loss.append(train_loss.item())
            iter_count_val += 1
        val_label_loss_epoch = sum(epoch_val_label_loss) / iter_count_val
        val_loss_epoch = sum(epoch_val_loss) / iter_count_val
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss_epoch)
        print('Val epoch %05d: Label_loss %.4f' % (_, val_label_loss_epoch))
        print('Val epoch %05d: Loss %.4f' % (_, val_loss_epoch))

        if val_loss_epoch < best_val_loss:
            best_val_loss = val_loss_epoch
            save_path = os.path.join('./log/dual_finetuned', 'best_checkpoint_{}_{}.pt'.format(_, val_loss_epoch))
            torch.save(emb_model, save_path)
        
    return save_path


class SmilesDataset(Dataset):
    def __init__(self, root, pre_split, dataset_type, dataset_name, norm, pretrain, tdc_data=False):
        # self.pretrain = pretrain
        self.smiles_path = root
        self.labels, self.task_num = self.read_labels()
        self.smiles = self.read_smiles()
        self.pre_split = pre_split
        self.dataset_type = dataset_type
        if self.pre_split:
            self.split = self.get_split(dataset_type, dataset_name)
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
        if self.dataset_type == 'regression':
            return self.smiles[idx], self.labels[idx], self.scaled_labels[idx]
        else:
            if self.task_num == 1:
                return self.smiles[idx], self.labels[idx]
            else:
                labels = []
                for i in range(self.task_num):
                    labels.append(self.labels[i][idx])
                return self.smiles[idx], labels
    
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
    
    def read_smiles(self):
        smiles = pd.read_csv(self.smiles_path)
        smiles = smiles.iloc[:, 0].tolist()
        return smiles
    
    def get_split(self, dataset_type, dataset_name):
        df = pd.read_csv(self.smiles_path)
        save_path = f'./dataset/ADMET/{dataset_type}/{dataset_name}/{dataset_name}_split.pkl'
        if 'scaffold_train_test_label' not in df.columns:
            raise ValueError(f'No scaffold column')
        train_idx = df.index[df['scaffold_train_test_label'].isin(['train', 'train_idx'])].tolist()
        test_idx = df.index[df['scaffold_train_test_label'].isin(['test', 'test_idx'])].tolist()
        split = defaultdict()
        val_size = len(test_idx) // 2 + 1
        val_idx = test_idx[:val_size]
        test_idx = test_idx[val_size:]
        split['train'] = train_idx
        split['val'] = val_idx
        split['test'] = test_idx
        with open(save_path, 'wb') as f:
            pickle.dump(split, f)
        return split
    
class StratifiedSampler(Sampler):
    def __init__(self, labels, num_samples):
        self.labels = np.array(labels)
        self.num_samples = num_samples
        self.class_indices = {label: np.where(self.labels == label)[0] for label in np.unique(self.labels)}

    def __iter__(self):
        samples = []
        for label, indices in self.class_indices.items():
            chosen_indices = np.random.choice(indices, size=self.num_samples // len(self.class_indices), replace=True)
            samples.extend(chosen_indices)
        np.random.shuffle(samples)
        return iter(samples)
    
    def __len__(self):
        return self.num_samples
    
    
def finetuning_testing(model, emb_model, args, seed, dataset, model_save, logger):

    # init seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    info = logger.info if logger is not None else print
    if not args.pre_split:
        if args.split_type == 'random':
            test_size = int(len(dataset) * 0.1)
            val_size = test_size
            train_data_size = len(dataset) - test_size
            train_dataset, test_data = random_split(dataset, [train_data_size, test_size],
                                                    generator=torch.Generator().manual_seed(seed))
            train_data, valid_data = random_split(train_dataset, [len(train_dataset) - val_size, val_size],
                                                generator=torch.Generator().manual_seed(seed))
        elif args.split_type == 'scaffold':
            train_idx, val_idx, test_idx = scaffold_class_split(dataset.smiles, dataset.labels, frac_valid=0.1, frac_test=0.1, balanced=False, tested=True)

            train_data_size = len(train_idx)
            val_size, test_size = len(val_idx), len(test_idx)
            train_data, valid_data, test_data = Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

    else:
        train_dataset, test_data = Subset(dataset, dataset.split['train']), Subset(dataset, dataset.split['test'])
        val_size = int(len(train_dataset) * 0.2)
        test_size = len(test_data)
        train_data, valid_data = random_split(train_dataset, [len(train_dataset) - val_size, val_size],
                                              generator=torch.Generator().manual_seed(args.seed))
        train_data_size = len(train_data)

    model.create_predictor(args, args.output_size)

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # init loss_func and metric_func
    if args.dataset_type == 'classification':
        metric_func = roc_auc_score
        loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        metric_func = rmse
        loss_func = torch.nn.MSELoss(reduction='none')
    
    train_name, pretrain_name, freeze_name = [], [], []
    for name, param in model.named_parameters():
        if 'emb_model' in name:
            freeze_name.append(name)
        elif 'predictor' in name:
            train_name.append(name)
        else:
            pretrain_name.append(name)
    
    train_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in train_name, model.named_parameters())))
    )
    pretrain_params = list(
        map(lambda x: x[1], list(filter(lambda kv: kv[0] in pretrain_name, model.named_parameters())))
    )

    optimizer = torch.optim.Adam([
        {'params': train_params},
        {'params': pretrain_params, 'lr': args.pretrain_lr}],
        lr=args.init_lr, weight_decay=1e-6)
    scheduler = NoamLR(optimizer, [args.warmup_epochs] * 2,
                       total_epochs=[args.epochs] * 2, steps_per_epoch=train_data_size // args.batch_size,
                       init_lr=[args.init_lr, args.init_lr / 5], max_lr=[args.max_lr, args.max_lr / 5], final_lr=[args.final_lr, args.final_lr / 5])
    
    # Run lossing
    all_loss = []
    best_score = float('inf') if args.minimize_score else -float('inf')
    best_test_score = float('inf') if args.minimize_score else -float('inf') 
    model_path = None
    # test
    # for _ in range(1):
    for _ in range(args.epochs):
        model.train()
        iter_count_train = 0
        iter_count_valid = 0
        epoch_train_loss = []
        epoch_valid_loss = []
        # test for _iter in range(2):
        for _iter in tqdm(range(train_data_size // args.batch_size), desc='training'):
            model.zero_grad()
            if args.dataset_type == 'regression':
                train_smiles, _labels, train_labels = next(iter(train_dataloader))
            else:
                train_smiles, train_labels = next(iter(train_dataloader))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            train_labels = train_labels.to(device)
            # process train_smiles to data
            pred = model.finetune(train_smiles, train_labels, emb_model, args).reshape(-1)
            loss = loss_func(pred, train_labels.float())
            loss = loss.sum() / args.batch_size

            loss.backward()
            optimizer.step()
            if isinstance(scheduler, NoamLR):
                scheduler.step()
            epoch_train_loss.append(loss.item())
            iter_count_train += 1
        train_loss_epoch = sum(epoch_train_loss) / iter_count_train
        info('Epoch %d: Loss %.4f' % (_, train_loss_epoch))

        val_scores, val_acc = predict_eval(model=model, dataloader=val_dataloader, metric_func=metric_func,
                                          all_size=val_size, batch_size=args.batch_size, model_args=model_args,
                                          task=args.dataset_type, norm=reg_norm, emb_model=emb_model)
        avg_val_score = np.nanmean(val_scores)
        info('Epoch %05d: Val_score %.4f' % (_, float(avg_val_score)))

        if not args.minimize_score and avg_val_score > best_score or \
            args.minimize_score and avg_val_score < best_score:
            best_score, epoch = avg_val_score, _
            model_path = os.path.join(args.model_save, f'{epoch}-{best_score}-facl.pt')
            save_checkpoint(model_path, model, args)
    
        # Evaluate on test set using model with best validation score
        test_scores, test_acc = predict_eval(model=model, dataloader=test_dataloader, metric_func=metric_func, 
                                            all_size =test_size, batch_size=args.batch_size, model_args=model_args,
                                            task=args.dataset_type, norm=reg_norm, emb_model=emb_model)

        avg_test_score = np.nanmean(test_scores)
        if avg_val_score == 1:
            info('val_score is 1, scanning best test_score')
            if avg_test_score > best_test_score:
                best_test_score = avg_test_score
        elif best_score == avg_val_score:
            best_test_score = avg_test_score
        info('Test %s score %.4f, acc %.4f' % (args.metric, avg_test_score, test_acc))
    info('Best epoch is %d: val_score %.4f, test_score %.4f' %(epoch, best_score, best_test_score))
    return best_test_score

def predict_eval(model, dataloader, metric_func, all_size, batch_size, model_args, task='classification', norm=None, emb_model=None, print_roc=False):
    model.eval()
    preds = torch.tensor([]).cuda()
    labels = torch.tensor([])

    for _iter in range(all_size // batch_size + 1):
        if task == 'regression':
            eval_smiles, eval_labels, _ = next(iter(dataloader))
        else:
            eval_smiles, eval_labels = next(iter(dataloader))

        if task == 'classification':
            sigmoid = torch.nn.Sigmoid()
                
            pred_labels = sigmoid(model.finetune(eval_smiles, eval_labels, emb_model, model_args).reshape(-1))
        else:
            pred_labels = model.finetune(eval_smiles, eval_labels, emb_model, model_args).reshape(-1)
        preds = torch.cat((preds, pred_labels), dim=-1)
        labels = torch.cat((labels, eval_labels), dim=-1)
    assert len(preds) == len(labels)
    nan = False
    if task == 'classification':
        if all(labels[i] == 0 for i in range(len(labels))) or all(labels[i] == 1 for i in range(len(labels))):
            nan = True
            print('Warning: Found a batch with true_labels all 0 or all 1')
        if all(preds[i] == 0 for i in range(len(preds))) or all(preds[i] == 1 for i in range(len(preds))):
            nan = True
            print('Warning: Found a batch with pred_labels all 0 or all 1')
    elif task == 'regression':
        preds = preds * norm[1] + norm[0]
    if nan:
        return float('nan')
    else:
        preds = preds.detach().cpu().numpy()
        # preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    labels_pred = np.where(preds > 0.5, 1, 0)
    count = 0
    for i in range(len(labels_pred)):
        if labels_pred[i] == labels[i]:
            count += 1

    return metric_func(labels, preds), count/len(labels_pred)

def save_checkpoint(save_path, model, args):
    torch.save(model, save_path)

def load_model(model_path):
    model = torch.load(model_path)
    return model

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


def get_emb_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/fgib_config.yml')
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--enable_progress_bar', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--minimize_score', type=bool, default=True)
    parser.add_argument('--step', type=str, choices=['train', 'finetune'], default='train',
                        help='Which step to run')
    parser.add_argument('--dataset_type', type=str, choices=['classification', 'regression'])
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args = merge_args_and_yaml(args, config)
    return args

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="regression",
                        type=str, choices=['classification', 'regression'], help="Which type of dataset")
    parser.add_argument("--root_dir",
                        default='./dataset/ADMET', type=str)
    parser.add_argument("--name", default="ppb", type=str)
    parser.add_argument("--pretrained_save", type=str,
                        default='./checkpoints/facl/pretrain_facl.pt')
    parser.add_argument("--pre_split", type=bool, default=True)
    parser.add_argument("--split_type", type=str, default='pre_split')
    parser.add_argument("--pretrain", type=bool, default=False, help="Used to determine stage of encoder")
    parser.add_argument("--exp_name", type=str, default='finetune')
    parser.add_argument("--dump_path", type=str, default='dumped')

    # test args
    parser.add_argument("--minimize_score", type=bool, default=True, help="How to determine best score")
    parser.add_argument("--metric", default=None, type=str, choices=['auc', 'cross_entropy', 'rmse'],
                        help="How to evaluate model")
    parser.add_argument("--num_runs", type=int, default=3, help="Nums of running test")

    # predictor argument
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate of predictor")
    parser.add_argument("--activation", type=str, default='LeakyReLU')
    parser.add_argument("--predictor_dims", type=int, default=130, help="Dimensions of predictor")
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--use_input_features", type=bool, default=False,
                        help="Whether to input atom_features in finetuning")
    parser.add_argument("--feature_dims", type=int, default=130, help="Dimensions of atom_features")
    parser.add_argument("--output_size", type=int, default=2, help="Should be determined by num_class")
    parser.add_argument("--hidden_dims", type=int, default=130, help="Hidden dimensions of linear in predictor")

    # optimizer
    parser.add_argument("--init_lr", type=float, default=5e-3, help="init_lr of finetuned params (emb 5e-3)")
    parser.add_argument("--pretrain_lr", type=float, default=5e-4, help="init_lr of pretrained params") # 1e-4
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight_decay of adam (emb 1e-6)") # 1e-8
    parser.add_argument("--max_lr", type=float, default=1e-2, help="Maximum learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-5, help="Final learning rate")


    # Training 
    parser.add_argument("--epochs", type=int, default=30, help="How many epochs to finetune")
    parser.add_argument("--warmup_epochs", type=int, default=4, help="How many warmup_epochs per finetune")# 5 -17 -20 -24
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size") 
    parser.add_argument("--seed", type=int, default=23, help="Random seed for emb_training")
    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--model_save", type=str, default='./log/finetuned_cl/')
    parser.add_argument("--regression_norm", type=str, default='zscore', choices=['minmax', 'zscore', 'log'])
    parser.add_argument("--gpus", type=int, default=1, help="How many GPU to use")
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    init_seed = 2025
    num_runs = 3
    emb_path = None
    model_args = get_args()
    emb_args = get_emb_args()

    # 加载及微调碎片嵌入模型
    emb_re = './checkpoint/fgib/best_emb_checkpoint.pt'
    emb_model = torch.load(emb_re)

    # init emb_dataset
    data_dir = f'{args.root_dir}/{args.dataset_type}/{args.name}'
    path_dict = {'smiles': f'{args.name}.csv', 'processed': 'processed.lmdb', 'split': f'{args.name}_scaffold.pkl'}
    emb_dataset = DatasetFrag(data_dir, path_dict, split=args.split_type, finetune=True)

    if not emb_path:
        emb_path = finetune_emb(args, emb_model, model_args, emb_dataset, emb_args)

    if args.regression_norm == 'zscore' and args.dataset_type == 'regression':
        reg_norm = [emb_dataset.mean_l, emb_dataset.std_l]
    else:
        reg_norm = None
    emb_model = init_frag_emb_model(args, model_args, reg_norm, model_path=emb_path)

    # init facl_dataset
    smiles_path = os.path.join(data_dir, path_dict['smiles'])
    dataset = SmilesDataset(smiles_path, args.pre_split, args.dataset_type, args.name, args.regression_norm, pretrain=False)
    args.output_size = dataset.task_num

    # determine metric type
    if args.metric is None:
        if args.dataset_type == 'classification' and args.output_size == 1:
            args.metric = 'auc'
        elif args.dataset_type == 'regression':
            args.metric = 'rmse'
        else:
            args.metric = 'cross_entropy'
    
    logger, args.save_dir = initialize_exp(Namespace(**args.__dict__))
    info = logger.info if logger is not None else print
    

    for k in args.__dict__:
        info(k + ':' + str(args.__dict__[k]))
    all_scores = []
    for _run in range(num_runs):
        run_seed = init_seed + _run
        model_save = os.path.join(args.model_save, f'run-{_run}')
        os.makedirs(model_save, exist_ok=True)
        # test_scores = training_testing(args, run_seed, model_save, emb_model)
        if os.path.exists(args.pretrained_save):
            facl_model = torch.load(args.pretrained_save)
        test_scores = finetuning_testing(facl_model, emb_model, args, run_seed, dataset, model_save, logger)
        all_scores.append(test_scores)
    for score in all_scores:
        info('Test score is %.4f' % score)
    all_scores = np.array(all_scores)
    mean_score, std_score = np.nanmean(all_scores), np.nanstd(all_scores)
    info(f'Results: {mean_score:.4f} +/- {std_score:.4f}')