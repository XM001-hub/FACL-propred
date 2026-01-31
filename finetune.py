import torch
import argparse
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from data.fgib_data import DatasetFrag
from torch.utils.data import random_split, Subset
from torch_geometric.data import DataLoader
from pretrain import get_args
from torch_fgib.fgib import FGIBModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset
from torch_facl.facl_model import FACL
from splitter import scaffold_class_split, scaffold_split
import random
import webcolors as bcolors
from torch_facl.facl_utils import rmse, NoamLR, mol2graph, StandardScaler
from rdkit import RDLogger
from t_finetune import SmilesDataset, StratifiedSampler
from logger_utils import initialize_exp
from argparse import Namespace
from sklearn.model_selection import train_test_split
import torch.nn as nn
import math


def predict_eval(encoder, dataloader, metric_func, all_size, batch_size, model_args, task='classification', norm=None, task_num=1, scaler=None):

    encoder.eval()
    preds = torch.tensor([]).cuda()
    labels = torch.tensor([]).cuda()
    one = False

    for batch in dataloader:
        if task == 'regression':
            eval_smiles, eval_labels, _ = batch[0], batch[1], batch[2]
            eval_labels = eval_labels.cuda()
            # eval_labels = eval_labels * scaler[1] + scaler[0]
        else:
            eval_smiles, eval_labels = batch[0], batch[1]
            if not len(eval_labels) == len(eval_smiles):
                mask, targets = process_labels(eval_labels)
                mask = mask.cuda()
                eval_labels = targets.cuda()
                # eval_labels = list_transport(eval_labels).cuda()
                '''ori_labels = eval_labels
                eval_labels = torch.tensor([]).cuda()
                for label in ori_labels:
                    eval_labels = torch.cat((eval_labels, label.cuda()))'''
            else:
                one = True
                eval_labels = eval_labels.cuda()
        data = mol2graph(eval_smiles, augmentation=False, args=model_args)
        if task == 'classification':
            sigmoid = torch.nn.Sigmoid()
            pred_labels = sigmoid(encoder(data, pretrain=False))
        else:
            pred_labels = encoder(data, pretrain=False)
            pred_labels = pred_labels * scaler[1] + scaler[0]
        preds = torch.cat((preds, pred_labels), dim=0)
        labels = torch.cat((labels, eval_labels), dim=0)
    if one:
        labels = labels.reshape(preds.size())
    assert len(preds) == len(labels)
    '''if task == 'classification':
        if all(labels[i] == 0 for i in range(len(labels))) or all(labels[i] == 1 for i in range(len(labels))):
            nan = True
            print('Warning: Found a batch with true_labels all 0 or all 1')
        if all(preds[i] == 0 for i in range(len(preds))) or all(preds[i] == 1 for i in range(len(preds))):
            nan = True
            print('Warning: Found a batch with pred_labels all 0 or all 1')
    elif task == 'regression':
        preds = preds * norm[1] + norm[0]'''
    nan, valid_labels, valid_preds = eval_labels_class(preds, labels, task_num, task)
    if nan: 
        return float('nan'), float('nan')
    else:
        preds = preds.detach().cpu().numpy()
        labels = labels.cpu().numpy()
        valid_labels = valid_labels.cpu().numpy()
        valid_preds = valid_preds.detach().cpu().numpy()

    if task == 'classification':
        count = count_acc(valid_preds, valid_labels, task_num, where=0.5)
        
        return metric_func(labels, preds), count/(len(preds) * task_num)
    else:
        return metric_func(labels, preds)

def count_acc(preds, labels, task_num, where=0.5):
    count = 0
    labels_pred = np.where(preds > 0.5, 1, 0)
    for i in range(task_num):
        for j in range(labels.shape[1]):
            if labels_pred[i][j] == labels[i][j]:
                count += 1
    return count


def eval_labels_class(preds, labels, task_num, task):
    nan = False
    valid_preds, valid_labels = preds.T, labels.T
    for i in range(task_num):
        if task == 'classification':
            if all(label == 0 for label in valid_labels[i]) or all(label == 1 for label in valid_labels[i]):
                nan = True
                print('Warning: Found a batch with true_labels all 0 or all 1')
            if all(pred == 0 for pred in valid_preds[i]) or all(pred == 1 for pred in valid_preds[i]):
                nan = True
                info('Warning: Found a task with predictions all 0s or all 1s')
    return nan, valid_labels, valid_preds

def get_scheduler(args, optimizer):
    if args.scheduler == 'NoamLR':
        scheduler = NoamLR(optimizer, [args.warmup_epochs] * 2,
                            total_epochs=[args.epochs] * 2, steps_per_epoch=args.train_data_size // args.batch_size,
                            init_lr=[args.init_lr, args.init_lr / 5], max_lr=[args.max_lr, args.max_lr / 5], final_lr=[args.final_lr, args.final_lr / 5])

    elif args.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=20, verbose=True)
    return scheduler

def save_checkpoint(save_path, model, args):
    torch.save(model, save_path)

def stratified_split(dataset, test_size=0.1, val_size=0.11, random_state=2025):

    if hasattr(dataset, 'labels'):
        labels = dataset.labels
    labels = np.array(labels)
    indices = np.arange(len(labels))

    train_val_idx, test_idx = train_test_split(
        indices, 
        test_size=test_size,
        stratify=labels,
        random_state=random_state
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_size,
        stratify=labels[train_val_idx],
        random_state=random_state
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    return train_set, val_set, test_set


def list_transport(list_tensor, device='cuda:0'):
    tensor_tuple = tuple(list_tensor)
    target = torch.stack(tensor_tuple, dim=0)
    '''for _tensor in list_tensor:
        tensor = _tensor.to_(device)
        target = torch.stack((target, tensor), dim=0)'''
    return target.T

def process_labels(labels):
    mask_list = []
    for t in labels:
        mask_t = (~torch.isnan(t)).int()
        mask_list.append(mask_t)
    mask_2d = torch.stack(mask_list, dim=0)

    m_list = []
    for t in labels:
        m_t = torch.zeros_like(t, dtype=torch.int)
        nan_mask = ~torch.isnan(t)
        m_t[nan_mask] = t[nan_mask].to(torch.int)
        m_list.append(m_t)
    target_2d = torch.stack(m_list, dim=0)

    return mask_2d.T, target_2d.T


def pre_split(smiles):
    return None, None, None


def encoder_testing(model, args, seed, dataset, model_save, model_args=None, logger=None, scaler=None):

    info = logger.info if logger is not None else print
    # init seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if not args.pre_split:
        if args.split_type == 'random':
            test_size = int(len(dataset) * 0.1)
            val_size = test_size
            train_data_size = len(dataset) - test_size
            args.train_data_size = train_data_size
            train_dataset, test_data = random_split(dataset, [train_data_size, test_size],
                                                    generator=torch.Generator().manual_seed(int(seed)))
            train_data, valid_data = random_split(train_dataset, [len(train_dataset) - val_size, val_size],
                                                generator=torch.Generator().manual_seed(int(seed)))
        elif args.split_type == 'scaffold_class':
            train_idx, val_idx, test_idx = scaffold_class_split(dataset.smiles, dataset.labels, frac_valid=0.1, frac_test=0.1, balanced=False, tested=True)

            train_data_size = len(train_idx)
            val_size, test_size = len(val_idx), len(test_idx)
            train_data, valid_data, test_data = Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
        elif args.split_type == 'stratify':
            train_data, valid_data, test_data = stratified_split(dataset, random_state=seed)
            args.train_data_size = len(train_data)
            train_data_size = len(train_data)
            val_size, test_size = len(valid_data), len(test_data)
        elif args.split_type == 'scaffold':
            train_idx, val_idx, test_idx = scaffold_split(dataset.smiles, frac_valid=0.1, frac_test=0.1)
            train_data_size = len(train_idx)
            args.train_data_size = len(train_idx)
            val_size, test_size = len(val_idx), len(test_idx)
            train_data, valid_data, test_data = Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)

    else:
        train_idx, val_idx, test_idx = dataset.split['train'], dataset.split['val'], dataset.split['test']
        train_data_size = len(train_idx)
        val_size, test_size = len(val_idx), len(test_idx)
        args.train_data_size = len(train_idx)
        train_data, valid_data, test_data = Subset(dataset, train_idx), Subset(dataset, val_idx), Subset(dataset, test_idx)
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    
    encoder.create_predictor(args)

    '''for batch in val_dataloader:
        print(batch[1])'''
    # init loss_func and metric_func
    if args.dataset_type == 'classification':
        metric_func = roc_auc_score
        loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
    else:
        metric_func = rmse
        loss_func = torch.nn.MSELoss(reduction='none')
    
    train_name, pretrain_name, freeze_name = [], [], []
    for name, param in encoder.named_parameters():
        if 'predictor' in name:
            train_name.append(name)
            if param.dim() != 1:
                nn.init.xavier_normal_(param)
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
        lr=args.init_lr, weight_decay=1e-8)
    scheduler = get_scheduler(args, optimizer)

    # Run lossing
    best_score = float('inf') if args.minimize_score else -float('inf')
    best_test_score = float('inf') if args.minimize_score else -float('inf')
    model_path = None

    for _ in range(args.epochs):
        encoder.train()
        iter_count_train = 0
        epoch_train_loss = []

        for _iter in tqdm(range(train_data_size // args.batch_size), desc='training'):
            one = False
            encoder.zero_grad()
            if args.dataset_type == 'regression':
                train_smiles, _labels, train_labels = next(iter(train_dataloader))
                mean, std = dataset.mean_l, dataset.std_l
                scaler = [mean, std]
            else:
                train_smiles, train_labels = next(iter(train_dataloader))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if len(train_labels) == len(train_smiles):
                one = True
                train_targets = train_labels.to(device)
                mask = torch.ones_like(train_targets)
            else:
                mask, targets = process_labels(train_labels)
                mask = mask.to(device)
                train_targets = targets.to(device)
            data = mol2graph(train_smiles, augmentation=False, args=model_args)
            pred = encoder(data, pretrain=False) # (batch_size, num_tasks)
            # mask = torch.where(torch.isnan(train_targets), torch.zeros_like(train_targets), torch.ones_like(train_targets))
            if one:
                loss = loss_func(pred.reshape(-1), train_targets.float())
            else:
                loss = loss_func(pred, train_targets.float()) * mask # (batch_size, num_tasks)
            '''for i in range(len(loss)):
                row_loss = loss[i]'''
            loss = loss.sum() / mask.sum()

            loss.backward()
            optimizer.step()
            if isinstance(scheduler, NoamLR):
                scheduler.step()
            epoch_train_loss.append(loss.item())
            iter_count_train += 1
        train_loss_epoch = sum(epoch_train_loss) / iter_count_train
        info('Epoch %d: Loss %.4f' % (_, train_loss_epoch))
        if args.dataset_type == 'classification':
            val_scores, val_acc = predict_eval(encoder, val_dataloader, metric_func, val_size, args.batch_size, model_args=model_args, task=args.dataset_type, task_num=args.output_size, scaler=scaler)
        else:
            val_scores = predict_eval(encoder, val_dataloader, metric_func, val_size, args.batch_size, model_args=model_args, task=args.dataset_type, task_num=args.output_size, scaler=scaler)
        avg_val_score = np.nanmean(val_scores)
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_score)
        info('Epoch %05d: Val_score %.4f' % (_, float(avg_val_score)))

        if not args.minimize_score and avg_val_score > best_score or \
            args.minimize_score and avg_val_score < best_score:
            best_score, epoch = avg_val_score, _
            model_path = os.path.join(args.model_save, f'{epoch}-{best_score}-facl.pt')
            save_checkpoint(model_path, model, args)
        
        if args.dataset_type == 'classification':
            test_scores, test_acc = predict_eval(encoder, test_dataloader, metric_func, test_size, args.batch_size, model_args=model_args, task=args.dataset_type, task_num=args.output_size, scaler=scaler)
        else:
            test_scores = predict_eval(encoder, test_dataloader, metric_func, test_size, args.batch_size, model_args=model_args, task=args.dataset_type, task_num=args.output_size, scaler=scaler)
        avg_test_score = np.nanmean(test_scores)
        if avg_val_score == 1:
            info('val_score is 1, scanning best test_score')
            if avg_test_score > best_test_score:
                best_test_score = avg_test_score
        elif best_score == avg_val_score:
            best_test_score = avg_test_score
        info('Test %s score %.4f' % (args.metric, avg_test_score))
        # info('Test %s score %.4f, acc %.4f' % (args.metric, avg_test_score, test_acc))
    info('Best epoch is %d: val_score %.4f, test_score %.4f' %(epoch, best_score, best_test_score))
    # if not args.minimize_score and avg_val_score > best_score or 
    return best_test_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", default="classification",
                        type=str, choices=['classification', 'regression'], help="Which type of dataset")
    parser.add_argument("--root_dir",
                        default='./dataset/ADMET', type=str)
    parser.add_argument("--name", default="carc", type=str)
    parser.add_argument("--pretrained_old", type=str,
                        default='./log/gpscl_checkpoints/pretrained/val_epoch=0.01.ckpt')
    parser.add_argument("--pretrained_save", type=str,
                        default='./checkpoints/facl/pretrain_facl.pt')
    
    parser.add_argument("--pre_split", type=bool, default=False)
    parser.add_argument("--split_type", type=str, default='scaffold')
    parser.add_argument("--pretrain", type=bool, default=False, help="Used to determine stage of encoder")
    parser.add_argument("--dump_path", type=str, default='./dumped')
    parser.add_argument("--exp_name", type=str, default='finetune')
    parser.add_argument("--tdc_data", type=bool, default=False)

    # test args
    parser.add_argument("--minimize_score", type=bool, default=False, help="How to determine best score")
    parser.add_argument("--metric", default=None, type=str, choices=['auc', 'cross_entropy', 'rmse'],
                        help="How to evaluate model")
    parser.add_argument("--num_runs", type=int, default=3, help="Nums of running test")

    # predictor argument
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate of predictor")
    parser.add_argument("--activation", type=str, default='ReLU') # 2 - ReLU
    parser.add_argument("--predictor_dims", type=int, default=130, help="Dimensions of predictor")
    parser.add_argument("--num_layers", type=int, default=3) # 2 -1
    parser.add_argument("--use_input_features", type=bool, default=False,
                        help="Whether to input atom_features in finetuning")
    parser.add_argument("--feature_dims", type=int, default=130, help="Dimensions of atom_features")
    parser.add_argument("--output_size", type=int, default=1, help="Should be determined by num_class")
    parser.add_argument("--hidden_dims", type=int, default=130, help="Hidden dimensions of linear in predictor")

    # optimizer
    parser.add_argument("--init_lr", type=float, default=1e-3, help="init_lr of finetuned params")
    parser.add_argument("--pretrain_lr", type=float, default=1e-4, help="init_lr of pretrained params")
    parser.add_argument("--weight_decay", type=float, default=1e-8, help="weight_decay of adam")
    parser.add_argument("--max_lr", type=float, default=1e-2, help="Maximum learning rate")
    parser.add_argument("--final_lr", type=float, default=1e-5, help="Final learning ra`te")
    parser.add_argument("--scheduler", type=str, default='NoamLR')

    # Training 
    parser.add_argument("--epochs", type=int, default=50, help="How many epochs to finetune")
    parser.add_argument("--warmup_epochs", type=int, default=5, help="How many warmup_epochs per finetune")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--init_seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--save_every", type=int, default=3)
    parser.add_argument("--model_save", type=str, default='/home/rujinxiao/Desktop/GPSCL-main/log/finetuned_cl/')
    parser.add_argument("--regression_norm", type=str, default='zscore', choices=['minmax', 'zscore', 'log'])
    parser.add_argument("--gpus", type=int, default=1, help="How many GPU to use")
    args = parser.parse_args()
    RDLogger.DisableLog('rdApp.*')

    init_seed = args.init_seed
    seeds = []
    num_runs = 3
    model_args = get_args()
    if init_seed:
        for i in range(num_runs):
            seeds.append(init_seed + i)
    else:
        seeds = np.random.randint(100, size=num_runs)
        print(seeds)

    # init gpscl_dataset
    data_dir = f'{args.root_dir}/{args.dataset_type}/{args.name}'
    smiles_path = os.path.join(data_dir, '{}.csv'.format(args.name))
    dataset = SmilesDataset(smiles_path, args.pre_split, args.dataset_type, args.regression_norm, pretrain=False, tdc_data=args.tdc_data)
    args.output_size = dataset.task_num

    # determine metric type
    if args.metric is None:
        if args.dataset_type == 'classification':
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
        run_seed = seeds[_run]
        random.seed(run_seed)
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        model_save = os.path.join(args.model_save, f'run-{_run}')
        os.makedirs(model_save, exist_ok=True)
        if os.path.exists(args.pretrained_save):
            facl_model = torch.load(args.pretrained_save)
            encoder = facl_model.get_encoder()
        test_scores = encoder_testing(encoder, args, run_seed, dataset, model_save, model_args=model_args, logger=logger)
        all_scores.append(test_scores)
    for score in all_scores:
        info('Test score is %.4f' % score)
    all_scores = np.array(all_scores)
    mean_score, std_score = np.nanmean(all_scores), np.nanstd(all_scores)
    info(f'Results: {mean_score:.4f} +/- {std_score:.4f}')