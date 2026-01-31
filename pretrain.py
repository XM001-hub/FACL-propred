from torch_facl.facl_model import FACL
import argparse
from argparse import Namespace
import pytorch_lightning as pl
import datetime
import os
import yaml
import warnings
from tqdm import tqdm
from pathlib import Path
from data.ZINC_dataset import SmilesDataset, MoleculeDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import AdamW, adam
import wandb
import sys
import torch
from data.fgib_data import DatasetFrag
from torch_geometric.data import DataLoader
from torch.utils.data import Dataset, random_split, Subset, DataLoader, SequentialSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
from torch.optim.lr_scheduler import ExponentialLR



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
    parser.add_argument('--config', type=str, default='./configs/facl_config.yml')
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

def train_epoch(model, dataloader, db_path=None, epoch=None, optimizer=None, scheduler=None, device=None, step_per_schedule=500):

    model.train()
    epoch_loss = 0
    # contruct triplet of molecule to learn
    for batch_idx, batch in tqdm(enumerate(dataloader), desc='Epoch {}'.format(epoch)):
        loss = model(batch, db_path=db_path)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.abs().mean().item()
        optimizer.step()
        optimizer.zero_grad()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        '''for name, param in model.named_parameters():
            if 'encoder1' in name:
                print(param)'''
        epoch_loss += loss.item()
        print('Loss: {}'.format(loss.item()))
        # wandb.log({'loss/train': loss.item()})
        if batch_idx % step_per_schedule == 0:
            scheduler.step()
    epoch_loss = epoch_loss / (batch_idx + 1)
    return epoch_loss

def val_epoch(model, dataloader, db_path=None, epoch=None, optimizer=None, scheduler=None, device=None):
    model.eval()
    val_loss = 0

    for batch_idx, batch in tqdm(enumerate(dataloader), desc='Epoch {}'.format(epoch)):
        loss = model(batch, db_path=db_path)
        val_loss += loss.item()
        scheduler.step(val_loss)
        print('Val_loss is {}'.format(float(loss)))
    val_loss = val_loss / (batch_idx + 1)
    return val_loss

def build_optimizer(model, args):
    if args.step == 'pretrain':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optimizer.lr, weight_decay=args.optimizer.decay)
    else:
        train_params, fine_params = [], []
        for name, param in model.named_parameters():
            if name.split('.')[0] in ['f_predictor', 'compressor', 'predictor']:
                train_params.append(param)
            else:
                fine_params.append(param)
        optimizer1 = AdamW(train_params, lr=args.optimizer.lr, weight_decay=args.optimizer.weight_decay)
        optimizer2 = AdamW(fine_params, lr=float(args.optimizer.lr) * 0.1, weight_decay=args.optimizer.weight_decay)
        optimizer = [optimizer1, optimizer2]
    return optimizer
    

if __name__ == '__main__':
    args1 = get_args()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/fgib_config.yml')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='./log_no_aug/')
    # Trainer
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--enable_progress_bar', type=bool, default=True)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--minimize_score', type=bool, default=True)
    # step
    parser.add_argument('--step', type=str, choices=['train', 'finetune'], default='train',
                        help='Which step to run')
    parser.add_argument('--dataset_type', type=str, choices=['classification', 'regression'])
    parser.add_argument('--frag_path', type=str, default='./dataset/ZINC_250k/fragments.db')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args = merge_args_and_yaml(args, config)
    
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    torch.manual_seed(2025)
    dataset = MoleculeDataset(args.dataset.root)
    train_size, valid_size = int(0.9 * len(dataset)), len(dataset) - int(0.9 * len(dataset))
    train_data, valid_data = random_split(dataset, lengths=[train_size, valid_size])
    smi, neg_smi, smi_smi, sim_weight, neg_weight = dataset[1]

    '''wandb.login()
    run = wandb.init(
        project='gpscl-25',
        name=f'gpscl-{args.dataset.batch_size}-{current_time}',
        config={
            'learning_rate': args.optimizer.lr,
            'epochs': args.n_epochs,
        },
        mode='offline'
    )'''
    # dataset = SmilesDataset(args.dataset.root, pre_split=False, pretrain=True, dataset_type=None, norm=None)
    # dataset = DatasetFrag(args2.dataset.root, args2.dataset.path_dict, split=args2.dataset.split)
    train_loader = DataLoader(train_data, batch_size=args.dataset.batch_size, shuffle=False, sampler=SequentialSampler(train_data))
    valid_loader = DataLoader(valid_data, batch_size=args.dataset.batch_size, shuffle=False, sampler=SequentialSampler(valid_data))

    # Load fgib_model
    emb_model = torch.load(args1.model_path)
    # emb_model.load_state_dict(torch.load(args1.model_path, weights_only=True))
    # emb_model.eval()
    # emb_model = FGIBModel.load_from_checkpoint(args1.model_path, args=args2.model, out_dir=None)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emb_model.to(device)

    gpscl = FACL(args1, emb_model=emb_model, out_dir=None, pretrain=True)
    gpscl.to(device)
    optimizer = build_optimizer(gpscl, args1)

    # init scheduler
    # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args1.optimizer.patience, verbose=True)
    scheduler = ExponentialLR(optimizer, 0.99, -1)

    epoch_idx, global_val_loss, best_checkpoint = 0, float('inf'), None
    # train
    for epoch in range(1, args.n_epochs + 1):
        loss = train_epoch(gpscl, train_loader, args.frag_path, epoch=epoch, optimizer=optimizer, scheduler=scheduler, device=device)
        val_loss = val_epoch(gpscl, valid_loader, args.frag_path, epoch=epoch, optimizer=optimizer, scheduler=scheduler, device=device)

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(val_loss)
        if val_loss < global_val_loss:
            global_val_loss = val_loss
            best_checkpoint = copy.deepcopy(gpscl.state_dict())
            epoch_idx = epoch
            checkpoint_path = '{}/best-{}-{}.pt'.format(args.log_dir, epoch_idx, val_loss)
            torch.save(gpscl, checkpoint_path)
            print('saving')
    
    