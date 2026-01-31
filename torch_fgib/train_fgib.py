import argparse
import wandb
import yaml
import warnings
from argparse import Namespace
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
import os
import sys
sys.path.append("..")

from data.fgib_data import DatasetFrag
import torch
from torch.utils.data import Dataset, random_split, Subset
from fgib import FGIBModel
from tqdm import tqdm
from torch.optim import AdamW, Adam, SGD
import copy
import datetime
from tqdm import tqdm
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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

def build_optimizer(model, args):
    if args.step == 'train':
        optimizer = AdamW(model.parameters(), lr=float(args.optimizer.lr), weight_decay=args.optimizer.weight_decay)
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


def train_epoch(model, dataloader, epoch, optimizer, beta, device=None):
    model.train()
    epoch_loss = 0
    for batch_idx, batch in tqdm(enumerate(dataloader), desc='Training:'):
        batch.to(device)
        regression_loss, KL_loss, preserve_rate, preserve_std, re_mae = model(batch, device=device)
        loss = regression_loss + KL_loss * float(beta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        wandb.log({'loss/train': loss.item(), 'remae/train': re_mae.item(),
                   'w_rate/train': preserve_rate.item(), 'w_std/train': preserve_std.item()})
    epoch_loss = epoch_loss / (batch_idx + 1)
    wandb.log({'loss/train_epoch': epoch_loss})
    return epoch_loss


def eval_model(model, dataloader, epoch, step, device=None):
    model.eval()
    valid_loss = 0
    for batch_idx, batch in tqdm(enumerate(dataloader), desc='Epoch {}'.format(epoch)):
        batch.to(device)
        loss, _, _, _, re_mae = model(batch, device=device)
        valid_loss += loss.item()
        wandb.log({f'{step}': loss.item()})
        wandb.log({f'{step}_mae': re_mae.item()})
    epoch_loss = valid_loss / (batch_idx + 1)
    wandb.log({f'{step}_epoch': epoch_loss})
    return epoch_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/fgib_config.yml')
    # parser.add_argument('--fingerprint_bits', type=int, default=679)
    # parser.add_argument('--regression', type=int, default=1613)
    parser.add_argument('--resume', type=str, default=None)
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
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    args = merge_args_and_yaml(args, config)
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")

    wandb.login()

    run = wandb.init(
        project='fgib_02',
        name=f'fgib-{args.dataset.batch_size}-{current_time}',
        config={
            'learning_rate': args.optimizer.lr,
            'epochs': args.n_epochs,
            'beta': args.model.beta
        }
    )

    # init dataset
    if args.step == 'train':
        dataset = DatasetFrag(args.dataset.root, args.dataset.path_dict, split='random')
    else:
        dataset = DatasetFrag(args.dataset.root, args.dataset.path_dict, split='random', finetune=True)

    # split
    test_size = int(len(dataset) * 0.1)
    train_size = len(dataset) - test_size * 2
    val_size = test_size
    if args.dataset.split == 'random':
        train_data, test_dataset = random_split(dataset, [train_size + val_size, test_size],
                                                generator=torch.Generator().manual_seed(args.seed))
        train_dataset, val_dataset = random_split(train_data, [train_size, val_size],
                                                  generator=torch.Generator().manual_seed(args.seed))
    elif args.dataset.split == 'presplit':
        raise(Exception('Not implemented yet!'))
    else:
        raise(Exception('Not implemented yet!'))

    # dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.dataset.batch_size, shuffle=True, num_workers=16, 
                              persistent_workers=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.dataset.batch_size, shuffle=False, 
                            num_workers=16, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=args.dataset.batch_size, shuffle=False, 
                             num_workers=16)

    # init fgib model
    model = FGIBModel(args.model)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    # init optimizer
    optimizer = build_optimizer(model, args)
    if args.minimize_score:
        best_score, best_checkpoint = float('inf'), None
    else:
        best_score, best_checkpoint = -float('inf'), None

    # init scheduler
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=args.optimizer.patience, verbose=True)

    # train
    for epoch in tqdm(range(1, args.n_epochs + 1)):
        print('Epoch {}:'.format(epoch))
        loss = train_epoch(model, train_loader, epoch, optimizer=optimizer, beta=args.model.beta, device=device)

        # evaluate validation
        score = eval_model(model, val_loader, epoch, 'loss_valid', device=device)
        test_score = eval_model(model, test_loader, epoch, 'loss_test', device=device)

        if args.minimize_score and score < best_score:
            best_score = score
            best_checkpoint = copy.deepcopy(model.state_dict())
        elif not args.minimize_score and score > best_score:
            best_score = score
            best_checkpoint = copy.deepcopy(model.state_dict())
        else:
            continue

    model.load_state_dict(best_checkpoint)
    score_best_checkpoint = eval_model(model, test_loader, 'test', 'loss_test', device=device)
    print()
    save_dir = os.path.join(args.logdir, 'best_checkpoint.pt')
    torch.save(model, save_dir)




