import os
import pickle
import argparse

import time
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything

from dataset import Evaluate_Dataset
from models import Evaluate4rec, Kiosk4Rec, ALBert4Rec, Bert4Rec, GRU4Rec, SASRec

def load_model(args, vocab_size):
    model = None
    if args.name == 'kiosk4rec':
        model = Kiosk4Rec(args=args, vocab_size=vocab_size)
    elif args.name == 'albert4rec':
        model = ALBert4Rec(args=args, vocab_size=vocab_size)
    elif args.name == 'bert4rec':
        model = Bert4Rec(args=args, vocab_size=vocab_size)
    elif args.name == 'gru4rec':
        model = GRU4Rec(args=args, vocab_size=vocab_size)
    elif args.name == 'sasrec':
        model = SASRec(args=args, vocab_size=vocab_size)
    return model


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--name', default='kiosk4rec')
    args.add_argument('--seed', default=42)
    args.add_argument('--device', default=0)
    args.add_argument('--runs', default=5)

    args.add_argument('--epochs', default=100)
    args.add_argument('--lr', default=0.0001)
    args.add_argument('--batch_size', default=512)

    args.add_argument('--embed_size', default=128)
    args.add_argument('--num_layers', default=12)
    args.add_argument('--num_heads', default=12)
    args.add_argument('--hidden_size', default=768)
    args.add_argument('--dropout', default=0.0)
    args = args.parse_args()

    seed_everything(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # Load DataLoader
    vocab = pickle.load(open(os.getcwd() + '/dataset/transaction/vocab.pkl', 'rb'))

    # Train & Valid
    metric = {run: {'train_loss': [], 'valid_loss': []} for run in range(1, args.runs+1)}
    for run in range(1, args.runs+1):
        transaction = {
            'train': pickle.load(open(os.getcwd() + f'/dataset/transaction/train/train_{run}.pkl', 'rb')),
            'valid': pickle.load(open(os.getcwd() + f'/dataset/transaction/valid/valid_{run}.pkl', 'rb'))
        }

        dataloader = {
            'train': DataLoader(
                Evaluate_Dataset(transaction['train'], vocab, args.name),
                batch_size=args.batch_size, shuffle=True, num_workers=8
            ),
            'valid': DataLoader(
                Evaluate_Dataset(transaction['valid'], vocab, args.name, train=False),
                batch_size=args.batch_size, num_workers=8
            )
        }
        # Load Model
        # model = load_model(args=args, vocab_size=vocab['size']).to(device)
        model = torch.load(os.getcwd()+f'/outputs/pretrain/model_1.pth').to(device)
        if run == 1:
            print(f'{args.name.upper():10s} #Params {sum([p.numel() for p in model.parameters()])}')
        evaluater = Evaluate4rec(model).to(device)

        optimizer = torch.optim.AdamW(params=evaluater.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs/10))

        print()
        time.sleep(3)
        for epoch in range(1, args.epochs+1):
            for task in ['train']:
                evaluater.train()

                avg_loss = 0.
                batch_iter = tqdm(enumerate(dataloader[task]), desc=f'RUN{run:02d}_EP{epoch:02d}_{task}', total=len(dataloader[task]))
                for i, batch in batch_iter:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    outputs = evaluater(batch)

                    loss = nn.CrossEntropyLoss(ignore_index=0)(outputs.transpose(1, 2), batch['t_label'])
                    avg_loss += loss

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    batch_iter.set_postfix({'loss': loss.item(), 'avg_loss': avg_loss.item() / (i+1)})
                metric[run][f'{task}_loss'].append(avg_loss.item() / len(batch_iter))
                scheduler.step()

            for task in ['valid']:
                evaluater.eval()

                avg_loss = 0.
                batch_iter = tqdm(enumerate(dataloader[task]), desc=f'RUN{run:02d}_EP{epoch:02d}_{task}', total=len(dataloader[task]))
                for i, batch in batch_iter:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    with torch.no_grad():
                        outputs = evaluater(batch)
                    loss = nn.CrossEntropyLoss(ignore_index=0)(outputs.transpose(1, 2), batch['t_label'])
                    avg_loss += loss

                    batch_iter.set_postfix({'loss': loss.item(), 'avg_loss': avg_loss.item() / (i + 1)})
                metric[run][f'{task}_loss'].append(avg_loss.item() / len(batch_iter))
                if min(metric[run][f'{task}_loss']) == avg_loss.item() / len(batch_iter):
                    torch.save(evaluater.cpu(), os.getcwd()+f'/outputs/{args.name}/model_{run:02d}.pth')
                    evaluater.to(device)
    torch.save(metric, os.getcwd()+f'/outputs/{args.name}/metric.tar')
