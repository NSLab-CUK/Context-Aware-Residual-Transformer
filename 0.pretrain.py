import os
import pickle
import argparse

import time
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from pytorch_lightning import seed_everything

from dataset import Pretrain_Dataset
from models import Pretrain4rec, Kiosk4Rec

if __name__ == "__main__":
    args = argparse.ArgumentParser()
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

    transaction = {
        'pretrain': pickle.load(open(os.getcwd() + '/dataset/transaction/pretrain.pkl', 'rb'))
    }

    dataloader = {
        'pretrain': DataLoader(
            Pretrain_Dataset(transaction['pretrain'], vocab),
            batch_size=args.batch_size, shuffle=True, num_workers=8
        )
    }

    # Load Model
    model = Kiosk4Rec(args=args, vocab_size=vocab['size']).to(device)
    print(f'{"Kiosk4Rec":10s} #Params {sum([p.numel() for p in model.parameters()])}')

    evaluater = Pretrain4rec(model).to(device)

    optimizer = torch.optim.AdamW(params=evaluater.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(args.epochs/10))

    print()
    time.sleep(3)
    metric = {'pretrain_loss': []}
    for epoch in range(1, args.epochs+1):
        for task in ['pretrain']:
            evaluater.train()

            avg_loss = 0.
            batch_iter = tqdm(enumerate(dataloader[task]), desc=f'EP{epoch:02d}_{task}', total=len(dataloader[task]))
            for i, batch in batch_iter:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = evaluater(batch)

                user_loss = nn.CrossEntropyLoss()(outputs[0], batch['same_user'])
                domain_loss = nn.CrossEntropyLoss()(outputs[1], batch['same_domain'])
                a_loss = nn.CrossEntropyLoss(ignore_index=0)(outputs[2].transpose(1, 2), batch['a_label'])
                b_loss = nn.CrossEntropyLoss(ignore_index=0)(outputs[3].transpose(1, 2), batch['b_label'])

                loss = (user_loss + domain_loss) # 1
                # loss = (domain_loss) + ((a_loss + b_loss) / 2) # 2
                # loss = (user_loss) + ((a_loss + b_loss) / 2) # 3
                avg_loss += loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_iter.set_postfix({'user': user_loss.item(), 'domain': domain_loss.item(),
                                        'a_loss': a_loss.item(), 'b_loss': b_loss.item(),
                                        'loss': loss.item(), 'avg_loss': avg_loss.item() / (i+1)})
            metric[f'{task}_loss'].append(avg_loss.item() / len(batch_iter))
            scheduler.step()
            if min(metric[f'{task}_loss']) == avg_loss.item() / len(batch_iter):
                torch.save(evaluater.base_model1.cpu(), os.getcwd()+f'/outputs/pretrain/model_1.pth')
                evaluater.base_model1.to(device)
    torch.save(metric, os.getcwd()+f'/outputs/pretrain/metric_1.tar')
