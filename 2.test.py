import os
import pickle
import argparse

import time
import torch
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import DataLoader

from pytorch_lightning.utilities.seed import seed_everything

from utils import hit_k, ndcg_k, map_k
from dataset import Evaluate_Dataset

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--name', default='kiosk4rec')
    args.add_argument('--seed', default=1004)
    args.add_argument('--device', default=0)
    args.add_argument('--runs', default=5)
    args.add_argument('--topk', default=3)

    args.add_argument('--batch_size', default=1024)

    args = args.parse_args()

    seed_everything(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # Load DataLoader
    vocab = pickle.load(open(os.getcwd() + '/dataset/transaction/vocab.pkl', 'rb'))

    # TEST
    test_metric = {'hitk': [], 'mapk': [], 'ndcgk': []}
    for run in range(1, args.runs + 1):

        transaction = {
            'test': pickle.load(open(os.getcwd() + f'/dataset/transaction/test/test_{run}.pkl', 'rb'))
        }

        dataloader = {
            'test': DataLoader(
                Evaluate_Dataset(transaction['test'], vocab, args.name, train=False),
                batch_size=args.batch_size, num_workers=8
            )
        }

        evaluater = torch.load(os.getcwd() + f'/outputs/{args.name}/model_{run:02d}.pth').to(device)
        for task in ['test']:
            evaluater.eval()

            avg_hitk, avg_mapk, avg_ndcgk = 0., 0., 0.
            batch_iter = tqdm(enumerate(dataloader[task]), desc=f'RUN{run:02d}_{task}', total=len(dataloader[task]))
            for i, batch in batch_iter:
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.no_grad():
                    outputs = evaluater(batch)
                labels_idx = np.array([np.nonzero(t_label)[0][0].cpu().numpy() for t_label in batch['t_label']])
                labels = np.array([t_label[labels_idx[i]].cpu().numpy() for i, t_label in enumerate(batch['t_label'])])
                predicts = np.array([output[labels_idx[i]].cpu().numpy() for i, output in enumerate(outputs)])

                hitk = hit_k(predicts, labels, int(args.topk))
                mapk = map_k(predicts, labels, int(args.topk))
                ndcgk = ndcg_k(predicts, labels, int(args.topk))

                avg_hitk += hitk
                avg_mapk += mapk
                avg_ndcgk += ndcgk

                batch_iter.set_postfix({'hitk': hitk, 'mapk': mapk, 'ndcgk': ndcgk,
                                        'avg_hitk': avg_hitk / (i + 1), 'avg_mapk': avg_mapk / (i + 1), 'avg_ndcgk': avg_ndcgk / (i + 1)})
            test_metric["hitk"].append(avg_hitk / len(batch_iter))
            test_metric["mapk"].append(avg_mapk / len(batch_iter))
            test_metric["ndcgk"].append(avg_ndcgk / len(batch_iter))
        time.sleep(3)
    print(f'BEST TEST  HITK : {np.array(test_metric["hitk"]).mean():.4f} ± {np.array(test_metric["hitk"]).std():.4f}')
    print(f'BEST TEST  MAPK : {np.array(test_metric["mapk"]).mean():.4f} ± {np.array(test_metric["mapk"]).std():.4f}')
    print(f'BEST TEST NDCGK : {np.array(test_metric["ndcgk"]).mean():.4f} ± {np.array(test_metric["ndcgk"]).std():.4f}')