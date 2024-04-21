import torch
import numpy as np

def hit_k(predicts, labels, k):
    _, recommends = torch.topk(torch.tensor(predicts), k=k)
    recommends = recommends.detach().numpy().tolist()
    score = [1 if label in recommend else 0 for label, recommend in zip(labels, recommends)]

    return np.mean(score)

def map_k(predicts, labels, k):
    user_score = []
    for k in range(1, k+1):
        _, recommends = torch.topk(torch.tensor(predicts), k=k)
        recommends = recommends.detach().numpy().tolist()
        score = [1 if label in recommend else 0 for label, recommend in zip(labels, recommends)]
        user_score.append(score)
    return np.mean(user_score)

def ndcg_k(predicts, labels, k):
    _, recommends = torch.topk(torch.tensor(predicts), k=k)
    recommends = recommends.detach().numpy().tolist()
    score = [np.reciprocal(np.log2(recommend.index(label) + 2)) if label in recommend else 0 for label, recommend in zip(labels, recommends)]

    return np.mean(score)