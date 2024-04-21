import torch
import random
from torch.utils.data import Dataset

class Evaluate_Dataset(Dataset):
    def __init__(self, transaction, vocab, model, train=True):
        super().__init__()
        self.transaction = transaction
        self.pad_idx = vocab["vocab_to_idx"]["<pad>"]
        self.mask_idx = vocab["vocab_to_idx"]["<mask>"]

        self.model = model
        self.train = train

    def __len__(self):
        return len(self.transaction)

    def __getitem__(self, idx):
        transaction = self.transaction[idx]

        t_input, t_label = self.random_mask(transaction)

        if self.model in ['kiosk4rec']:
            output = {
                "t_inputs": t_input[:1] + t_input[1:4] + t_input[4:5] + t_input[5:],
                "t_position": range(len(transaction[5:])),
                "t_label": t_label[:1] + t_label[1:4] + t_label[4:5] + t_label[5:]
            }
        elif self.model in ['albert4rec', 'bert4rec', 'gru4rec', 'sasrec']:
            output = {
                "t_inputs": t_input[5:],
                "t_position": range(len(transaction[5:])),
                "t_label": t_label[5:]
            }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_mask(self, transaction):
        inputs, labels = [], []
        index = (len(transaction) if transaction[-1] != self.pad_idx else transaction.index(self.pad_idx)) - 1  # last item

        for i, token in enumerate(transaction):
            if i < 5 or token == self.pad_idx:
                inputs.append(token)
                labels.append(self.pad_idx)
            else:
                if self.train:
                    prob = random.random()
                    inputs.append(self.mask_idx if prob < 0.2 else token)
                    labels.append(token if prob < 0.2 else self.pad_idx)
                else:
                    inputs.append(self.mask_idx if index == i else token)
                    labels.append(token if index == i else self.pad_idx)
        return inputs, labels


class Pretrain_Dataset(Dataset):
    def __init__(self, transaction, vocab):
        super().__init__()
        self.transaction = transaction
        self.pad_idx = vocab["vocab_to_idx"]["<pad>"]
        self.mask_idx = vocab["vocab_to_idx"]["<mask>"]

    def __len__(self):
        return len(self.transaction)

    def __getitem__(self, idx):
        a = self.transaction[idx]
        a_input, a_label = self.random_mask(a)

        b = self.transaction[random.randint(0, len(self.transaction)-1)]
        b_input, b_label = self.random_mask(b)

        output = {
            "a_inputs": a_input[:1] + a_input[1:4] + a_input[4:5] + a_input[5:],
            "a_label": a_label[:1] + a_label[1:4] + a_label[4:5] + a_label[5:],

            "b_inputs": b_input[:1] + b_input[1:4] + b_input[4:5] + b_input[5:],
            "b_label": b_label[:1] + b_label[1:4] + b_label[4:5] + b_label[5:],

            "t_position": range(15),
            "same_user": 1 if a_input[0] == b_input[0] else 0,
            "same_domain": 1 if a_input[4] == b_input[4] else 0
        }

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_mask(self, transaction):
        inputs, labels = [], []

        for i, token in enumerate(transaction):
            if i == 5 or token == self.pad_idx:
                inputs.append(token)
                labels.append(self.pad_idx)
            else:
                prob = random.random()
                inputs.append(self.mask_idx if prob < 0.2 else token)
                labels.append(token if prob < 0.2 else self.pad_idx)
        return inputs, labels