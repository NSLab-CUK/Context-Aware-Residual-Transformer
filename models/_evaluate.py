import torch
import torch.nn as nn

class Evaluate4rec(nn.Module):
    def __init__(self, base_model):
        super().__init__()

        self.base_model = base_model
        self.evaluater = nn.Linear(base_model.args.hidden_size, base_model.vocab_size)

    def forward(self, inputs):
        x = self.base_model(inputs)
        x = self.evaluater(x)
        return x

class Pretrain4rec(nn.Module):
    def __init__(self, base_model):
        super().__init__()

        self.base_model1 = base_model
        self.base_model2 = base_model

        self.same_user = nn.Linear(base_model.args.hidden_size*2, 2)
        self.same_domain = nn.Linear(base_model.args.hidden_size*2, 2)
        self.evaluater = nn.Linear(base_model.args.hidden_size, base_model.vocab_size)

    def forward(self, inputs):
        a = self.base_model1({"t_inputs": inputs['a_inputs'], "t_position": inputs['t_position']})
        b = self.base_model2({"t_inputs": inputs['b_inputs'], "t_position": inputs['t_position']})
        return self.same_user(torch.cat((a[:, 0], b[:, 0]), dim=1)), self.same_domain(torch.cat((a[:, 4], b[:, 4]), dim=1)),\
               self.evaluater(a), self.evaluater(b)

