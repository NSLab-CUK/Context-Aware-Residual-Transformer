import torch
import torch.nn as nn

class GRU4Rec(nn.Module):
    def __init__(self, args, vocab_size):
        super(GRU4Rec, self).__init__()

        self.args = args
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, args.hidden_size, padding_idx=0)

        self.norm = nn.LayerNorm(args.hidden_size)

        self.gru = nn.GRU(
            input_size=args.hidden_size,
            hidden_size=args.hidden_size,
            num_layers=args.num_layers
        )

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, inputs):
        x = self.embedding(inputs['t_inputs'])

        x = self.dropout(x)

        x = self.norm(x)
        x = torch.transpose(x, 0, 1)
        x, _ = self.gru(x)
        x = torch.transpose(x, 0, 1)
        x = self.norm(x)
        return x
