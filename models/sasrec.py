import torch
import torch.nn as nn

class SASRec(nn.Module):
    def __init__(self, args, vocab_size):
        super(SASRec, self).__init__()

        self.args = args
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, args.hidden_size, padding_idx=0)
        self.positional = nn.Embedding(15, args.hidden_size)

        self.norm = nn.LayerNorm(args.hidden_size)

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=args.hidden_size,
                nhead=args.num_heads,
                dim_feedforward=args.hidden_size * 4
            ),
            num_layers=args.num_layers
        )

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, inputs):
        x = self.embedding(inputs['t_inputs']) + self.positional(inputs['t_position'])
        x = self.dropout(x)

        mask = torch.tril(torch.ones(inputs['t_inputs'].size(1), inputs['t_inputs'].size(1)) == 1)
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))

        x = self.norm(x)
        x = torch.transpose(x, 0, 1)
        x = self.transformer_encoder(x, mask=mask.to(int(self.args.device)))
        x = torch.transpose(x, 0, 1)
        x = self.norm(x)

        return x
