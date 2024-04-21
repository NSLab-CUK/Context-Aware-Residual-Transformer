import torch
import torch.nn as nn

class ALBert4Rec(nn.Module):
    def __init__(self, args, vocab_size):
        super(ALBert4Rec, self).__init__()

        self.args = args
        self.vocab_size = vocab_size

        self.embedding1 = nn.Embedding(vocab_size, args.embed_size, padding_idx=0)
        self.embedding2 = nn.Linear(args.embed_size, args.hidden_size)
        self.positional = nn.Embedding(15, args.hidden_size)

        self.norm = nn.LayerNorm(args.hidden_size)

        self.num_layers = args.num_layers
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_size,
            nhead=args.num_heads,
            dim_feedforward=args.hidden_size*4
        )

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, inputs):
        x = self.embedding2(self.embedding1(inputs['t_inputs'])) + self.positional(inputs['t_position'])
        x = self.dropout(x)

        mask = (inputs['t_inputs'] == 0).unsqueeze(1).expand(-1, 15, -1)
        mask = mask.repeat_interleave(self.args.num_heads, dim=0)

        x = self.norm(x)
        for _ in range(self.num_layers):
            x = torch.transpose(x, 0, 1)
            x = self.transformer_encoder_layer(x, mask)
            x = torch.transpose(x, 0, 1)
        x = self.norm(x)

        return x
