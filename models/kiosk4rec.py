import torch
import torch.nn as nn

class Kiosk4Rec(nn.Module):
    def __init__(self, args, vocab_size):
        super().__init__()

        self.args = args
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, args.embed_size, padding_idx=0)
        self.position_emb = nn.Embedding(15, args.hidden_size)

        self.user_emb = nn.Linear(args.embed_size, args.hidden_size)
        self.context_emb = nn.Linear(args.embed_size, args.hidden_size)
        self.domain_emb = nn.Linear(args.embed_size, args.hidden_size)
        self.item_emb = nn.Linear(args.embed_size, args.hidden_size)

        self.norm = nn.LayerNorm(args.hidden_size)

        self.num_layers = args.num_layers
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=args.hidden_size,
            nhead=args.num_heads,
            dim_feedforward=args.hidden_size*4
        )
        self.residual_h = nn.Linear(args.hidden_size, args.hidden_size)

        self.dropout = nn.Dropout(p=args.dropout)

    def forward(self, inputs):
        user = self.user_emb(self.tok_emb(inputs['t_inputs'][:, :1]))
        context = self.context_emb(self.tok_emb(inputs['t_inputs'][:, 1:4]))
        domain = self.domain_emb(self.tok_emb(inputs['t_inputs'][:, 4:5]))
        item = self.item_emb(self.tok_emb(inputs['t_inputs'][:, 5:])) + self.position_emb(inputs['t_position'])

        x = torch.cat([user, context, domain, item], dim=1)

        x = self.dropout(x)

        mask = (inputs['t_inputs'] == 0).unsqueeze(1).expand(-1, 20, -1)
        mask = mask.repeat_interleave(self.args.num_heads, dim=0)

        x = self.norm(x)
        for _ in range(self.num_layers):
            x = torch.transpose(x, 0, 1)
            x = self.transformer_encoder_layer(x, mask)
            x = torch.transpose(x, 0, 1)
            x = self.norm(torch.cat([x[:, :5], x[:, 5:] + self.residual_h(item)], dim=1))
        return x
