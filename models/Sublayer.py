import torch.nn as nn

class Sublayer(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(Sublayer, self).__init__()
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, res_x):
        if res_x is None:
            return self.dropout(self.norm(x))
        return res_x + self.dropout(self.norm(x))
