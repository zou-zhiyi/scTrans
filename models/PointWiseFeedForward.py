import torch.nn as nn


class PointWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ffw, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(in_features=d_model, out_features=d_ffw)
        self.linear2 = nn.Linear(in_features=d_ffw, out_features=d_model)
        self.a = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        # self.dropout2 = nn.Dropout(dropout_rate)
        # nn.init.xavier_uniform_(self.linear1.weight)
        # nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, x):
        return self.linear2(self.a(self.dropout1(self.linear1(x))))
        # return self.dropout2(self.linear2(self.a(self.dropout1(self.linear1(x)))))
