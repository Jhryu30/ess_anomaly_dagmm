from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSVDD(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=n_layers, bias=False, 
                            dropout=dropout, bidirectional=bidirectional, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x