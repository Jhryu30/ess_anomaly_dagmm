from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepSVDD(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, bidirectional=False, model_path=None):
        super().__init__()
        self.model_path = model_path
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

    def get_loss(self, output):
        return (output - self.center) ** 2

    def init(self, dataloader, num_cycle, device):
        outputs = []
        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                data = data.float().to(device)
                output = self(data) # (B, H)
                output = output.mean(axis=0) # (H, )
                outputs.append(output.cpu())
                if i == num_cycle:
                    break
            center = torch.mean(torch.stack(outputs), axis=0)
        torch.save(center, self.model_path / 'center.pt')
        self.center = center.to(device)

    def load(self, device):
        center = torch.load(self.model_path / 'center.pt')
        self.center = center.to(device)
