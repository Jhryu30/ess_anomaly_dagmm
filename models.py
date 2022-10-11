from unicodedata import bidirectional
import torch
import torch.nn as nn
import torch.nn.functional as F

from dagmm_get_loss import ComputeLoss

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






class DAGMM(nn.Module):
    def __init__(self, window_size, input_dim, hidden_dim, n_gmm=2, z_dim=1, lambda_energy=0.01, lambda_cov=1e-5, device=None):
        super(DAGMM, self).__init__()
        self.device = device
        self.lambda_energy = lambda_energy
        self.lambda_cov = lambda_cov
        self.n_gmm = n_gmm
        #squeeze raw data
        self.fc0 = nn.Linear(input_dim, 1)

        #Encoder network
        self.fc1 = nn.Linear(window_size, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, hidden_dim//4)
        self.fc3 = nn.Linear(hidden_dim//4, hidden_dim//8)
        self.fc4 = nn.Linear(hidden_dim//8, z_dim)

        #Decoder network
        self.fc5 = nn.Linear(z_dim, hidden_dim//8)
        self.fc6 = nn.Linear(hidden_dim//8, hidden_dim//4)
        self.fc7 = nn.Linear(hidden_dim//4, hidden_dim//2)
        self.fc8 = nn.Linear(hidden_dim//2, window_size)

        #Estimation network
        self.fc9 = nn.Linear(z_dim+2, 10)
        self.fc10 = nn.Linear(10, self.n_gmm)

        #unsqueeze
        self.fc11 = nn.Linear(1, input_dim)

    def encode(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))
        return self.fc4(h)

    def decode(self, x):
        h = torch.tanh(self.fc5(x))
        h = torch.tanh(self.fc6(h))
        h = torch.tanh(self.fc7(h))
        return self.fc8(h)
    
    def estimate(self, z):
        h = F.dropout(torch.tanh(self.fc9(z)), 0.5)
        return F.softmax(self.fc10(h), dim=1)
    
    def compute_reconstruction(self, x_ori, x_hat_ori):
        # relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        # cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        relative_euclidean_distance = (x_ori-x_hat_ori).norm(2, dim=(1,2)) / x_ori.norm(2, dim=(1,2))
        cosine_similarity = F.cosine_similarity(x_ori, x_hat_ori, dim=-1).mean(1)
        return relative_euclidean_distance, cosine_similarity
    
    def forward(self, x):
        x_ori = x.clone()
        x = self.fc0(x)
        z_c = self.encode(x.squeeze())
        x_hat = self.decode(z_c)
        x_hat_ori = self.fc11(x_hat.unsqueeze(dim=2))
        rec_1, rec_2 = self.compute_reconstruction(x_ori, x_hat_ori)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z)
        return x_ori, x_hat_ori, z, gamma
    
    def get_loss(self, output):
        compute = ComputeLoss(lambda_energy=self.lambda_energy, lambda_cov=self.lambda_cov, device=self.device, n_gmm=self.n_gmm)
        x_ori, x_hat_ori, z, gamma = output
        loss = compute.forward(x_ori, x_hat_ori, z, gamma)
        return loss

    def get_score(self, output):
        compute = ComputeLoss(lambda_energy=self.lambda_energy, lambda_cov=self.lambda_cov, device=self.device, n_gmm=self.n_gmm)
        _, _, z, gamma = output
        sample_energy, _ = compute.compute_energy(z, gamma, sample_mean=False)
        return sample_energy