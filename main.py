from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support
import hydra
import wandb

from util import set_seed
from dataloader import ESSDataset
from models import DeepSVDD

def get_loss(output, center):
    return (output - center) ** 2

def get_init(model, dataloader, num_cycle, device):
    outputs = []
    with torch.no_grad():
        for i, (data, _) in enumerate(dataloader):
            data = data.float().to(device)
            output = model(data) # (B, H)
            output = output.mean(axis=0) # (H, )
            outputs.append(output.cpu())
            if i == num_cycle:
                break
        c = torch.mean(torch.stack(outputs), axis=0)
    return c

def train(cfg):
    model_path = Path(cfg.model.path) / cfg.data.name
    if not model_path.exists():
        model_path.mkdir(parents=True)

    device = torch.device(cfg.device)

    train_dataset = ESSDataset(cfg.data.path, cfg.data.name, cfg.model.window_size,
                               cfg.model.step, mode='train')
    valid_dataset = ESSDataset(cfg.data.path, cfg.data.name, cfg.model.window_size,
                               cfg.model.step, mode='val')

    train_loader = DataLoader(train_dataset, batch_size=cfg.model.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=cfg.model.batch_size, shuffle=False)

    model = DeepSVDD(input_dim=cfg.data.dim,
                     hidden_dim=cfg.model.hidden_dim, 
                     n_layers=cfg.model.n_layers, 
                     dropout=cfg.model.dropout, 
                     bidirectional=cfg.model.bidirectional)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.model.lr)

    center = get_init(model, train_loader, cfg.model.num_cycle, device)
    torch.save(center, model_path / 'center.pt')

    center = center.to(device)

    for epoch in range(cfg.train.epochs):
        train_losses, valid_losses = [], []
        model.train()
        for i, (data, _) in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.float().to(device)
            output = model(data)

            loss = get_loss(output, center).mean()
            train_losses.append(loss.item())

            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            for i, (data, target) in enumerate(valid_loader):
                data = data.float().to(device)
                output = model(data)
                loss = get_loss(output, center).mean()
                valid_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        valid_loss = np.mean(valid_losses)
        print(f"Epoch: {epoch+1} | train loss: {train_loss:.10f}, valid loss: {valid_loss:.10f}")

        torch.save(model.state_dict(), model_path / f'{epoch}.pth')

def test(cfg):
    epoch = cfg.train.epochs - 1

    model_path = Path(cfg.model.path) / cfg.data.name
    device = torch.device(cfg.device)
    
    test_dataset = ESSDataset(cfg.data.path, cfg.data.name, cfg.model.window_size,
                               cfg.model.step, mode='test')

    test_loader = DataLoader(test_dataset, batch_size=cfg.model.batch_size, shuffle=False)

    model = DeepSVDD(input_dim=cfg.data.dim,
                     hidden_dim=cfg.model.hidden_dim, 
                     n_layers=cfg.model.n_layers, 
                     dropout=cfg.model.dropout, 
                     bidirectional=cfg.model.bidirectional)
    model.to(device)

    center = torch.load(model_path / 'center.pt')
    center = center.to(device)

    model.load_state_dict(torch.load(model_path / f'{epoch}.pth'))
    model.eval()

    test_losses = []
    test_labels = []
    with torch.no_grad():
        for i, (data, labels) in enumerate(test_loader):
            data = data.float().to(device)
            output = model(data)
            loss = get_loss(output, center).mean(axis=-1)
            loss = loss.cpu().numpy()
            test_losses.append(loss)
            test_labels.append(labels)

    test_losses = np.concatenate(test_losses, axis=0).reshape(-1)
    test_labels = np.concatenate(test_labels, axis=0).reshape(-1)

    np.save(model_path / 'test_losses.npy', test_losses)

@hydra.main(config_path="./", config_name="config", version_base='1.2')
def main(cfg):
    set_seed(cfg.seed)

    wandb.init(
        project='ESSAnomaly',
        config=dict(cfg),
        mode='disabled'
    )

    if cfg.mode == 'train':
        train(cfg)
    else:
        test(cfg)

if __name__ == "__main__":
    main()