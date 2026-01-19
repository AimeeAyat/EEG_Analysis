import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
from eeg_dataset import EEGClipDataset
from conformer_small import get_conformer


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    ds = EEGClipDataset(root=root, sub='sub01')
    dl = DataLoader(ds, batch_size=32, shuffle=True, collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    clip_dim = ds[0][1].numel()
    model = get_conformer(in_channels=64, out_dim=clip_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    model.train()
    for i, (x, y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)
        preds = model(x)
        loss = loss_fn(preds, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print('Batch', i, 'MSE loss', loss.item())
        if i >= 2:
            break


if __name__ == '__main__':
    main()
