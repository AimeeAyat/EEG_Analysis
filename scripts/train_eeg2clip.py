import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from eeg_dataset import EEGClipDataset


class SimpleEEGEncoder(nn.Module):
    def __init__(self, in_channels=64, timesteps=250, clip_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32, 1024),
            nn.ReLU(),
            nn.Linear(1024, clip_dim)
        )

    def forward(self, x):
        # x: (B, C, T)
        out = self.conv(x)
        out = self.fc(out)
        return out


def collate_fn(batch):
    xs = torch.stack([b[0] for b in batch], dim=0)
    ys = torch.stack([b[1] for b in batch], dim=0)
    return xs, ys


def main():
    root = os.path.dirname(os.path.dirname(__file__))
    ds = EEGClipDataset(root=root, sub='sub01')
    # infer clip dim from dataset target
    clip_dim = ds[0][1].numel()
    print('Dataset trials:', len(ds), 'clip dim:', clip_dim)
    dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate_fn)

    model = SimpleEEGEncoder(in_channels=64, timesteps=250, clip_dim=clip_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # single epoch quick run
    model.train()
    for i, (x, y) in enumerate(dl):
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f'Batch {i} loss {loss.item():.6f}')
        if i >= 2:
            break


if __name__ == '__main__':
    main()
