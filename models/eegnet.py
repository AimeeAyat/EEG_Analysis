import torch
import torch.nn as nn


class EEGNet(nn.Module):
    # A compact EEGNet-like architecture adapted for (C, T) inputs
    def __init__(self, in_channels=64, timesteps=250, out_dim=1024, dropout=0.5):
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.MaxPool1d(2),
        )
        self.spatial = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.AdaptiveAvgPool1d(32),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 32, 1024),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, out_dim),
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.classifier(x)
        return x


def get_model(in_channels=64, timesteps=250, out_dim=1024):
    return EEGNet(in_channels=in_channels, timesteps=timesteps, out_dim=out_dim)
