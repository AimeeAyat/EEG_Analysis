import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EEGConformer(nn.Module):
    """A larger Conformer-like encoder for EEG (suitable for contrastive training).

    Input: (B, C, T)
    Output: (B, out_dim)
    """

    def __init__(self, in_channels=64, timesteps=250, d_model=512, nhead=8, nlayers=6, out_dim=1024, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=False)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pos = PositionalEncoding(d_model, dropout=dropout, max_len=timesteps)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(d_model, out_dim))

    def forward(self, x):
        # x: (B, C, T)
        x = self.input_proj(x)  # (B, D, T)
        x = x.permute(2, 0, 1)  # (T, B, D)
        x = self.pos(x)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # (B, D, T)
        x = self.pool(x).squeeze(-1)
        out = self.fc(x)
        return out


def get_eeg_conformer(in_channels=64, timesteps=250, out_dim=1024):
    return EEGConformer(in_channels=in_channels, timesteps=timesteps, d_model=512, nhead=8, nlayers=6, out_dim=out_dim)
import torch
import torch.nn as nn


class SimpleConformer(nn.Module):
    def __init__(self, in_channels=64, timesteps=250, d_model=256, nhead=8, num_layers=3, out_dim=1024):
        super().__init__()
        # initial conv to project channels -> d_model
        self.conv = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model, out_dim)
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv(x)  # (B, d_model, T)
        # transformer expects (T, B, d_model)
        x = x.permute(2, 0, 1)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # (B, d_model, T)
        x = self.pool(x)  # (B, d_model, 1)
        out = self.fc(x)
        return out


def get_conformer(in_channels=64, timesteps=250, out_dim=1024):
    return SimpleConformer(in_channels=in_channels, timesteps=timesteps, out_dim=out_dim)
