import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
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
        # x: (T, B, D)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ConformerSmall(nn.Module):
    def __init__(self, in_channels=64, timesteps=250, d_model=256, nhead=8, nlayers=3, out_dim=1024):
        super().__init__()
        # project channels to d_model
        self.conv_sub = nn.Conv1d(in_channels, d_model, kernel_size=3, padding=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=512, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)
        self.pos = PositionalEncoding(d_model)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(d_model, out_dim))

    def forward(self, x):
        # x: (B, C, T)
        x = self.conv_sub(x)  # (B, D, T)
        # transformer expects (T, B, D)
        x = x.permute(2, 0, 1)
        x = self.pos(x)
        x = self.transformer(x)
        x = x.permute(1, 2, 0)  # (B, D, T)
        x = self.pool(x).squeeze(-1)
        out = self.fc(x)
        return out


def get_conformer(in_channels=64, timesteps=250, out_dim=1024):
    return ConformerSmall(in_channels=in_channels, timesteps=timesteps, out_dim=out_dim)
