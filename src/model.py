
from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class MLPConfig:
    in_dim: int
    hidden: int = 256
    hidden2: int = 128
    dropout: float = 0.2
    out_dim: int = 10

class MLP(nn.Module):
    def __init__(self, cfg: MLPConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.in_dim, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden2),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden2, cfg.out_dim),
        )
    def forward(self, x):
        return self.net(x)
