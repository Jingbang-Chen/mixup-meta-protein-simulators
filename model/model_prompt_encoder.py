from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
class prompt_encoder(nn.Module):

    def __init__(self, hidden_dim, act_fn=nn.SiLU()):
        super(prompt_encoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h):
        return self.mlp(h)