from torch import nn
import torch
import numpy as np
import torch.nn.functional as F

class Automix(nn.Module):

    def __init__(self, hidden_nf, device):
        super(Automix, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(hidden_nf * 3, hidden_nf),
            nn.SiLU(),
            nn.Linear(hidden_nf, hidden_nf)
        )
        self.ones = torch.ones(hidden_nf)
        self.device = device

    def get_weight(self, p1, p2, lamb):
        x1 = self.ones * lamb
        input = torch.cat([p1, p2, x1.to(self.device)])
        out = self.mlp(input)
        return out

    def forward(self, prompt1, prompt2, lamb):

        prompt_weight = self.get_weight(prompt1, prompt2, lamb)
        prompt_new = prompt_weight * prompt1 + (self.ones.to(self.device) - prompt_weight) * prompt2
        return prompt_new

    def get_similar(self):
        return None