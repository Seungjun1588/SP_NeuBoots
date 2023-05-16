import torch
from torch import nn
import torch.nn.functional as F



class Reg_model(nn.Module):
    def __init__(self, in_feat):
        super().__init__()
        self.in_feat = in_feat
        self.h1 = nn.Linear(in_feat,in_feat*2)
        self.h2 = nn.Linear(in_feat*2,in_feat)
        self.h3 = nn.Linear(in_feat,in_feat)
        self.activation = nn.ReLU()

    def forward(self, x, alpha):
        out1 = self.activation(self.h1(x))
        out2 = self.activation(self.h2(out1))
        out3 = self.activaton(self.h3(out2))
        # out4 = self.activaton(self.h3(out3))

        # out5 = torch.exp(-F.interpolate(alpha[:, None], self.in_feat))[:, 0]
        return out3
    
