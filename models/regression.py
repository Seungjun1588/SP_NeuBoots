import torch
from torch import nn
import torch.nn.functional as F



class Reg_model(nn.Module):
    def __init__(self, in_feat=400):
        super().__init__()
        self.in_feat = in_feat
        self.features = nn.Sequential(
            nn.Linear(in_feat,in_feat*2),
            nn.ReLU(inplace=True),  
            nn.Linear(in_feat*2,in_feat),
            nn.ReLU(inplace=True),
            nn.Linear(in_feat,in_feat),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        out1 = self.features(x)

        return out1
    
class Reg_model2(nn.Module):
    def __init__(self, in_feat=400):
        super().__init__()
        self.in_feat = in_feat
        self.features = nn.Sequential(
            nn.Linear(in_feat,10),
            nn.ReLU(inplace=True),  
            nn.Linear(10,5),
            nn.ReLU(inplace=True),
            nn.Linear(5,in_feat),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):
        out1 = self.features(x)

        return out1