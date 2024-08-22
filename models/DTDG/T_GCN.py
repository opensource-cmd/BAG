import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import TGCN
from .MLP import MLP


class T_GCN(torch.nn.Module): 
    def __init__(self, input_dim: int, hidden_dim=64, **kwargs):
        super().__init__()
        self.recurrent1 = TGCN(input_dim, hidden_dim)
        self.recurrent2 = TGCN(hidden_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        ) 
        self.scorer = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim)
        
    def forward(self, x, edge_index) -> torch.FloatTensor:
        h = self.recurrent1(x, edge_index)
        h = F.relu(h)
        h = self.recurrent2(h, edge_index)
        return h
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits   