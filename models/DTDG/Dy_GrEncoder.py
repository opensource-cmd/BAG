import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import  DyGrEncoder
from .MLP import MLP


class Dy_GrEncoder(torch.nn.Module):
    def __init__(self, input_dim:int, hidden_dim = 64, **kwargs):
        super().__init__()
        self.recurrent = DyGrEncoder(hidden_dim, 1, 'mean', hidden_dim, 1)
        self.linear = torch.nn.Linear(hidden_dim, hidden_dim)
        self.scorer = MLP(input_dim=input_dim, hidden_dim=input_dim) 

    def forward(self, x, edge_index):
        H_tilde, H, C = self.recurrent(x, edge_index)
        h = F.relu(H_tilde) 
        h = self.linear(h)       
        return h
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits   