import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNH
from .MLP import MLP

class EvolveGCN_H(torch.nn.Module):    
    def __init__(self, input_dim: int, num_nodes:int, **kwargs):
        super().__init__()
        self.recurrent = EvolveGCNH(num_nodes, input_dim)
        self.linear = nn.Linear(input_dim, input_dim)
        self.scorer = MLP(input_dim=input_dim, hidden_dim=input_dim)
             
    def forward(self, x, edge_index) -> torch.FloatTensor:
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits