import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from .MLP import MLP


class EvolveGCN_O(torch.nn.Module):
    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.recurrent = EvolveGCNO(input_dim)
        self.scorer = MLP(input_dim=input_dim, hidden_dim=input_dim)
           
    def forward(self, x, edge_index) -> torch.FloatTensor:
        h = self.recurrent(x, edge_index)
        # h = F.relu(h)
        return h
    
    def decode(self, z, pos_edge_index, neg_edge_index): 
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) 
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits
