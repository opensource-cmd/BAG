import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import MPNNLSTM
from .MLP import MLP


class MPNN_LSTM(torch.nn.Module):   
    def __init__(self, input_dim: int, num_nodes:int, hidden_dim=64, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.recurrent = MPNNLSTM(input_dim, hidden_dim, num_nodes, 1, 0.1)
        self.linear = torch.nn.Linear(self.hidden_dim*2 + self.input_dim, self.hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(self.hidden_dim*2 + self.input_dim, 1, bias=True),
        ) 
        
    def forward(self, x, edge_index, edge_weight=None) -> torch.FloatTensor:
        h = self.recurrent(x, edge_index, edge_weight)
        # h = F.relu(h)
        # h = self.linear(h)
        return h
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits  