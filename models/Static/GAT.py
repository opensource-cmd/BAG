import torch
import torch.nn as nn
from torch.nn import BatchNorm1d
from torch_geometric.nn import GATConv
from .MLP import MLP

class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim:int, heads:int, 
                 dropout_rate:float, activation='ReLU', **kwargs):
        super().__init__()
        self.norm1 = BatchNorm1d(input_dim)
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout_rate)
        self.norm2 = BatchNorm1d(hidden_dim*heads)
        self.act = getattr(nn, activation)()
        self.gat2 = GATConv(hidden_dim*heads, hidden_dim, heads=heads, concat=False, dropout=dropout_rate)
        self.scorer = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim)
        
    def forward(self, x, edge_index):
        h = self.norm1(x)
        h = self.gat1(h, edge_index)
        h = self.norm2(h)
        h = self.act(h)
        h = self.gat2(h, edge_index)
        return h

    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits