import torch
import torch.nn as nn
from torch_geometric.nn import SGConv
from .MLP import MLP

class SGC(nn.Module):
    def __init__(self, input_dim, hidden_dim:int, K:int, 
                 activation='ReLU', **kwargs):
        super().__init__()
        self.conv1 = SGConv(input_dim, hidden_dim, K=K, cached=True)
        self.act = getattr(nn, activation)()
        self.conv2 = SGConv(hidden_dim, hidden_dim, K=K, cached=True)
        self.scorer = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim)
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, pos_edge_index, neg_edge_index): 
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) 
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits   
    