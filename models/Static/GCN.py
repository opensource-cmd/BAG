import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from .MLP import MLP

class GCN(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, num_layers:int, 
                 dropout_rate:float, activation='ReLU', **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.scorer = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim)
        
    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            h = layer(x, edge_index)
            if i < len(self.layers) - 1:  
                h = self.act(h)
                h = self.dropout(h)
        return h

    def decode(self, z, pos_edge_index, neg_edge_index): 
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) 
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits