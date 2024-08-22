import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv
from .MLP import MLP

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim:int, num_layers:int, 
                 dropout_rate:int, activation='ReLU', **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = self.act(x)
                x = self.dropout(x)
        return x
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits