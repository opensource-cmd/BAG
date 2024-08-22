import torch
import torch.nn as nn
from torch_geometric.nn import ChebConv
from .MLP import MLP

class ChebNet(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, num_layers:int, K:int, 
                 dropout_rate:float, activation='ReLU', **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(ChebConv(input_dim, hidden_dim, K=K))
        for _ in range(1, num_layers - 1):
            self.layers.append(ChebConv(hidden_dim, hidden_dim, K=K))
        self.layers.append(ChebConv(hidden_dim, hidden_dim, K=K))
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            h = layer(x, edge_index)
            if i < len(self.layers) - 1:  
                h = self.act(h)
                h = self.dropout(h)
        return h

    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits 