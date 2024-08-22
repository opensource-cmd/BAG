import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from .MLP import MLP

class GIN(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, num_layers:int, 
                 dropout_rate:float, activation='ReLU', **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.act = getattr(nn, activation)()

        nn1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            self.act,
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.layers.append(GINConv(nn1))

        for _ in range(1, num_layers):
            nn_k = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                self.act,
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.layers.append(GINConv(nn_k))

        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.scorer = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim)
        
    def forward(self, x, edge_index, batch=None):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:  
                x = self.dropout(x)
        return x

    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits