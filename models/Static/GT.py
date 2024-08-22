import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv
from .MLP import MLP

class GT(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, num_layers:int, 
                 heads:int, dropout_rate:float, activation='ReLU', **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TransformerConv(input_dim, hidden_dim, heads=heads, 
                                           dropout=dropout_rate, concat=True))
        for _ in range(1, num_layers - 1):
            self.layers.append(TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, 
                                               dropout=dropout_rate, concat=True))
        self.layers.append(TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, 
                                           dropout=dropout_rate, concat=False))
        
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.scorer = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim)

    def forward(self, x, edge_index):
        for layer in self.layers[:-1]:
            x = layer(x, edge_index)
            x = self.act(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index)
        return x
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits