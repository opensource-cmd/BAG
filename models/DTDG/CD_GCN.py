import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import LSTM
from .MLP import MLP

class CD_GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64, dropout_rate=0, activation='ReLU', **kwargs):
        super().__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.lstm = LSTM(hidden_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        ) 
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.scorer = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim)
        
    def forward(self, temporal_graphs):
        gcn_outputs = []

        for data in temporal_graphs:
            gcn_out = self.gcn(data.x, data.edge_index)
            gcn_out = self.act(gcn_out)
            gcn_out = self.dropout(gcn_out)
            gcn_outputs.append(gcn_out.unsqueeze(0))

        gcn_outputs = torch.cat(gcn_outputs, 0)
        lstm_out, _ = self.lstm(gcn_outputs)
        return lstm_out

    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits