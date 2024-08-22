import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from .MLP import MLP

class WD_GCN(nn.Module):
    def __init__(self, input_dim, num_nodes, hidden_dim=64, dropout_rate=0.5, activation='ReLU', **kwargs):
        super().__init__()
        # Graph Convolutional Layer
        self.gcn = GCNConv(input_dim, hidden_dim)
        # LSTM Layer
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        ) 
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate)
        self.max_nodes = num_nodes
        self.scorer = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim)
        
    def forward(self, data_list):
        gcn_outputs = torch.stack([self.process_graph(data) for data in data_list])
        gcn_outputs = gcn_outputs.squeeze(0)
        lstm_out, _ = self.lstm(gcn_outputs)
        return lstm_out

    def process_graph(self, data):
        # GCN forward pass
        gcn_out = self.gcn(data.x, data.edge_index)
        gcn_out = self.act(gcn_out)
        gcn_out = self.dropout(gcn_out)
        return gcn_out  

    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits