import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import GRU
import math

class AddGraph(torch.nn.Module):
    def __init__(self, input_dim, num_nodes, hidden_dim=64, dropout_rate=0, w=3, activation='ReLU', **kwargs):
        super().__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.hca = HCA(hidden_dim, dropout_rate)
        self.gru = GRU(hidden_dim*2, hidden_dim)
        self.w = w
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        ) 
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self, temporal_graphs):
        h_list = []  # Start with an empty list
        for t, data in enumerate(temporal_graphs):
            x, edge_index = data.x, data.edge_index  
            current = self.gcn(x, edge_index)  # (n,h)
            current = current.unsqueeze(0)   # (1,n,h)

            if t == 0:
                C = torch.zeros(1, self.num_nodes, self.hidden_dim, device=self.device)  # Initial hidden state
            elif t < self.w:
                C = torch.cat([torch.zeros(1, self.num_nodes, self.hidden_dim, device=self.device)] + h_list[:t], dim=0)
            else:
                C = torch.cat(h_list[t-self.w:t], dim=0)

            short = self.hca(C)  # (1,n,h)
            h_t, _ = self.gru(torch.cat([current, short], dim=-1))  # (1,n,h)
            h_list.append(h_t)

        return h_list   
        
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits
        
class HCA(nn.Module):
    def __init__(self, hidden, dropout):
        super(HCA, self).__init__()
        self.hidden = hidden
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Q = nn.Parameter(torch.FloatTensor(hidden, hidden)).to(self.device)
        self.r = nn.Parameter(torch.FloatTensor(hidden)).to(self.device)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt((self.Q.size(0)))
        self.Q.data.uniform_(-stdv, stdv)
        stdv = 1. / math.sqrt(self.r.size(0))
        self.r.data.uniform_(-stdv, stdv)

    def forward(self, C):
        C_ = C.permute(1, 0, 2)
        C_t = C_.permute(0, 2, 1)
        e_ = torch.einsum('ih,nhw->niw', self.Q, C_t)
        e_ = F.dropout(e_, self.dropout, training=self.training)
        e = torch.einsum('h,nhw->nw', self.r, torch.tanh(e_))
        e = F.dropout(e, self.dropout, training=self.training)
        a = F.softmax(e, dim=1)
        short = torch.einsum('nw,nwh->nh', a, C_)
        short = short.unsqueeze(0)
        return short
        