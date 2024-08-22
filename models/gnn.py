# Importing standard torch packages
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LSTM, GRU, Linear, BatchNorm1d
from torch.nn.parameter import Parameter
from torch_sparse import SparseTensor
from torch_geometric.data import Data
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.nn import GCNConv, SAGEConv, GINConv, GATConv, ChebConv, SGConv, TransformerConv, Node2Vec, global_mean_pool, MessagePassing, global_sort_pool, InnerProductDecoder
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.norm.batch_norm import BatchNorm
from torch_geometric.utils import  degree, to_networkx, k_hop_subgraph, get_laplacian, to_scipy_sparse_matrix, negative_sampling, remove_self_loops, add_self_loops 
from torch_geometric_temporal.nn.recurrent import MPNNLSTM, GConvGRU, DCRNN,GCLSTM, DyGrEncoder, EvolveGCNO, EvolveGCNH, LRGCN, TGCN,TGCN2,A3TGCN, AGCRN
from transformers.models.bert.modeling_bert import BertAttention, BertIntermediate, BertOutput, BertPreTrainedModel, BertPooler
import numpy as np
import networkx as nx
from node2vec import Node2Vec
from typing import Optional, Tuple
import math

from torch_scatter import scatter_add
from copy import deepcopy
from scipy.sparse.linalg import eigs, eigsh

class GCNLayer(nn.Module):

    def __init__(self, dim_in, dim_out, pos_isn):
        super(GCNLayer, self).__init__()
        self.pos_isn = pos_isn
        self.linear_skip_weight = nn.Parameter(torch.ones(size=(dim_out, dim_in)))
        self.linear_skip_bias = nn.Parameter(torch.ones(size=(dim_out, )))

        self.linear_msg_weight = nn.Parameter(torch.ones(size=(dim_out, dim_in)))
        self.linear_msg_bias = nn.Parameter(torch.ones(size=(dim_out, )))

        self.activate = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.linear_skip_weight, gain=gain)
        nn.init.xavier_normal_(self.linear_msg_weight, gain=gain)

        nn.init.constant_(self.linear_skip_bias, 0)
        nn.init.constant_(self.linear_msg_bias, 0)

    def norm(self, graph):
        edge_index = graph.edges()
        row = edge_index[0]
        edge_weight = torch.ones((row.size(0),),
                                 device=row.device)

        deg = scatter_add(edge_weight, row, dim=0, dim_size=graph.num_nodes())
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt

    def message_fun(self, edges):
        return {'m': edges.src['h'] * 0.1}

    def forward(self, g, feats):
        
        linear_skip_weight = self.linear_skip_weight
        linear_skip_bias = self.linear_skip_bias
        linear_msg_weight = self.linear_msg_weight
        linear_msg_bias = self.linear_msg_bias

        feat_src, feat_dst = expand_as_pair(feats, g)
        norm_ = self.norm(g)
        feat_src = feat_src * norm_.view(-1, 1)
        g.srcdata['h'] = feat_src
        aggregate_fn = fn.copy_u('h', 'm')

        g.update_all(message_func=aggregate_fn, reduce_func=fn.sum(msg='m', out='h'))
        rst = g.dstdata['h']
        rst = rst * norm_.view(-1, 1)

        rst_ = F.linear(rst, linear_msg_weight, linear_msg_bias)
        skip_x = F.linear(feats, linear_skip_weight, linear_skip_bias)

        return rst_ + skip_x


class WinGNN(nn.Module):
    def __init__(self, in_features:int, out_features=64, hidden_dim=64, num_layers=2, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        for i in range(num_layers):
            d_in = in_features if i == 0 else out_features
            pos_isn = True if i == 0 else False
            layer = GCNLayer(d_in, out_features, pos_isn)
            self.add_module('layer{}'.format(i), layer)

        self.weight1 = nn.Parameter(torch.ones(size=(hidden_dim, out_features)))
        self.weight2 = nn.Parameter(torch.ones(size=(1, hidden_dim)))

        self.decode_module = nn.CosineSimilarity(dim=-1)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.weight1, gain=gain)
        nn.init.xavier_normal_(self.weight2, gain=gain)

    def forward(self, g, x, fast_weights=None):
        count = 0
        for layer in self.children():
            x = layer(g, x, fast_weights[2 + count * 4: 2 + (count + 1) * 4])
            count += 1

        if fast_weights:
            weight1 = fast_weights[0]
            weight2 = fast_weights[1]
        else:
            weight1 = self.weight1
            weight2 = self.weight2

        x = F.normalize(x)
        g.node_embedding = x

        pred = F.dropout(x, self.dropout)
        pred = F.relu(F.linear(pred, weight1))
        pred = F.dropout(pred, self.dropout)
        pred = F.sigmoid(F.linear(pred, weight2))

        node_feat = pred[g.edge_label_index]
        nodes_first = node_feat[0]
        nodes_second = node_feat[1]

        pred = self.decode_module(nodes_first, nodes_second)

        return pred
    
# Traditional Methods
# class Node2vec(nn.Module):
#     def __init__(self, embedding_dim=128,  walk_length=20, context_size=10, walks_per_node=10,
#         num_negative_samples=1, p=1.0, q=1.0, sparse=True, **kwargs):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         self.layers.append(GCNConv(input_dim, hidden_dim))
#         for _ in range(1, num_layers):
#             self.layers.append(GCNConv(hidden_dim, hidden_dim))
#         self.act = getattr(nn, activation)()
#         self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

#     def forward(self, x, edge_index):
#         for i, layer in enumerate(self.layers):
#             x = layer(x, edge_index)
#             if i < len(self.layers) - 1:  
#                 x = self.act(x)
#                 x = self.dropout(x)
#         return x

#     def decode(self, z, pos_edge_index, neg_edge_index): 
#         edge_index = torch.cat([pos_edge_index,neg_edge_index], dim=-1) 
#         return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1) 

    
# class Node2vec():
#     def __init__(self, dimensions=32, walk_length=30, num_walks=200, 
#                  workers=4, window=10, min_count=1, batch_words=4, **kwargs):
#         self.dimensions = dimensions
#         self.walk_length = walk_length
#         self.num_walks = num_walks
#         self.workers = workers
#         self.window = window
#         self.min_count = min_count
#         self.batch_words = batch_words   
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
         
#     def fit(self, data: Data):
#         graph = to_networkx(data, to_undirected=True)
#         model = Node2Vec(graph, dimensions=self.dimensions, walk_length=self.walk_length)
#         x = torch.tensor(model.wv.vectors, dtype=torch.float32).to(self.device)
#         return x
      
# Static GNNS
class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout_rate=0,
                 activation='ReLU', **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        ) 
        
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

class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout_rate=0, 
                 activation='ReLU', **kwargs):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(input_dim, hidden_dim))
        for _ in range(1, num_layers):
            self.layers.append(SAGEConv(hidden_dim, hidden_dim))
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        ) 
        
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
    

class GIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128, num_layers=2, dropout_rate=0.0, activation='ReLU', **kwargs):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        self.layers.append(GINConv(
            torch.nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                getattr(nn, activation)(),
                nn.Linear(hidden_dim, hidden_dim)
            )
        ))

        for _ in range(1, num_layers - 1):
            self.layers.append(GINConv(
                torch.nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    getattr(nn, activation)(),
                    nn.Linear(hidden_dim, hidden_dim)
                )
            ))

        self.layers.append(GINConv(
            torch.nn.Sequential(
                nn.Linear(hidden_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                getattr(nn, activation)(),
                nn.Linear(output_dim, output_dim)
            )
        ))
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        ) 
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)  
                x = self.dropout(x)  
        return x
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits
    
class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, heads=8, activation='ReLU', **kwargs):
        super().__init__()
        self.norm1 = BatchNorm1d(input_dim)
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads,
                              dropout=0.3)
        self.norm2 = BatchNorm1d(hidden_dim*heads)
        self.act = getattr(nn, activation)()
        self.gat2 = GATConv(hidden_dim*heads, hidden_dim, heads=heads,
                              concat=False, dropout=0.6)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        ) 
        
    def forward(self, x, edge_index):
        h = self.norm1(x)
        x = self.gat1(x, edge_index)
        h = self.norm2(h)
        x = self.act(x)
        x = self.gat2(x, edge_index)
        return x
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits 
    
class ChebNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, output_dim=64, num_layers=3, K=3, dropout_rate=0.5, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()

        self.layers.append(ChebConv(input_dim, hidden_dim, K=K))
        for _ in range(1, num_layers - 1):
            self.layers.append(ChebConv(hidden_dim, hidden_dim, K=K))
        self.layers.append(ChebConv(hidden_dim, output_dim, K=K))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        ) 
        
    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i != len(self.layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)
        return x
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits        
    
class SGC(nn.Module):
    def __init__(self, input_dim, hidden_dim=128,  K=2, activation='ReLU', **kwargs):
        super().__init__()
        self.conv1 = SGConv(input_dim, hidden_dim, K=K, cached=True)
        self.act = getattr(nn, activation)()
        self.conv2 = SGConv(hidden_dim, hidden_dim, K=K, cached=True)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        ) 
        
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.act(x)
        x = self.conv2(x, edge_index)
        return x

    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits   
    
class GT(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, output_dim=128, num_layers=2, heads=4, dropout=0.1, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(TransformerConv(input_dim, hidden_dim, heads=heads, dropout=dropout, concat=True))

        for _ in range(1, num_layers - 1):
            self.layers.append(TransformerConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True))
        self.layers.append(TransformerConv(hidden_dim * heads, output_dim, heads=heads, dropout=dropout, concat=False))
        
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        ) 
        
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
              
# Specialized GNNs

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
        
    
class StrGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout_rate=0, activation='ReLU', k=2, **kwargs):
        super().__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.GRU = GRU(hidden_dim, hidden_dim)
    
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()
        
    def create_subgraph(self, u, v, edge_index):
        # u = snapshot.edge_index[0, i]
        # v = snapshot.edge_index[1, i]
        subset1, subedge_index1, mapping, edge_mask = k_hop_subgraph(u, self.k, edge_index, relabel_nodes=True)
        subset2, subedge_index2, mapping, edge_mask = k_hop_subgraph(v, self.k, edge_index, relabel_nodes=True)
        combined_tensor = torch.cat((subedge_index1, subedge_index2), dim=1)
        transposed_tensor = combined_tensor.t()
        unique_tensor, indices = torch.unique(transposed_tensor, dim=0, return_inverse=True)
        unique_edge_index = unique_tensor.t() ##删除重复边
        
        unique_nodes = torch.unique(unique_edge_index).tolist()
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_nodes)}
        new_edge_index = torch.tensor([[node_mapping[node.item()] for node in edge_list] for edge_list in unique_edge_index])  ##重新label
        degree_count = degree(new_edge_index[1], num_nodes=new_edge_index.max().item() + 1)

        node_features = degree_count.view(-1, 1)
        subgraph = Data(x=node_features.float(), edge_index=new_edge_index)
            
        return subgraph    
             
    def forward(self, temporal_graphs):
        gcn_outputs = []
        s = temporal_graphs[-1]
        
        hidden_list=[]
        for i in range(s.edge_index.shape[1]):
            subgraphs = []
            for data in temporal_graphs:
                u = s.edge_index[0, i]
                v = s.edge_index[1, i]
                subgraph = self.create_subgraph(u, v, data.edge_index)
                subgraphs.append(subgraph)
                
            combined_x = torch.cat([sg.x for sg in subgraphs], dim=0)
            batch = torch.cat([torch.full((sg.x.size(0),), i, dtype=torch.long) for i, sg in enumerate(subgraphs)])
            sorted_pool = global_sort_pool(combined_x, batch, k=3)
            
            gcn_outputs = []
            for j, graph in enumerate(subgraphs):
                k = 3
                num_features = graph.x.size(1)  # 每个节点的特征维度 (1)
                total_features = k * num_features
                pooled_features = sorted_pool[j, :total_features].view(k, num_features)
                graph.x = pooled_features
            
                gcn_out = self.gcn(graph.x, graph.edge_index)
                gcn_out = self.act(gcn_out)
                gcn_out = self.dropout(gcn_out)
                
                gcn_outputs.append(gcn_out.unsqueeze(0))
                
            gcn_outputs = torch.cat(gcn_outputs, 0)   ## (T, N , F)
            output, hn = self.GRU(gcn_outputs)
            hidden_list.append(hn.squeeze(0))
        
        return hidden_list
    
    def decode(self, z, pos_edge_index, neg_edge_index): 
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1) 
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        score = self.scorer(edge_embs)
        return score.squeeze()     


# Discrete-time temporal GNNs

class Snapcat(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout_rate=0, activation='ReLU', **kwargs):
        super().__init__()
        self.gcn = GCNConv(input_dim, hidden_dim)
        self.lstm = LSTM(hidden_dim, hidden_dim)
    
        self.act = getattr(nn, activation)()
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, temporal_graphs):
        gcn_outputs = []

        for data in temporal_graphs:
            gcn_out = self.gcn(data.x, data.edge_index)
            gcn_out = self.act(gcn_out)
            gcn_out = self.dropout(gcn_out)
            gcn_outputs.append(gcn_out.unsqueeze(0))

        gcn_outputs = torch.cat(gcn_outputs, 0)
        lstm_out, _ = self.lstm(gcn_outputs)
        
        return lstm_out[-1]
    
    
class Dy_GrEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim = 64, **kwargs):
        super().__init__()
        self.recurrent = DyGrEncoder(hidden_dim, 1, 'mean', hidden_dim, 1)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        ) 

    def forward(self, x, edge_index):
        H_tilde, H, C = self.recurrent(x, edge_index)
        h = F.relu(H_tilde)        
        return h
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits     
    
class EvolveGCN_H(torch.nn.Module):    
    def __init__(self, input_dim: int, num_nodes:int, **kwargs):
        super().__init__()
        self.recurrent = EvolveGCNH(num_nodes, input_dim)
        self.linear = nn.Linear(input_dim, input_dim)
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, 1, bias=True),
        )   
             
    def forward(self, x, edge_index) -> torch.FloatTensor:
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits  

class EvolveGCN_O(torch.nn.Module):
    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.recurrent = EvolveGCNO(input_dim)
        self.scorer = nn.Sequential(
            nn.Linear(input_dim, 1, bias=True),
        ) 
        
    def forward(self, x, edge_index) -> torch.FloatTensor:
        h = self.recurrent(x, edge_index)
        # h = F.relu(h)
        return h
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits   


class MPNN_LSTM(torch.nn.Module):    ## epoch 70后下降
    def __init__(self, input_dim: int, num_nodes:int, hidden_dim=64, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.recurrent = MPNNLSTM(input_dim, hidden_dim, num_nodes, 1, 0.1)
        self.linear = torch.nn.Linear(self.hidden_dim*2 + self.input_dim, self.hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(self.hidden_dim*2 + self.input_dim, 1, bias=True),
        ) 
        
    def forward(self, x, edge_index, edge_weight=None) -> torch.FloatTensor:
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        # h = self.linear(h)
        return h
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits  

class T_GCN(torch.nn.Module): 
    def __init__(self, input_dim: int, hidden_dim=64, **kwargs):
        super().__init__()
        self.recurrent1 = TGCN(input_dim, hidden_dim)
        self.recurrent2 = TGCN(hidden_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
        ) 
        
    def forward(self, x, edge_index) -> torch.FloatTensor:
        h = self.recurrent1(x, edge_index)
        h = F.relu(h)
        h = self.recurrent2(h, edge_index)
        return h
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits        
    
    
#### Still need to be optimized
class T_GCN2(torch.nn.Module):  
    def __init__(self, input_dim: int, hidden_dim=32, **kwargs):
        super().__init__()
        self.recurrent = TGCN2(input_dim, hidden_dim, 0 )
        self.MLP = MLP(32)
    def forward(self, x, edge_index) -> torch.FloatTensor:
        h = self.recurrent(x, edge_index)
        # h = F.relu(h)
        h = self.MLP(h, edge_index)
        return h

class A3T_GCN(torch.nn.Module):   
    def __init__(self, input_dim: int, hidden_dim=32, periods=4, **kwargs):
        super().__init__()
        self.recurrent = A3TGCN(input_dim, hidden_dim, periods)

    def forward(self, x, edge_index) -> torch.FloatTensor:
        h = self.recurrent(x.view(x.shape[0], 1, x.shape[1]), edge_index)
        h = F.relu(h)
        # h = self.MLP(h, edge_index)
        return h

class MLP(nn.Module):
    def __init__(self, hidden_dim, with_dropout=False, **kwargs):
        super().__init__()
        
        self.h1_weights = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.with_dropout = with_dropout

        nn.init.xavier_uniform_(self.h1_weights.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, x, edge_index):
        
        head_embs = x[edge_index[0]]  
        tail_embs = x[edge_index[1]]  
        edge_embs = head_embs * tail_embs 

        h1 = F.relu(self.h1_weights(edge_embs))
        if self.with_dropout:
            h1 = F.dropout(h1, training=self.training)
        logits = self.output_layer(h1)
        pred_probs = torch.sigmoid(logits)

        return pred_probs


# class MultiplexAttention(nn.Module):
#     def __init__(self, n_views, in_channels, out_channels) -> None:
#         super().__init__()
#         self.n_views = n_views
#         self.linear = [nn.Linear(in_channels, out_channels) for _ in range(n_views)]
#         self.activation = nn.Tanh()

#         # self.reset_parameters()

#     def forward(self, x):
#         # input size: views * batch_size * #nodes * in_channels
#         out = torch.zeros_like(x)
#         for view in range(self.n_views):
#             out[view] = self.linear[view](x[view])

#         s   = x.mean(dim=2) 
#         s   = torch.unsqueeze(s, 2)

#         out = torch.transpose(out, dim0=2, dim1=3) 
#         out = torch.matmul(s, out)
#         out = torch.squeeze(out, 2)
#         out = self.activation(out)
#         out = F.softmax(out, dim=0) 

#         out = torch.transpose(out, dim0=0, dim1=1)
#         out = torch.transpose(out, dim0=1, dim1=2)
#         out = torch.unsqueeze(out, 2)
#         x   = torch.transpose(x, dim0=0, dim1=1)
#         x   = torch.transpose(x, dim0=1, dim1=2)
#         out = torch.matmul(out, x)
#         out = torch.squeeze(out, 2) # Size: batch_size * #nodes * in_channels

#         return out
    
# class ANOMULY(nn.Module):
#     def __init__(self, input_dim:int, n_views=5,  hidden_dims=[64,64], gru_hidden_dims=[64,64], attns_in_channels=[64,64], attns_out_channels=[64,64], n_layers=2, **kwargs) -> None:
#         super(ANOMULY, self).__init__()  

#         self.n_layers = n_layers
#         self.n_views = n_views
#         self.hidden_dims = hidden_dims
#         self.gru_hidden_dims = gru_hidden_dims
#         self.input_dim = input_dim
#         self.attns_in_channels = attns_in_channels
#         self.attns_out_channels = attns_out_channels
#         # Initialize the graph layers as a two-dimensional list
#         self.graph_gcn_layers = [[[] for _ in range(n_layers)] for _ in range(n_views)]
#         self.graph_gru_layers = [[[] for _ in range(n_layers)] for _ in range(n_views)]
#         self.batch_norm = [[[] for _ in range(n_layers)] for _ in range(n_views)]
#         self.attns = []

#         for layer in range(n_layers):
#             for view in range(n_views):
#                 if layer == 0:
#                     self.graph_gcn_layers[view][layer].append(GCNConv(self.input_dim, hidden_dims[layer]))
#                 else:
#                     self.graph_gcn_layers[view][layer].append(GCNConv(gru_hidden_dims[layer - 1], hidden_dims[layer]))

#                 self.graph_gru_layers[view][layer].append(nn.GRU(input_size=hidden_dims[layer], hidden_size=gru_hidden_dims[layer]))
#                 self.batch_norm[view][layer].append(BatchNorm(hidden_dims[layer]))
            
#             self.attns.append(MultiplexAttention(n_views, self.attns_in_channels[layer], self.attns_out_channels[layer]))

#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.reset_parameters()      
                
#     def forward(self, x, edge_index, hiddens=None):
#         if hiddens is None:
#             for view in range(self.n_views):
#                 hiddens[view] = self.init_hidden(self.hidden_dim)
#         for layer in range(self.n_layers):
#             for view in range(self.n_views):
#                 x[view] = self.graph_gcn_layers[view][layer](x[view], edge_index[view])
#                 x[view] = self.batch_norm[view][layer](x[view])
#                 x[view] = self.act(x[view])
#                 x[view] = F.dropout(x[view], training=self.training)
#                 x[view], hiddens[view][layer] = self.graph_gru_layers[view][layer](x[view], hiddens[view][layer])
#             attn_out = self.attns[layer](x)
#             for view in range(self.n_views):
#                 x[view] = x[view] + attn_out
        
#         x = torch.sigmoid(x)
#         return x.squeeze()


# class SingleANOMULY(nn.Module):
#     def __init__(self, input_dim:int, hidden_dims=[64, 64], gru_hidden_dims=[64, 64], n_layers=2, **kwargs)-> None:
#         super().__init__()  

#         self.n_layers = n_layers
#         self.hidden_dims = hidden_dims
#         self.gru_hidden_dims = gru_hidden_dims
#         self.input_dim = input_dim
#         self.graph_gcn_layers = nn.ModuleList()
#         self.graph_gru_layers = nn.ModuleList()
#         self.batch_norm = nn.ModuleList()

#         for layer in range(n_layers):
#             if layer == 0:
#                 self.graph_gcn_layers.append(GCNConv(self.input_dim, hidden_dims[layer]))
#             else:
#                 self.graph_gcn_layers.append(GCNConv(gru_hidden_dims[layer - 1], hidden_dims[layer]))

#             self.graph_gru_layers.append(nn.GRU(input_size=hidden_dims[layer], hidden_size=gru_hidden_dims[layer], batch_first=True))
#             self.batch_norm.append(BatchNorm(hidden_dims[layer]))

#         self.act = F.relu
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # self.reset_parameters()      
                
#     def forward(self, x, edge_index, hiddens=None):
#         if hiddens is None:
#             hiddens = self.init_hidden(64)

#         new_hiddens = []  # 创建一个新的隐藏状态列表，以避免修改原有的隐藏状态
#         for layer in range(self.n_layers):
#             x = self.graph_gcn_layers[layer](x, edge_index)
#             x = self.batch_norm[layer](x)
#             x = self.act(x)
#             x = F.dropout(x, training=self.training)
#             x, new_hidden = self.graph_gru_layers[layer](x, hiddens[layer])
#             new_hiddens.append(new_hidden.squeeze(0))  # 不直接修改原来的隐藏状态

#         x = torch.sigmoid(x)

#         return x.squeeze()


#     def reset_parameters(self):
#         for layer in self.graph_gcn_layers:
#             layer.reset_parameters()
#         for gru in self.graph_gru_layers:
#             gru.reset_parameters()

   
#     def init_hidden(self, dim):
#         return torch.zeros((self.n_layers, 1, self.gru_hidden_dims[0])).to(self.device)
                   
class GConv(nn.Module):
    def __init__(self, input_dim:int, output_dim=64, act = None, bias = True, dropout = 0.):
        super(GConv, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.conv = SAGEConv(input_dim, output_dim).to(self.device)
        self.conv = GCNConv(input_dim, output_dim, edge_dim=64).to(self.device)
        self.dropout = dropout
        self.act = act
    
    def forward(self, x, edge_index):
        z = self.conv(x, edge_index)
        if self.act != None:
            z = self.act(z)
        z = F.dropout(z, p = self.dropout, training = self.training)
        return z 

class Graph_GRU(nn.Module):
    def __init__(self, input_size, hidden_size, layer_num, device, bias = True):
        super(Graph_GRU, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.layer_num = layer_num
        self.weight_xz = []
        self.weight_hz = []
        self.weight_xr = []
        self.weight_hr = []
        self.weight_xh = []
        self.weight_hh = []
        
        for i in range(layer_num):
            if i == 0:
                self.weight_xz.append(GConv(input_size, hidden_size, bias=bias)) 
                self.weight_hz.append(GConv(hidden_size, hidden_size,bias=bias)) 
                self.weight_xr.append(GConv(input_size, hidden_size, bias=bias)) 
                self.weight_hr.append(GConv(hidden_size, hidden_size, bias=bias)) 
                self.weight_xh.append(GConv(input_size, hidden_size, bias=bias)) 
                self.weight_hh.append(GConv(hidden_size, hidden_size, bias=bias)) 
            else:
                self.weight_xz.append(GConv(hidden_size, hidden_size, bias=bias)) 
                self.weight_hz.append(GConv(hidden_size, hidden_size, bias=bias)) 
                self.weight_xr.append(GConv(hidden_size, hidden_size, bias=bias)) 
                self.weight_hr.append(GConv(hidden_size, hidden_size, bias=bias)) 
                self.weight_xh.append(GConv(hidden_size, hidden_size, bias=bias)) 
                self.weight_hh.append(GConv(hidden_size, hidden_size, bias=bias)) 
    
    def forward(self, x, edge_index, h):
        h_out = torch.zeros(h.size()).to(self.device)
        for i in range(self.layer_num):
            if i == 0:
                z_g = torch.sigmoid(self.weight_xz[i](x, edge_index) + self.weight_hz[i](h[i], edge_index))
                r_g = torch.sigmoid(self.weight_xr[i](x, edge_index) + self.weight_hr[i](h[i], edge_index))
                h_tilde_g = torch.tanh(self.weight_xh[i](x, edge_index) + self.weight_hh[i](r_g * h[i], edge_index))
                out = z_g * h[i] + (1 - z_g) * h_tilde_g
            else:
                z_g = torch.sigmoid(self.weight_xz[i](out, edge_index) + self.weight_hz[i](h[i], edge_index))
                r_g = torch.sigmoid(self.weight_xr[i](out, edge_index) + self.weight_hr[i](h[i], edge_index))
                h_tilde_g = torch.tanh(self.weight_xh[i](out, edge_index) + self.weight_hh[i](r_g * h[i], edge_index))
                out = z_g * h[i] + (1 - z_g) * h_tilde_g
            h_out[i] = out
        return h_out
    
    
class Generative(nn.Module):
    def __init__(self, x_dim, h_dim, z_dim, layer_num, device):
        super().__init__()
        self.phi_x = nn.Sequential(nn.Linear(x_dim, h_dim), nn.ReLU())
        self.phi_z = nn.Sequential(nn.Linear(z_dim, h_dim), nn.ReLU())
        
        self.enc = GConv(h_dim + h_dim, h_dim, act=F.relu)
        self.enc_mean = GConv(h_dim, z_dim,)
        self.enc_std = GConv(h_dim, z_dim, act=F.softplus)
        
        self.prior = nn.Sequential(nn.Linear(h_dim+1, h_dim), nn.ReLU())
        self.prior_mean = nn.Sequential(nn.Linear(h_dim, z_dim))
        self.prior_std = nn.Sequential(nn.Linear(h_dim, z_dim), nn.Softplus())
        self.device = device

        self.rnn = Graph_GRU(h_dim+h_dim, h_dim, layer_num, device)


    def forward(self, x, h, diff, edge_index):
        phiX = self.phi_x(x)
        enc_x = self.enc(torch.cat([phiX, h[-1]], 1), edge_index)
        enc_x_mean = self.enc_mean(enc_x, edge_index)
        enc_x_std = self.enc_std(enc_x, edge_index)
        prior_x = self.prior(torch.cat([h[-1], diff], 1))
        # prior_x = torch.randn(prior_x.shape).cuda()
        prior_x_mean = self.prior_mean(prior_x)
        prior_x_std = self.prior_std(prior_x)
        z = self.random_sample(enc_x_mean, enc_x_std)
        phiZ = self.phi_z(z)
        h_out = self.rnn(torch.cat([phiX, phiZ], 1), edge_index, h)
        
        return (prior_x_mean, prior_x_std), (enc_x_mean, enc_x_std), z, h_out
    
    def random_sample(self, mean, std):
        eps1 = torch.FloatTensor(std.size()).normal_().to(self.device)
        return eps1.mul(std).add_(mean)

class Contrastive(nn.Module):
    def __init__(self, device, z_dim, window):
        super(Contrastive, self).__init__()
        self.device = device
        self.max_dis = window
        self.linear = nn.Linear(z_dim, z_dim)
    
    def forward(self, all_z, all_node_idx):
        t_len = len(all_node_idx)
        nce_loss = 0
        f = lambda x: torch.exp(x)
        # self.neg_sample = last_h
        for i in range(t_len - self.max_dis):
            for j in range(i+1, i+self.max_dis+1):
                nodes_1, nodes_2 = all_node_idx[i].tolist(), all_node_idx[j].tolist()
                common_nodes = list(set(nodes_1) & set(nodes_2))
                z_anchor = all_z[i][common_nodes]
                z_anchor = self.linear(z_anchor)
                positive_samples = all_z[j][common_nodes]
                pos_sim = f(self.sim(z_anchor, positive_samples, True))
                neg_sim = f(self.sim(z_anchor, all_z[j], False))
                #index = torch.LongTensor(common_nodes).unsqueeze(1).to(self.device)
                neg_sim = neg_sim.sum(dim=-1).unsqueeze(1) #- torch.gather(neg_sim, 1, index)
                nce_loss += -torch.log(pos_sim / (neg_sim)).mean()
                # nce_loss += -(torch.log(pos_sim / (pos_sim + neg_sim.sum(dim=-1) - torch.gather(neg_sim, 1, index)))).mean()
        return nce_loss / (self.max_dis * (t_len - self.max_dis))   

    def sim(self, h1, h2, pos=False):
        z1 = F.normalize(h1, dim=-1, p=2)
        z2 = F.normalize(h2, dim=-1, p=2)
        if pos == True:
            return torch.einsum('ik, ik -> i', z1, z2).unsqueeze(1)
        else:
            return torch.mm(z1, z2.t())       
                

class RustGraph(nn.Module):
    def __init__(self, input_dim:int, hidden_dim=64, z_dim=64, layer_num=2, **kwargs):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Generative(input_dim, hidden_dim, z_dim, layer_num, self.device)
        self.contrastive = Contrastive(self.device, z_dim, window=1)
        self.dec = InnerProductDecoder()
        self.mse = nn.MSELoss(reduction='mean')
        self.fcc = FCC(z_dim, 1, self.device)
        self.linear = nn.Sequential(nn.Linear(z_dim, input_dim), nn.ReLU())
        self.layer_num = layer_num
        self.hidden_dim = hidden_dim
        self.eps = 0.2
        self.EPS = 1e-15

    
    def forward(self, snapshots, y_rect=None, h_t=None):
        kld_loss = 0
        recon_loss = 0
        reg_loss = 0
        all_z, all_h, all_node_idx = [], [], []
        score_list = []
        next_y_list = []
        if isinstance(snapshots, list):
            for t, data in enumerate(snapshots):
                x = data.x
                edge_index = data.edge_index
                y = torch.zeros(edge_index.shape[1], dtype=torch.long)  # 正样本边的标签
                node_index = torch.unique(edge_index)
                
                if h_t == None:
                    h_t = torch.zeros(self.layer_num, x.size(0), self.hidden_dim).to(self.device)
                ev=self._compute_ev(data, is_undirected=True)
                if t == 0:
                    diff = torch.zeros(x.size(0), 1).to(self.device)
                else:
                    diff = torch.abs(torch.sub(ev, pre_ev)).to(self.device)
                pre_ev = ev
                
                (prior_mean_t, prior_std_t), (enc_mean_t, enc_std_t), z_t, h_t = self.encoder(x, h_t, diff, edge_index)
                enc_mean_t_sl = enc_mean_t[node_index, :]
                enc_std_t_sl = enc_std_t[node_index, :]
                prior_mean_t_sl = prior_mean_t[node_index, :]
                prior_std_t_sl = prior_std_t[node_index, :]
                h_t_sl = h_t[-1, node_index, :]
                
                edge_emb = z_t[edge_index[0]] + z_t[edge_index[1]]
                edge_score = self.fcc(edge_emb)
                # 如果 edge_score 是 [N, 1] 形状，确保 y 也是这样
                y = y.unsqueeze(1).float().to(self.device)  # 增加一个维度以匹配 edge_score

                if t == 0:
                    bce_loss = F.binary_cross_entropy(edge_score, y, reduction='none')

                else:
                    bce_loss = torch.vstack([bce_loss, F.binary_cross_entropy(edge_score, y, reduction='none')])
                # bce_loss += self._cal_at_loss(pos_edge, y_pos)
                label_rectifier = self.dec(z_t, edge_index, sigmoid=True)
                label_rectifier = 1 - label_rectifier  # 反转sigmoid输出，现在接近1代表异常，接近0代表正常
                label_rectifier = label_rectifier.unsqueeze(1)
                next_y_list.append((0.9 * y + 0.1 * label_rectifier).detach())

                reg_loss += torch.norm(label_rectifier - edge_score, dim=1, p=2).mean()

                kld_loss += self._kld_gauss(enc_mean_t_sl, enc_std_t_sl, prior_mean_t_sl, prior_std_t_sl)
                recon_loss += self._recon_loss(z_t, x, edge_index)

                
                all_z.append(z_t)
                all_node_idx.append(node_index)
                all_h.append(h_t_sl)
                score_list.append(edge_score)
            bce_loss = bce_loss.squeeze() 
            reg_loss /= snapshots.__len__()
            recon_loss /= snapshots.__len__()
            kld_loss /= snapshots.__len__()
            nce_loss = self.contrastive(all_z, all_node_idx)
            
        return bce_loss, reg_loss, recon_loss + kld_loss, nce_loss, next_y_list, h_t, score_list, all_z

    
    def _compute_ev(self, data, normalization=None, is_undirected=False):
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'
        edge_weight = data.edge_attr
        if edge_weight is not None and edge_weight.numel() != data.num_edges:
            edge_weight = None

        edge_index, edge_weight = get_laplacian(data.edge_index, edge_weight, normalization, num_nodes=data.num_nodes)
        L = to_scipy_sparse_matrix(edge_index, edge_weight, data.num_nodes)
        eig_fn = eigs
        if is_undirected and normalization != 'rw':
            eig_fn = eigsh
        lambda_max,ev = eig_fn(L, k=1, which='LM', return_eigenvectors=True)
        ev = torch.from_numpy(ev)
        return ev 
    
    def decode(self, z, pos_edge_index, neg_edge_index): 
        edge_index = torch.cat([pos_edge_index,neg_edge_index], dim=-1) 
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1) 
    
    def get_link_labels(self, pos_edge_index, neg_edge_index):
        num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(num_links, dtype=torch.float)  
        link_labels[pos_edge_index.size(1):] = 1
        return 
    def _kld_gauss(self, mean_1, std_1, mean_2, std_2):
        num_nodes = mean_1.size()[0]
        kld_element =  (2 * torch.log(std_2 + self.eps) - 2 * torch.log(std_1 + self.eps) +
                        (torch.pow(std_1 + self.eps ,2) + torch.pow(mean_1 - mean_2, 2)) / 
                        torch.pow(std_2 + self.eps ,2) - 1)
        return (0.5 / num_nodes) * torch.mean(torch.sum(kld_element, dim=1), dim=0)    
    
    def _recon_loss(self, z, x, pos_edge_index, neg_edge_index=None):        
        x_hat = self.linear(z)
        feature_loss = self.mse(x, x_hat)
        weight = torch.sigmoid(torch.exp(-torch.norm(z[pos_edge_index[0]] - z[pos_edge_index[1]], dim=1, p=2)))
        pos_loss = (-torch.log(self.dec(z, pos_edge_index) + self.EPS)*weight).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        weight = torch.sigmoid(torch.exp(torch.norm(z[neg_edge_index[0]] - z[neg_edge_index[1]], dim=1, p=2)))
        neg_loss = (-torch.log(1 - self.dec(z, neg_edge_index) + self.EPS)*weight).mean()
        return pos_loss + neg_loss + feature_loss
    
class FCC(nn.Module):
    def __init__(self, in_channels, out_channels, device):
        super(FCC,self).__init__()
        self.device = device
        self.linear = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=True, device=self.device),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.linear(x)
    
    
    
    
from transformers.models.bert.modeling_bert import BertPreTrainedModel    
from transformers import PretrainedConfig

class MyConfig(PretrainedConfig):

    def __init__(
        self,
        k=5,
        max_hop_dis_index = 100,
        max_inti_pos_index = 100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=32,
        hidden_act="gelu",
        hidden_dropout_prob=0.5,
        attention_probs_dropout_prob=0.3,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        is_decoder=False,
        batch_size = 256,
        window_size = 1,
        weight_decay = 5e-4,
        **kwargs
    ):
        super(MyConfig, self).__init__(**kwargs)
        self.max_hop_dis_index = max_hop_dis_index
        self.max_inti_pos_index = max_inti_pos_index
        self.k = k
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.is_decoder = is_decoder
        self.batch_size = batch_size
        self.window_size = window_size
        self.weight_decay = weight_decay
        
class TADDY(BertPreTrainedModel):
    learning_record_dict = {}
    lr = 0.001
    weight_decay = 5e-4
    max_epoch = 500
    spy_tag = True

    def __init__(self, hidden_dim = 32, **kwargs):
        config = MyConfig()  # Create a config with hidden_dim
        super(TADDY, self).__init__(config) 
        self.config = config
        self.transformer = BaseModel(config)
        self.weight_decay = config.weight_decay
        self.init_weights()
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, 1, bias=True),
            nn.Sigmoid()
        ) 
        
    def forward(self, init_pos_ids, hop_dis_ids, time_dis_ids, idx=None):

        outputs = self.transformer(init_pos_ids, hop_dis_ids, time_dis_ids)
        sequence_output = 0
        for i in range(self.config.k+1):
            sequence_output += outputs[0][:,i,:]
        sequence_output /= float(self.config.k+1)

        return sequence_output
    
    def decode(self, z): 
        score = self.scorer(z)
        return score.squeeze()
    
class BaseModel(BertPreTrainedModel):
    data = None
    def __init__(self, config):
        super(BaseModel, self).__init__(config)
        self.config = config

        self.embeddings = EdgeEncoding(config)
        self.encoder = TransformerEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.raw_feature_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.raw_feature_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def setting_preparation(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None, encoder_hidden_states=None, encoder_attention_mask=None,):

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(torch.long)  # not converting to long will cause errors with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype)
        else:
            head_mask = [None] * self.config.num_hidden_layers

        return token_type_ids, extended_attention_mask, encoder_extended_attention_mask, head_mask


    def forward(self, init_pos_ids, hop_dis_ids, time_dis_ids, head_mask=None):
        if head_mask is None:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(init_pos_ids=init_pos_ids,
                                           hop_dis_ids=hop_dis_ids, time_dis_ids=time_dis_ids)
        encoder_outputs = self.encoder(embedding_output, head_mask=head_mask) #这里的输出是tuple，因为在某些设定下要输出别的信息（中间分析用）
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs

    def run(self):
        pass

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None, encoder_hidden_states=None, encoder_attention_mask=None):
        all_hidden_states = ()
        all_attentions = ()
        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask)
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs

TransformerLayerNorm = torch.nn.LayerNorm
class EdgeEncoding(nn.Module):
    def __init__(self, config):
        super(EdgeEncoding, self).__init__()
        self.config = config

        self.inti_pos_embeddings = nn.Embedding(config.max_inti_pos_index, config.hidden_size)
        self.hop_dis_embeddings = nn.Embedding(config.max_hop_dis_index, config.hidden_size)
        self.time_dis_embeddings = nn.Embedding(config.max_hop_dis_index, config.hidden_size)

        self.LayerNorm = TransformerLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, init_pos_ids=None, hop_dis_ids=None, time_dis_ids=None):

        position_embeddings = self.inti_pos_embeddings(init_pos_ids)
        hop_embeddings = self.hop_dis_embeddings(hop_dis_ids)
        time_embeddings = self.hop_dis_embeddings(time_dis_ids)

        embeddings = position_embeddings + hop_embeddings + time_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs
    
    
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