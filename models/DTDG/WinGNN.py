import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from .MLP import MLP

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