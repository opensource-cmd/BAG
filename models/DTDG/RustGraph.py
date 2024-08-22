import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix, negative_sampling, remove_self_loops, add_self_loops
from torch_geometric.nn import GCNConv, InnerProductDecoder
from .MLP import MLP
from scipy.sparse.linalg import eigs, eigsh

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
        self.scorer = MLP(input_dim=input_dim, hidden_dim=input_dim)
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
    
    def decode(self, z, edge_index):  
        edge_embs = z[edge_index[0]] * z[edge_index[1]]  
        logits = self.scorer(edge_embs)
        return logits
    
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
                