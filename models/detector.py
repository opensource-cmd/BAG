from models.gnn import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import numpy as np
import pandas as pd
import networkx as nx
import itertools
from torch_geometric.nn import Node2Vec
import scipy.sparse as sp
from scipy.linalg import inv
       
class staticGNNDetector(object):
    def __init__(self, train_config, model_config, graph):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.graph = graph.to(train_config['device']) 
        self.best_score = -1
        self.patience_knt = 0
        
        gnn = globals()[model_config['model']]
        model_config['input_dim'] = self.graph.x.shape[1]
        self.model = gnn(**model_config).to(train_config['device'])

    def pairwise_loss(self, pos_score, neg_score, margin=0.6):
        return F.relu(margin + pos_score - neg_score).mean()
    
    def get_link_labels(self, pos_edge_index, neg_edge_index):
        num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(num_links, dtype=torch.float)  
        link_labels[pos_edge_index.size(1):] = 1
        return link_labels

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(),  lr=self.model_config['lr']) 
        
        for e in range(self.train_config['epochs']):
            self.model.train()
            train_pos = self.graph.train_pos_edge_index
            train_neg = negative_sampling(
                edge_index = train_pos,
                num_nodes = self.graph.num_nodes,
                num_neg_samples = train_pos.size(1)
            ).to(dtype=torch.int64)
            
            optimizer.zero_grad()  
            z = self.model(self.graph.x, self.graph.edge_index)  # Get node embeddings
            pos_logits = self.model.decode(z, train_pos)
            neg_logits = self.model.decode(z, train_neg)

            pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.zeros_like(pos_logits)) 
            neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.ones_like(neg_logits).cuda()) 
            loss = pos_loss + neg_loss
            loss.backward()  
            optimizer.step()  
            
            val_score = self.eval(self.model, self.graph, self.graph.val_pos_edge_index, self.graph.val_neg_edge_index)
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(self.model, self.graph, self.graph.test_pos_edge_index, self.graph.test_neg_edge_index)
 
                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break        
        return test_score

    @torch.no_grad()
    def eval(self, model, graph, eval_pos, eval_neg):
        self.model.eval()
        z = model(graph.x, graph.edge_index)  # Get node embeddings
        
        result = {}
        pos_logits = model.decode(z, eval_pos)
        neg_logits = model.decode(z, eval_neg)
        
        y_true = torch.cat([torch.zeros(pos_logits.size(0)), torch.ones(neg_logits.size(0))], dim=0)
        y_pred = torch.cat([torch.sigmoid(pos_logits), torch.sigmoid(neg_logits)], dim=0)  # 在这里使用Sigmoid
        y_pred_labels = (y_pred > 0.5).float()
        
        k = int(y_true.sum().item())
        top_k_preds = y_pred.cpu().argsort(descending=True)[:k] 
        
        rec = y_true.cpu()[top_k_preds].sum().float() / k 
        auc = roc_auc_score(y_true.cpu(), y_pred.cpu())
        prc = average_precision_score(y_true.cpu(), y_pred.cpu())
        f1 = f1_score(y_true.cpu().numpy(), y_pred_labels.cpu().numpy(), average='binary')
        
        result['AUROC'] = auc
        result['AUPRC'] = prc
        result['RecK'] = rec
        result['F1'] = f1
         
        return result

                      
class SnapshotDetector(object):
    def __init__(self, train_config, model_config, train_snapshots, valid_snapshots, test_snapshots):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.train_snapshots = [snapshot.to(train_config['device']) for snapshot in train_snapshots]
        if not isinstance(valid_snapshots, list):
            self.valid_snapshots = valid_snapshots.to(train_config['device'])
        else:
            self.valid_snapshots = [snapshot.to(train_config['device']) for snapshot in valid_snapshots]
        if not isinstance(test_snapshots, list):    
            self.test_snapshots = test_snapshots.to(train_config['device'])
        else:
            self.test_snapshots = [snapshot.to(train_config['device']) for snapshot in test_snapshots]
        self.best_score = -1
        self.patience_knt = 0
        
        gnn = globals()[model_config['model']]
        model_config['input_dim'] = self.train_snapshots[0].x.shape[1]
        model_config['num_nodes'] = self.train_snapshots[0].x.shape[0]
        self.model = gnn(**model_config).to(train_config['device'])
        self.device = train_config['device']
    
    def get_link_labels(self, pos_edge_index, neg_edge_index):
        num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(num_links, dtype=torch.float)  
        link_labels[pos_edge_index.size(1):] = 1
        return link_labels

    def pairwise_loss(self, pos_score, neg_score, margin=0.6):
        return F.relu(margin + pos_score - neg_score).mean()
                
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        for e in range(self.train_config['epochs']):
            loss = 0
            self.model.train()
            z_list = self.model(self.train_snapshots)
            for i, data in enumerate(self.train_snapshots):
                train_pos = data.edge_index  # Positive and negative edges
                train_neg = negative_sampling(
                    edge_index = train_pos,
                    num_nodes=data.num_nodes,
                    num_neg_samples = train_pos.size(1)
                ).to(dtype=torch.int64, device=self.device)

                z = z_list[i]   # Get node embeddings
                pos_logits = self.model.decode(z, train_pos)
                neg_logits = self.model.decode(z, train_neg)
                if self.train_config['loss'] == 'pairwise':
                    loss += self.pairwise_loss(torch.sigmoid(pos_logits), torch.sigmoid(neg_logits))   
                else:    
                    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.zeros_like(pos_logits)) 
                    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.ones_like(neg_logits).cuda()) 
                    loss += pos_loss + neg_loss 
                   
            loss = loss / len(self.train_snapshots)           
            loss.backward()
            optimizer.step()                
            optimizer.zero_grad()  
                
            self.model.eval()
            val_score = self.eval(self.model, self.valid_snapshots)
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(self.model, self.test_snapshots)

                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break  
                      
        return test_score  
    
    @torch.no_grad()
    def eval(self, model, snapshots):
        rec, auc, prc, f1 = 0, 0, 0, 0
        z = model(snapshots)
        for i, data in enumerate(snapshots):
            result = {}
            z = z[i]
            pos_mask = data.y == 0
            neg_mask = data.y == 1

            eval_pos = data.edge_index[:, pos_mask]
            eval_neg = data.edge_index[:, neg_mask]

            pos_logits = model.decode(z, eval_pos)
            neg_logits = model.decode(z, eval_neg)
            
            y_true = torch.cat([torch.zeros(pos_logits.size(0)), torch.ones(neg_logits.size(0))], dim=0)
            # y_true = self.get_link_labels(eval_pos, eval_neg)
            y_pred = torch.cat([torch.sigmoid(pos_logits), torch.sigmoid(neg_logits)], dim=0)
            y_true = y_true.cpu()
            y_pred = y_pred.cpu()
            y_pred_labels = (y_pred > 0.5).float()

            k = 50
            sorted_indices = np.argsort(y_pred.numpy())[::-1] 
            top_k_indices = sorted_indices[:k]
            
            rec += np.sum(y_true.numpy()[top_k_indices]) / np.sum(y_true.numpy()) if np.sum(y_true.numpy()) != 0 else 0
            auc += roc_auc_score(y_true.numpy(), y_pred.numpy())
            prc += average_precision_score(y_true.numpy(), y_pred.numpy())
            f1 += f1_score(y_true.numpy(), y_pred_labels.numpy(), average='binary')
        
        result['AUROC'] = auc/len(snapshots)
        result['AUPRC'] = prc/len(snapshots)
        result['RecK'] = rec/len(snapshots)
        result['F1'] = f1/len(snapshots)
            
        return result     


class SnapshotsDetector(object):
    def __init__(self, train_config, model_config, train_snapshots, valid_snapshot, test_snapshot):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.train_snapshots = [snapshot.to(train_config['device']) for snapshot in train_snapshots]
        if not isinstance(valid_snapshot, list):
            self.valid_snapshot = valid_snapshot.to(train_config['device'])
        else:
            self.valid_snapshot = [snapshot.to(train_config['device']) for snapshot in valid_snapshot]
        if not isinstance(test_snapshot, list):    
            self.test_snapshot = test_snapshot.to(train_config['device'])
        else:
            self.test_snapshot = [snapshot.to(train_config['device']) for snapshot in test_snapshot]
        self.best_score = -1
        self.patience_knt = 0
        
        gnn = globals()[model_config['model']]
        model_config['input_dim'] = self.train_snapshots[0].x.shape[1]
        model_config['num_nodes'] = self.train_snapshots[0].x.shape[0]
        self.model = gnn(**model_config).to(train_config['device'])

    def decode(self, z, pos_edge_index, neg_edge_index): 
        edge_index = torch.cat([pos_edge_index,neg_edge_index], dim=-1) 
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1) 
    
    def get_link_labels(self, pos_edge_index, neg_edge_index):
        num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(num_links, dtype=torch.float)  
        link_labels[pos_edge_index.size(1):] = 1
        return link_labels
    
    def pairwise_loss(self, pos_score, neg_score, margin=0.6):
        return F.relu(margin + pos_score - neg_score).mean()
                
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        for e in range(self.train_config['epochs']):
            loss = 0
            torch.autograd.set_detect_anomaly(True)
            
            self.model.train()
            _,_,_,_,_,_,_,z_list = self.model(self.train_snapshots)
            for i, data in enumerate(self.train_snapshots):
                pos_edge_index = data.edge_index  # Positive and negative edges
                neg_edge_index = negative_sampling(
                    edge_index = data.edge_index,
                    num_nodes = data.num_nodes,
                    num_neg_samples = pos_edge_index.size(1)
                ).to(dtype=torch.int64)
                z = z_list[i]   
                logits = self.model.decode(z, pos_edge_index, neg_edge_index)
                labels = self.get_link_labels(pos_edge_index, neg_edge_index)
                pos_score = logits[:pos_edge_index.size(1)]
                neg_score = logits[pos_edge_index.size(1):]
                
                if self.train_config['loss'] == 'pairwise':
                    loss += self.pairwise_loss(pos_score, neg_score)   
                else:    
                    loss += F.binary_cross_entropy(logits, labels.cuda())     
                   
            loss = loss / len(self.train_snapshots)     
            loss = loss / len(self.train_snapshots)           
            loss.backward()
            optimizer.step()                
            optimizer.zero_grad()  
                
            self.model.eval()
            val_score = self.eval(self.model, self.valid_snapshot)
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(self.model, self.test_snapshot)

                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break        
        return test_score  
    
    @torch.no_grad()
    def eval(self, model, graph):
        model.eval()
        rec, auc, prc, f1 = 0, 0, 0, 0
        _,_,_,_,_,_,_,z_list = self.model(graph)
        result = {}
        for i, data in enumerate(graph):
            z = z_list[i]     
            pos_mask = data.y == 0
            neg_mask = data.y == 1
            pos_edge_index = data.edge_index[:, pos_mask]
            neg_edge_index = data.edge_index[:, neg_mask]
            
            link_logits = model.decode(z, pos_edge_index, neg_edge_index)
        
            link_probs = link_logits.sigmoid()
            link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)
            link_preds = (link_probs > 0.5).float()
        
            k = int(link_labels.sum().item())
            top_k_preds = link_probs.cpu().argsort(descending=True)[:k] 
            rec += link_labels.cpu()[top_k_preds].sum().float() / k 
            auc += roc_auc_score(link_labels.cpu(),link_probs.cpu())
            prc += average_precision_score(link_labels.cpu(),link_probs.cpu())
            f1 += f1_score(link_labels.cpu().numpy(), link_preds.cpu().numpy(), average='binary')
        
        result['AUROC'] = auc/len(graph)
        result['AUPRC'] = prc/len(graph)
        result['RecK'] = rec/len(graph)
        result['F1'] = f1/len(graph)
            
        return result       



class DTDGDetector(object):
    def __init__(self, train_config, model_config, train_snapshots, valid_snapshots, test_snapshots):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.train_snapshots = [snapshot.to(train_config['device']) for snapshot in train_snapshots]
        if not isinstance(valid_snapshots, list):
            self.valid_snapshots = valid_snapshots.to(train_config['device'])
        else:
            self.valid_snapshots = [snapshot.to(train_config['device']) for snapshot in valid_snapshots]
        if not isinstance(test_snapshots, list):    
            self.test_snapshots = test_snapshots.to(train_config['device'])
        else:
            self.test_snapshots = [snapshot.to(train_config['device']) for snapshot in test_snapshots]
        self.best_score = -1
        self.patience_knt = 0
        
        gnn = globals()[model_config['model']]
        model_config['input_dim'] = self.train_snapshots[0].x.shape[1]
        model_config['num_nodes'] = self.train_snapshots[0].x.shape[0]
        self.model = gnn(**model_config).to(train_config['device'])
    
    def get_link_labels(self, pos_edge_index, neg_edge_index):
        num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(num_links, dtype=torch.float)  
        link_labels[pos_edge_index.size(1):] = 1
        return link_labels

    def pairwise_loss(self, pos_score, neg_score, margin=0.6):
        return F.relu(margin + pos_score - neg_score).mean()
                
    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'])
        for e in range(self.train_config['epochs']):
            loss = 0
            torch.autograd.set_detect_anomaly(True)

            self.model.train()
            for i, data in enumerate(self.train_snapshots):
                train_pos = data.edge_index  # Positive and negative edges
                train_neg = negative_sampling(
                    edge_index = train_pos,
                    num_nodes = data.num_nodes,
                    num_neg_samples = train_pos.size(1)
                ).to(dtype=torch.int64)
    
                z = self.model(data.x, data.edge_index)
                pos_logits = self.model.decode(z, train_pos)
                neg_logits = self.model.decode(z, train_neg)
                if self.train_config['loss'] == 'pairwise':
                    loss += self.pairwise_loss(torch.sigmoid(pos_logits), torch.sigmoid(neg_logits))   
                else:    
                    pos_loss = F.binary_cross_entropy_with_logits(pos_logits, torch.zeros_like(pos_logits)) 
                    neg_loss = F.binary_cross_entropy_with_logits(neg_logits, torch.ones_like(neg_logits)) 
                    loss = pos_loss + neg_loss  
                    loss += loss 
                   
            loss = loss / len(self.train_snapshots)           
            loss.backward()
            optimizer.step()                
            optimizer.zero_grad()  
                
            self.model.eval()
            val_score = self.eval(self.model, self.valid_snapshots)
            if val_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = val_score[self.train_config['metric']]
                test_score = self.eval(self.model, self.test_snapshots)

                print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                    e, loss, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                    test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break  
                      
        return test_score
    
    @torch.no_grad()
    def eval(self, model, graph):
        rec, auc, prc, f1 = 0, 0, 0, 0
        for i, data in enumerate(graph):
            z = model(data.x, data.edge_index)
            result = {}
            pos_mask = data.y == 0
            neg_mask = data.y == 1

            eval_pos = data.edge_index[:, pos_mask]
            eval_neg = data.edge_index[:, neg_mask]

            pos_logits = model.decode(z, eval_pos)
            neg_logits = model.decode(z, eval_neg)
            
            y_true = torch.cat([torch.zeros(pos_logits.size(0)), torch.ones(neg_logits.size(0))], dim=0)
            # y_true = self.get_link_labels(eval_pos, eval_neg)
            y_pred = torch.cat([torch.sigmoid(pos_logits), torch.sigmoid(neg_logits)], dim=0)
            y_true = y_true.cpu()
            y_pred = y_pred.cpu()
            y_pred_labels = (y_pred > 0.5).float()

            k = 50
            sorted_indices = np.argsort(y_pred.numpy())[::-1] 
            top_k_indices = sorted_indices[:k]
            
            rec += np.sum(y_true.numpy()[top_k_indices]) / np.sum(y_true.numpy()) if np.sum(y_true.numpy()) != 0 else 0
            auc += roc_auc_score(y_true.numpy(), y_pred.numpy())
            prc += average_precision_score(y_true.numpy(), y_pred.numpy())
            f1 += f1_score(y_true.numpy(), y_pred_labels.numpy(), average='binary')
        
        result['AUROC'] = auc/len(graph)
        result['AUPRC'] = prc/len(graph)
        result['RecK'] = rec/len(graph)
        result['F1'] = f1/len(graph)
            
        return result
                              
class CTTGDetector(object):
    def __init__(self, train_config, model_config):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config



class Detector_new(object):
    def __init__(self, train_config, model_config, train_snapshots, test_snapshots):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.train_snapshots = [snapshot.to(train_config['device']) for snapshot in train_snapshots]
        self.test_snapshots = [snapshot.to(train_config['device']) for snapshot in test_snapshots]

        self.best_score = -1
        self.patience_knt = 0
        
        gnn = globals()[model_config['model']]
        model_config['input_dim'] = self.train_snapshots[0].x.shape[1]
        self.model = gnn(**model_config).to(train_config['device'])

    
    def pairwise_loss(self, pos_score, neg_score, margin=0.6):
        return F.relu(margin + pos_score - neg_score).mean()
    
    def train(self):
        optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.model_config['lr'])

        for e in range(self.train_config['epochs']):
            loss = 0
            self.model.train()
            
            for i, data in enumerate(self.train_snapshots):
                optimizer.zero_grad()  
                
                pos_edge_index = data.edge_index  # Positive and negative edges
                neg_edge_index = negative_sampling(
                    edge_index = data.edge_index,
                    num_nodes = data.num_nodes,
                    num_neg_samples = pos_edge_index.size(1)
                ).to(dtype=torch.int64)
                pos_score = self.model(data.x, pos_edge_index)  
                neg_score = self.model(data.x, neg_edge_index)  
                loss += self.pairwise_loss(pos_score, neg_score)      
                
            loss /= len(self.train_snapshots)

            loss.backward()  
            optimizer.step()  
            optimizer.zero_grad()    
                  
            self.model.eval()

            test_score = self.eval(self.model, self.test_snapshots)
            if test_score[self.train_config['metric']] > self.best_score:
                self.patience_knt = 0
                self.best_score = test_score[self.train_config['metric']]
                print('Epoch {}, Loss {:.4f}, test AUC {:.4f}, PRC {:.4f}'.format(
                    e, loss, test_score['AUROC'], test_score['AUPRC']))
            else:
                self.patience_knt += 1
                if self.patience_knt > self.train_config['patience']:
                    break
 
            # print('Epoch {}, Loss {:.4f}, test AUC {:.4f}, PRC {:.4f}'.format(
            #         e, loss, test_score['AUROC'], test_score['AUPRC']))
            
        return test_score
                   
    def eval(self, model, snapshots):
        result = {}     
        y_true = []
        y_pred = []
        
        with torch.no_grad(): 
            auc,prc = 0,0
            for i, data in enumerate(snapshots):
                score = model(data.x, data.edge_index)  
                
                y_true.extend(data.y.cpu().numpy())
                y_pred.extend(score.cpu().numpy())
                auc += roc_auc_score(y_true, y_pred)
                prc += average_precision_score(y_true, y_pred)  

            auc /= len(snapshots)
            prc /= len(snapshots)
            
            result['AUROC'] = auc


        
class taddyDetector(object):
    def __init__(self, train_config, model_config, train_snapshots, valid_snapshot, test_snapshot):
        super().__init__()
        self.model_config = model_config
        self.train_config = train_config
        self.train_snapshots = [snapshot.to(train_config['device']) for snapshot in train_snapshots]
        if not isinstance(valid_snapshot, list):
            self.valid_snapshot = valid_snapshot.to(train_config['device'])
        else:
            self.valid_snapshot = [snapshot.to(train_config['device']) for snapshot in valid_snapshot]
        if not isinstance(test_snapshot, list):    
            self.test_snapshot = test_snapshot.to(train_config['device'])
        else:
            self.test_snapshot = [snapshot.to(train_config['device']) for snapshot in test_snapshot]
        
        self.train_len = len(self.train_snapshots)
        self.valid_len = len(self.valid_snapshot)    
        self.best_score = -1
        self.patience_knt = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gnn = globals()[model_config['model']]
        model_config['input_dim'] = self.train_snapshots[0].x.shape[1]
        model_config['num_nodes'] = self.train_snapshots[0].x.shape[0]
        all_snapshots = self.train_snapshots + self.valid_snapshot + self.test_snapshot

        self.model = gnn(**model_config).to(train_config['device'])
        
        pos_edges = []
        neg_edges = []
        pos_edges_index = []
        neg_edges_index = []
        for snapshot in all_snapshots:
            neg_edge_index = negative_sampling(
                        edge_index = snapshot.edge_index,
                        num_nodes = model_config['num_nodes'],
                        num_neg_samples = snapshot.edge_index.size(1)
                    ).to(dtype=torch.int64)
            edge_array = snapshot.edge_index.t().cpu().numpy()  # Assuming snapshot is an object with attribute edge_index
            neg_array = neg_edge_index.t().cpu().numpy() 
            
            pos_edges.append(edge_array)
            pos_edges_index.append(snapshot.edge_index)
            neg_edges.append(neg_array)
            neg_edges_index.append(neg_edge_index)
        self.neg_edges_index = neg_edges_index 
        self.neg_edges = neg_edges   
        self.pos_edges_index = pos_edges_index    
        self.pos_edges = pos_edges
        self.idx = np.array(range(model_config['num_nodes']))
        self.X = None
        self.c = 0.15
        _, self.S = self.compute_from_edges(self.pos_edges, model_config['num_nodes'])
        # self.S = [s.to(self.device) if torch.is_tensor(s) else s for s in self.S]



    def compute_from_edges(self, edges, nb_nodes, weights=None):
        """Compute adjs and eigen_adjs from edges."""
        adjs = []
        eigen_adjs = []

        for i, edge in enumerate(edges):
            if weights:
                weight = weights[i]
            else:
                weight = np.ones(edge.shape[0], dtype=np.float32)

            rows, cols = edge[:, 0], edge[:, 1]
            adj = sp.csr_matrix((weight, (rows, cols)), shape=(nb_nodes, nb_nodes), dtype=np.float32)
            adjs.append(self.preprocess_adj(adj))
            
            eigen_adj = self.c * inv((sp.eye(adj.shape[0]) - (1 - self.c) * self.adj_normalize(adj)).toarray())
            np.fill_diagonal(eigen_adj, 0)  # Set diagonal to zero
            eigen_adj = self.normalize(eigen_adj)
            eigen_adjs.append(eigen_adj)

        return adjs, eigen_adjs

    def preprocess_adj(self, adj):
        """Symmetrically normalize adjacency matrix."""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

    def adj_normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -0.5).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx).dot(r_mat_inv)
        return mx

    def normalize(self, mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def get_link_labels(self, pos_edge_index, neg_edge_index):
        num_links = pos_edge_index.size(1) + neg_edge_index.size(1)
        link_labels = torch.zeros(num_links, dtype=torch.float)  
        link_labels[pos_edge_index.size(1):] = 1
        return link_labels

    def pairwise_loss(self, pos_score, neg_score, margin=0.6):
        return F.relu(margin + pos_score - neg_score).mean()

    def WL_setting_init(node_list, link_list):
        node_color_dict = {}
        node_neighbor_dict = {}

        for node in node_list:
            node_color_dict[node] = 1
            node_neighbor_dict[node] = {}

        for pair in link_list:
            u1, u2 = pair
            if u1 not in node_neighbor_dict:
                node_neighbor_dict[u1] = {}
            if u2 not in node_neighbor_dict:
                node_neighbor_dict[u2] = {}
            node_neighbor_dict[u1][u2] = 1
            node_neighbor_dict[u2][u1] = 1

        return node_color_dict, node_neighbor_dict

    def compute_zero_WL(self, node_list, edge_list):
        WL_dict = {}
        for i in node_list:
            WL_dict[i] = 0
        return WL_dict

    # batching + hop + int + time
    def compute_batch_hop(self, node_list, edges_all, num_snap, Ss, k=5, window_size=2):

        batch_hop_dicts = [None] * (window_size-1)
        s_ranking = [0] + list(range(k+1))

        Gs = []
        for snap in range(num_snap):
            G = nx.Graph()
            G.add_nodes_from(node_list)
            G.add_edges_from(edges_all[snap])
            Gs.append(G)

        for snap in range(window_size - 1, num_snap):
            batch_hop_dict = {}
            # S = Ss[snap]
            edges = edges_all[snap]
            for edge in edges:
                edge_idx = str(snap) + '_' + str(edge[0]) + '_' + str(edge[1])
                batch_hop_dict[edge_idx] = []
                for lookback in range(window_size):
                    # s = np.array(Ss[snap-lookback][edge[0]] + Ss[snap-lookback][edge[1]].todense()).squeeze()
                    s = Ss[snap - lookback][edge[0]] + Ss[snap - lookback][edge[1]]
                    s[edge[0]] = -1000 # don't pick myself
                    s[edge[1]] = -1000 # don't pick myself
                    top_k_neighbor_index = s.argsort()[-k:][::-1]

                    indexs = np.hstack((np.array([edge[0], edge[1]]), top_k_neighbor_index))

                    for i, neighbor_index in enumerate(indexs):
                        try:
                            hop1 = nx.shortest_path_length(Gs[snap-lookback], source=edge[0], target=neighbor_index)
                        except:
                            hop1 = 99
                        try:
                            hop2 = nx.shortest_path_length(Gs[snap-lookback], source=edge[1], target=neighbor_index)
                        except:
                            hop2 = 99
                        hop = min(hop1, hop2)
                        batch_hop_dict[edge_idx].append((neighbor_index, s_ranking[i], hop, lookback))
            batch_hop_dicts.append(batch_hop_dict)

        return batch_hop_dicts

    # Dict to embeddings
    def dicts_to_embeddings(self, feats, batch_hop_dicts, wl_dict, num_snap, use_raw_feat=False):

        raw_embeddings = []
        wl_embeddings = []
        hop_embeddings = []
        int_embeddings = []
        time_embeddings = []

        for snap in range(num_snap):

            batch_hop_dict = batch_hop_dicts[snap]

            if batch_hop_dict is None:
                raw_embeddings.append(None)
                wl_embeddings.append(None)
                hop_embeddings.append(None)
                int_embeddings.append(None)
                time_embeddings.append(None)
                continue

            raw_features_list = []
            role_ids_list = []
            position_ids_list = []
            hop_ids_list = []
            time_ids_list = []

            for edge_idx in batch_hop_dict:

                neighbors_list = batch_hop_dict[edge_idx]
                edge = edge_idx.split('_')[1:]
                edge[0], edge[1] = int(edge[0]), int(edge[1])

                raw_features = []
                role_ids = []
                position_ids = []
                hop_ids = []
                time_ids = []

                for neighbor, intimacy_rank, hop, time in neighbors_list:
                    if use_raw_feat:
                        raw_features.append(feats[snap-time][neighbor])
                    else:
                        raw_features.append(None)
                    role_ids.append(wl_dict[neighbor])
                    hop_ids.append(hop)
                    position_ids.append(intimacy_rank)
                    time_ids.append(time)

                raw_features_list.append(raw_features)
                role_ids_list.append(role_ids)
                position_ids_list.append(position_ids)
                hop_ids_list.append(hop_ids)
                time_ids_list.append(time_ids)

            if use_raw_feat:
                raw_embedding = torch.FloatTensor(raw_features_list)
            else:
                raw_embedding = None
            wl_embedding = torch.LongTensor(role_ids_list)
            hop_embedding = torch.LongTensor(hop_ids_list)
            int_embedding = torch.LongTensor(position_ids_list)
            time_embedding = torch.LongTensor(time_ids_list)

            raw_embeddings.append(raw_embedding)
            wl_embeddings.append(wl_embedding)
            hop_embeddings.append(hop_embedding)
            int_embeddings.append(int_embedding)
            time_embeddings.append(time_embedding)

        return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings
    
    def generate_embedding(self, edges):
        num_snap = len(edges)
        WL_dict = self.compute_zero_WL(self.idx,  np.vstack(edges[:10]))
        batch_hop_dicts = self.compute_batch_hop(self.idx, edges, num_snap, self.S, 5, 2)
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = \
            self.dicts_to_embeddings(self.X , batch_hop_dicts, WL_dict, num_snap)
        return raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings
        
    def train(self):
        # self.pos_edges = [torch.tensor(edge, device=self.device, dtype=torch.int64) for edge in self.pos_edges]
        # self.neg_edges = [torch.tensor(edge, device=self.device, dtype=torch.int64) for edge in self.neg_edges]


        # self.pos_edges_index = [edge.to(self.device) for edge in self.pos_edges_index]
        # self.neg_edges_index = [edge.to(self.device) for edge in self.neg_edges_index]
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config['lr'], weight_decay=5e-4)
        raw_embeddings, wl_embeddings, hop_embeddings, int_embeddings, time_embeddings = self.generate_embedding(self.pos_edges)
        raw_embeddings_neg, wl_embeddings_neg, hop_embeddings_neg, int_embeddings_neg, \
            time_embeddings_neg = self.generate_embedding(self.neg_edges)
            
        for e in range(self.train_config['epochs']):

            self.model.train()
            loss_train = 0

            for snap, data in enumerate(self.train_snapshots):
                if wl_embeddings[snap] is None:
                    continue
                
                pos_edge_index = self.pos_edges_index[snap]
                neg_edge_index = self.neg_edges_index[snap]
                
                int_embedding_pos = int_embeddings[snap].to(self.device)
                hop_embedding_pos = hop_embeddings[snap].to(self.device)
                time_embedding_pos = time_embeddings[snap].to(self.device)
                
                int_embedding_neg = int_embeddings_neg[snap].to(self.device)
                hop_embedding_neg = hop_embeddings_neg[snap].to(self.device)
                time_embedding_neg = time_embeddings_neg[snap].to(self.device)

                int_embedding = torch.vstack((int_embedding_pos, int_embedding_neg))
                hop_embedding = torch.vstack((hop_embedding_pos, hop_embedding_neg))
                time_embedding = torch.vstack((time_embedding_pos, time_embedding_neg))
                optimizer.zero_grad() 
                
                z = self.model.forward(int_embedding, hop_embedding, time_embedding).squeeze()
                logits = self.model.decode(z)
                labels = self.get_link_labels(pos_edge_index, neg_edge_index)
                pos_score = logits[:pos_edge_index.size(1)]
                neg_score = logits[pos_edge_index.size(1):]
                
                # if self.train_config['loss'] == 'pairwise':
                #     loss = self.pairwise_loss(pos_score, neg_score)   
                # else:    
                loss = F.binary_cross_entropy(logits, labels.cuda())   
         
                loss.backward()
                optimizer.step()   
                             
                loss_train += loss.detach().item()

            loss_train /= len(self.train_snapshots) - 2 + 1
                
            self.model.eval()
            val_score = self.eval(self.model, self.valid_snapshot, int_embeddings, hop_embeddings, time_embeddings)
            # if val_score[self.train_config['metric']] > self.best_score:
            #     self.patience_knt = 0
            #     self.best_score = val_score[self.train_config['metric']]
            test_score = self.eval2(self.model, self.test_snapshot, int_embeddings, hop_embeddings, time_embeddings)

            print('Epoch {}, Loss {:.4f}, Val AUC {:.4f}, PRC {:.4f}, RecK {:.4f}, test AUC {:.4f}, PRC {:.4f}, RecK {:.4f}'.format(
                e, loss_train, val_score['AUROC'], val_score['AUPRC'], val_score['RecK'],
                test_score['AUROC'], test_score['AUPRC'], test_score['RecK']))
            # else:
            #     self.patience_knt += 1
            #     if self.patience_knt > self.train_config['patience']:
            #         break  
                      
        return test_score  
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    
    @torch.no_grad()
    def eval(self, model, graph, int_embeddings, hop_embeddings, time_embeddings):
        rec, auc, prc, f1 = 0, 0, 0, 0
        for i, data in enumerate(graph):
            num = self.train_len
            result = {}
            int_embedding = int_embeddings[num+i].to(self.device)
            hop_embedding = hop_embeddings[num+i].to(self.device)
            time_embedding = time_embeddings[num+i].to(self.device)
            z = model.forward(int_embedding, hop_embedding, time_embedding, None)
            
            pos_mask = data.y == 0
            neg_mask = data.y == 1

            pos_edge_index = data.edge_index[:, pos_mask]
            neg_edge_index = data.edge_index[:, neg_mask]
            
            link_probs = model.decode(z)
            link_probs = link_probs.cpu() 
            link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)
            link_labels = link_labels.cpu()
            link_preds = (link_probs > 0.5).int()
            

            k = 50
            sorted_indices = np.argsort(link_probs.numpy())[::-1] 
            top_k_indices = sorted_indices[:k]
            
            rec += np.sum(link_labels.numpy()[top_k_indices]) / np.sum(link_labels.numpy()) if np.sum(link_labels.numpy()) != 0 else 0
            auc += roc_auc_score(link_labels.numpy(), link_probs.numpy())
            prc += average_precision_score(link_labels.numpy(), link_probs.numpy())
            f1 += f1_score(link_labels.numpy(), link_preds.numpy(), average='binary')
        
        result['AUROC'] = auc/len(graph)
        result['AUPRC'] = prc/len(graph)
        result['RecK'] = rec/len(graph)
        result['F1'] = f1/len(graph)
            
        return result     

    @torch.no_grad()
    def eval2(self, model, graph, int_embeddings,hop_embeddings, time_embeddings):
        rec, auc, prc, f1 = 0, 0, 0, 0
        for i, data in enumerate(graph):
            num = self.train_len + self.valid_len
            result = {}
            int_embedding = int_embeddings[num+i].to(self.device)
            hop_embedding = hop_embeddings[num+i].to(self.device)
            time_embedding = time_embeddings[num+i].to(self.device)
            z = model.forward(int_embedding, hop_embedding, time_embedding, None)
            
            pos_mask = data.y == 0
            neg_mask = data.y == 1

            pos_edge_index = data.edge_index[:, pos_mask]
            neg_edge_index = data.edge_index[:, neg_mask]
            
            link_probs = model.decode(z)
            link_probs = link_probs.cpu() 
            link_labels = self.get_link_labels(pos_edge_index, neg_edge_index)
            link_labels = link_labels.cpu()
            link_preds = (link_probs > 0.5).int()
            
            k = 50
            sorted_indices = np.argsort(link_probs.numpy())[::-1] 
            top_k_indices = sorted_indices[:k]
            
            rec += np.sum(link_labels.numpy()[top_k_indices]) / np.sum(link_labels.numpy()) if np.sum(link_labels.numpy()) != 0 else 0
            auc += roc_auc_score(link_labels.numpy(), link_probs.numpy())
            prc += average_precision_score(link_labels.numpy(), link_probs.numpy())
            f1 += f1_score(link_labels.numpy(), link_preds.numpy(), average='binary')
        
        result['AUROC'] = auc/len(graph)
        result['AUPRC'] = prc/len(graph)
        result['RecK'] = rec/len(graph)
        result['F1'] = f1/len(graph)
            
        return result     