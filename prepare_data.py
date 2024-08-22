from models.detector import *
import torch
import os
import random
import numpy as np
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from sklearn.cluster import SpectralClustering
from io import StringIO
import argparse

class Dataset:
    def __init__(self, name='uci', prefix='./data/'):
        self.name = name
        self.prefix = prefix
        self.full_path = f"{prefix}{name}"
        self.data = self.load_data()

    def load_csv_with_utf8sig(self):
        with open(self.full_path, 'r', encoding='utf-8-sig') as file:
            content = file.read()
        data_io = StringIO(content)
        return np.loadtxt(data_io, delimiter=',', dtype=float)
    
    def load_data(self):
        try:
            loaders = {
                'bit_alpha': lambda: np.loadtxt(self.full_path, dtype=float, comments='%', delimiter='\t'),
                'bit_otc': lambda: np.loadtxt(self.full_path, dtype=float, comments='%', delimiter='\t'),
                'email_dnc': self.load_csv_with_utf8sig,
                'elliptic': self.load_elliptic_data,
                'reddit': lambda: pd.read_csv(self.full_path).rename(columns={'u': 'source', 'i': 'target', 'ts': 'timestamp', 'label': 'label'}),
                'wiki': lambda: pd.read_csv(self.full_path).rename(columns={'u': 'source', 'i': 'target', 'ts': 'timestamp', 'label': 'label'}),
                'bgl': lambda: pd.read_csv(self.full_path, header=None, names=['label', 'timestamp', 'source', 'target']),
                'yelp': lambda: pd.read_csv(self.full_path, delim_whitespace=True, header=None)
            }
            return loaders.get(self.name, lambda: np.loadtxt(self.full_path, dtype=float, comments='%', delimiter=' '))()
        
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def load_elliptic_data(self):
        edges = pd.read_csv(f"{self.prefix}elliptic/elliptic_txs_edgelist.csv")
        features = pd.read_csv(f"{self.prefix}elliptic/elliptic_txs_features.csv")
        result = pd.merge(edges, features[['txId', 'snap']], left_on='txId1', right_on='txId', how='left')
        return result.rename(columns={'snap': 'timestamp'})
            
    def process_data(self):
        if self.name in ['wiki', 'reddit']:
            data = pd.DataFrame(self.data, columns=['source', 'target', 'timestamp', 'label'])
        elif self.name in ['yelp']:
            self.data.columns=['source', 'target', 'weight', 'label', 'timestamp']
            data = self.data
            data['label'] = data['label'].replace({-1: 1, 1: 0})   
        elif self.name in ['bgl']:
            data = self.data
            data['label'] = data['label'].replace('-', 0) 
            data['label'] = data['label'].replace({x: 1 for x in set(data['label']) - {0}})   
        else:
            data = pd.DataFrame(self.data, columns=['source', 'target', 'weight', 'timestamp'])
            
        data[['source', 'target']] = data[['source', 'target']].astype(int)
        data['source'], data['target'] = np.where(data['source'] < data['target'],     ## make sure source node < target node
                                                  (data['source'], data['target']),
                                                  (data['target'], data['source']))
        data = data.sort_values(by='timestamp').drop(columns=['timestamp']).reset_index(drop=True)  ## sort by timestamp and reset column index
        # data = data[data['source'] != data['target']] 
        
        # Re-index nodes and edges
        edges_array = data[['source', 'target']].values.flatten()
        _, edges_indices = np.unique(edges_array, return_inverse=True)
        edges = np.reshape(edges_indices, [-1, 2])
        nodes, _ = np.unique(edges, return_inverse=True)
        data[['source', 'target']] = edges

        if self.name not in['reddit', 'wiki', 'yelp', 'bgl']:
           data['label'] = 0 
        data = data.loc[:, ['source', 'target', 'label']]
        
        return data, nodes, edges

    def generate_anomaly(self, data, nodes, n_clusters=10, random_seed=0):
        normal_edges = data[['source', 'target']].values
        n = len(nodes)
        m = len(normal_edges)
        adjacency_matrix = np.zeros((n, n), dtype=int)
        for _, row in data.iterrows():
            source = row['source']
            target = row['target']
            adjacency_matrix[source, target] = 1
            adjacency_matrix[target, source] = 1  
        sc = SpectralClustering(n_clusters, affinity='precomputed', n_init=10, assign_labels='discretize', n_jobs=-1)
        labels = sc.fit_predict(adjacency_matrix)

        np.random.seed(random_seed)
        idx_1 = np.random.choice(n, m)
        idx_2 = np.random.choice(n, m)
        generated_edges = np.column_stack((idx_1, idx_2))
        existing_edges = set(map(tuple, data[['source', 'target']].values))
        fake_edges = np.array([x for x in generated_edges if (x[0], x[1]) not in existing_edges and labels[x[0]] != labels[x[1]]])

        return fake_edges
    
    def load_static_graph(self, anomaly_percents, val_ratio=0.1, test_ratio=0.1, feature_dim=10):
        data, nodes, edges = self.process_data()
        data = data.drop_duplicates(subset=['source', 'target'])  

        fake_edges = self.generate_anomaly(data, nodes)
        node_features = torch.ones((len(nodes), feature_dim))
        edge_index = torch.from_numpy(edges)
        y = torch.tensor(data['label'].tolist(), dtype=torch.long)
        graph = Data(x=node_features, edge_index=edge_index, y=y)
        
        row, col = graph.edge_index
        graph.edge_index = None

        # Return upper triangular portion.
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        
        for anomaly_percent in anomaly_percents:
            ano_num = int(anomaly_percent * n_t) * 2  
                
            # Positive edges.
            perm = torch.randperm(row.size(0))
            row, col = row[perm], col[perm]

            r, c = row[:n_v], col[:n_v]
            graph.val_pos_edge_index = torch.stack([r, c], dim=0)
            r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
            graph.test_pos_edge_index = torch.stack([r, c], dim=0)

            r, c = row[n_v + n_t:], col[n_v + n_t:]
            graph.train_pos_edge_index = torch.stack([r, c], dim=0)
                
            # Negative edges.
            anomalies = fake_edges[:int(np.floor(ano_num)), :]
            anomalies_edge_index = torch.tensor(anomalies.T, dtype=torch.long)
            
            graph.val_neg_edge_index = anomalies_edge_index[:, :ano_num//2]
            graph.test_neg_edge_index = anomalies_edge_index[:, ano_num//2:]
            
            directory_path= os.path.join('data/static', self.name)
            data_path = os.path.join(directory_path, self.name + '_' + str(anomaly_percent) + '.pt')
            os.makedirs(directory_path, exist_ok=True)
            torch.save(graph, data_path)
        
        return graph


    def load_static_graph(self, anomaly_percents, val_ratio=0.1, test_ratio=0.1, feature_dim=10):
        data, nodes, edges = self.process_data()
        data = data.drop_duplicates(subset=['source', 'target'])  

        fake_edges = self.generate_anomaly(data, nodes)
        node_features = torch.ones((len(nodes), feature_dim))
        edge_index = torch.from_numpy(edges)
        y = torch.tensor(data['label'].tolist(), dtype=torch.long)
        graph = Data(x=node_features, edge_index=edge_index, y=y)
        
        row, col = graph.edge_index
        graph.edge_index = None

        # Return upper triangular portion.
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        
        for anomaly_percent in anomaly_percents:
            ano_num = int(anomaly_percent * n_t) * 2  
                
            # Positive edges.
            perm = torch.randperm(row.size(0))
            row, col = row[perm], col[perm]

            r, c = row[:n_v], col[:n_v]
            graph.val_pos_edge_index = torch.stack([r, c], dim=0)
            r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
            graph.test_pos_edge_index = torch.stack([r, c], dim=0)

            r, c = row[n_v + n_t:], col[n_v + n_t:]
            graph.train_pos_edge_index = torch.stack([r, c], dim=0)
                
            # Negative edges.
            anomalies = fake_edges[:int(np.floor(ano_num)), :]
            anomalies_edge_index = torch.tensor(anomalies.T, dtype=torch.long)
            
            graph.val_neg_edge_index = anomalies_edge_index[:, :ano_num//2]
            graph.test_neg_edge_index = anomalies_edge_index[:, ano_num//2:]
            
            directory_path= os.path.join('data/static', self.name)
            data_path = os.path.join(directory_path, self.name + '_' + str(anomaly_percent) + '.pt')
            os.makedirs(directory_path, exist_ok=True)
            torch.save(graph, data_path)
        
        return graph


if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Load a static graph and generate anomalies.")
    parser.add_argument("--name", type=str, required=True, help="The name of the dataset")
    parser.add_argument("--anomaly_percents", type=float, nargs='+', required=True, help="List of anomaly percentages")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation set ratio")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="Test set ratio")
    parser.add_argument("--feature_dim", type=int, default=10, help="Feature dimension of the nodes")

    args = parser.parse_args()
    
    dataset = Dataset(name=args.name, prefix='./data/')
    graph = dataset.load_static_graph(anomaly_percents=args.anomaly_percents, val_ratio=args.val_ratio, test_ratio=args.test_ratio, feature_dim=args.feature_dim)
    