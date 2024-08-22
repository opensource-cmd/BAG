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
                'reddit': lambda: pd.read_csv('data/reddit.csv').rename(columns={'u': 'source', 'i': 'target', 'ts': 'timestamp', 'label': 'label'}),
                'wiki': lambda: pd.read_csv('data/wiki.csv').rename(columns={'u': 'source', 'i': 'target', 'ts': 'timestamp', 'label': 'label'}),
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
            
    def process_data(self, ):
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
        data['source'], data['target'] = np.where(data['source'] < data['target'],
                                                  (data['source'], data['target']),
                                                  (data['target'], data['source']))
        data = data.sort_values(by='timestamp').drop(columns=['timestamp']).reset_index(drop=True)  ## 按照 timestamp 列排序、删除 timestamp 列并重置索引
        # data = data[data['source'] != data['target']] 
        
        edges_array = data[['source', 'target']].values.flatten()
        _, edges_indices = np.unique(edges_array, return_inverse=True)
        edges = np.reshape(edges_indices, [-1, 2])
        nodes, _ = np.unique(edges, return_inverse=True)
        data[['source', 'target']] = edges
        
        # Split the data   
        if self.name in ['reddit', 'wiki', 'yelp', 'bgl']:
            data = data.loc[:, ['source', 'target', 'label']] 
            total_rows = len(data)
            train_end = int(total_rows * 0.5)
            valid_end = train_end + int(total_rows * 0.2)

        elif self.name in ['bgl']:   
            data = data.loc[:, ['source', 'target', 'label']]  
            total_rows = len(data)
            train_end = int(total_rows * 0.5)
            valid_end = train_end + int(total_rows * 0.2)

        else: 
            data['label'] = 0 
            data = data.loc[:, ['source', 'target', 'label']] 
            total_rows = len(data)
            train_end = int(total_rows * 0.8)
            valid_end = train_end + int(total_rows * 0.1)
            
        train_data = data.iloc[:train_end]
        valid_data = data.iloc[train_end:valid_end]
        test_data = data.iloc[valid_end:]
        
        return data, nodes, edges, train_data, valid_data, test_data


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
        
    def split_snapshot(self, dataframe, num):
        snapshots = []
        snapshot_size = len(dataframe)//num
        for i in range(num):
            start = i * snapshot_size
            if i == num-1:
                snapshot = dataframe[start:]
            else:
                snapshot = dataframe[start:start + snapshot_size]
            snapshots.append(snapshot)
        return snapshots
                    
    def edgeList2Adj(self, data):
        max_node = max(data['source'].max(), data['target'].max())
        adjacency_matrix = np.zeros((max_node + 1, max_node + 1), dtype=int)
        for _, row in data.iterrows():
            source = row['source']
            target = row['target']
            adjacency_matrix[source, target] = 1
            adjacency_matrix[target, source] = 1  
        return adjacency_matrix

    def inject_anomalies(self, data, nodes, anomaly_percent, n_clusters=10, random_seed=0):
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

        num_anomalies = max(1, math.ceil(anomaly_percent * m))   ## make sure at least one
        anomalies = fake_edges[:min(num_anomalies, len(fake_edges)), :]
        anomalies_df = pd.DataFrame(anomalies, columns=['source', 'target'])
        anomalies_df['label'] = 1
        combined_data = pd.concat([data, anomalies_df])
        
        return combined_data


    def generate_graph(self, data, nodes, feature_dim=10):
        node_features = torch.ones((len(nodes), feature_dim))
        data = data.drop_duplicates(subset=['source', 'target'])
        edge_index = torch.tensor(data[['source', 'target']].values.T, dtype=torch.long).contiguous()
        y = torch.tensor(data['label'].values, dtype=torch.long)
        return Data(x=node_features, edge_index=edge_index, y=y)
        
    def generate_snapshots(self, nodes, data, is_training, feature_dim=10):
        node_features = torch.ones((len(nodes), feature_dim))

        def create_snapshot_graph(snapshot):
            snapshot = snapshot.drop_duplicates(subset=['source', 'target'])
            edge_index = torch.tensor(snapshot[['source', 'target']].values.T, dtype=torch.long).contiguous()
            y = torch.tensor(snapshot['label'].values, dtype=torch.long)
            return Data(x=node_features, edge_index=edge_index, y=y)

        if is_training:
            snapshots = []
            snapshot_size = len(data) // 8
            for i in range(8):
                start = i * snapshot_size
                if i == 7:  # For the last snapshot, include all remaining data
                    snapshot = data[start:]
                else:
                    snapshot = data[start:start + snapshot_size]
                snapshots.append(create_snapshot_graph(snapshot))
        else:
            snapshots = create_snapshot_graph(data)


    def generate_snapshots(self, nodes, data, is_train):
        if is_train:
            snapshots = []
            snapshot_size = len(data)//12
            for i in range(12):
                start = i * snapshot_size
                if i == 11:
                    snapshot = data[start:]
                else:
                    snapshot = data[start:start + snapshot_size]
                snapshots.append(self.generate_graph(snapshot, nodes))
        else:
            snapshots = self.generate_graph(data, nodes)

        return snapshots
    
    def snapshot_setting(self, anomaly_percent, n_clusters=42, random_seed=0):
        _, nodes, _, train_data, valid_data, test_data = self.process_data()
        valid_snapshot = self.split_snapshot(valid_data, 4)
        test_snapshot = self.split_snapshot(test_data, 6)
        if self.name in ['reddit', 'wiki', 'yelp','bgl']:
            train_data = train_data
            valid_data = valid_snapshot 
            test_data = test_snapshot
            
            train_snapshots = self.generate_snapshots(nodes, train_data, is_train=True)
            valid_snapshots = [self.generate_snapshots(nodes, data, is_train=False) for data in valid_data]
            test_snapshots = [self.generate_snapshots(nodes, data, is_train=False) for data in test_data]
            
            directory_path = os.path.join('data/snapshot', self.name)
            os.makedirs(directory_path, exist_ok=True)
            paths = {
                'train': os.path.join(directory_path, f"{self.name}_train.pt"),
                'valid': os.path.join(directory_path, f"{self.name}_valid.pt"),
                'test': os.path.join(directory_path, f"{self.name}_test.pt")
            }

        else:    
            train_data = train_data
            valid_data = [self.inject_anomalies(data, anomaly_percent, n_clusters, random_seed) for data in valid_snapshot]
            test_data = [self.inject_anomalies(data, anomaly_percent, n_clusters, random_seed) for data in test_snapshot]
        
            # valid_data = [data.drop_duplicates(subset=['source', 'target']) for data in valid_data]
            # test_data = [data.drop_duplicates(subset=['source', 'target']) for data in test_data]
            train_snapshots = self.generate_snapshots(dataframe=train_data, all_nodes=nodes, is_train=True)
            valid_snapshots = [self.generate_snapshots(dataframe=data, all_nodes=nodes, is_train=False) for data in valid_data]
            test_snapshots = [self.generate_snapshots(dataframe=data, all_nodes=nodes, is_train=False) for data in test_data]
            
            directory_path = os.path.join('data/snapshot_bit_alpha/1', self.name)
            os.makedirs(directory_path, exist_ok=True)
            paths = {
                'train': os.path.join(directory_path, f"{self.name}_{anomaly_percent}_train.pt"),
                'valid': os.path.join(directory_path, f"{self.name}_{anomaly_percent}_valid.pt"),
                'test': os.path.join(directory_path, f"{self.name}_{anomaly_percent}_test.pt")
            }

        torch.save(train_snapshots, paths['train'])
        torch.save(valid_snapshots, paths['valid'])
        torch.save(test_snapshots, paths['test'])
        
        return train_snapshots, test_snapshots, valid_snapshots


model_detector_dict = {
    
    # Traditional Methods
    # 'Deepwalk': DeepwalkDetector,
    # 'Node2vec': N2vDetector,
    'Spectral Clustering': staticGNNDetector,
    'SDNE': staticGNNDetector,
    'GOutlier': staticGNNDetector,
    'CM_Sketch': staticGNNDetector,

    # Static GNNs
    'GCN': staticGNNDetector,
    'GraphSAGE': staticGNNDetector,
    'ChebNet': staticGNNDetector,
    'GIN': staticGNNDetector,
    'SGC': staticGNNDetector,
    'GT': staticGNNDetector,    
    'GAT': staticGNNDetector,

    # Specialized GNNs
    'AddGraph': SnapshotDetector,
    'NetWalk': DTDGDetector,
    'StrGNN': SnapshotDetector,
    'TADDY': taddyDetector,
    'ANOMULY': SnapshotDetector,
    'SingleANOMULY': SnapshotDetector,
    'RegraphGAN': DTDGDetector,
    'THGNN': DTDGDetector,
    'RustGraph': SnapshotsDetector,

    # Discrete-time temporal GNNs
    'Dy_GrEncoder': DTDGDetector,
    'T_GCN': DTDGDetector,
    'EvolveGCN_O': DTDGDetector,   
    'EvolveGCN_H': DTDGDetector,
    'MPNN_LSTM': DTDGDetector,
    'WD_GCN':SnapshotDetector,
    'CD_GCN':SnapshotDetector,
    # 'AGC_RN': Agcrndetector,                
    # 'GConv_GRU': SnapshotDetector,
    # 'DC_RNN': SnapshotDetector,
    # 'GC_LSTM': SnapshotDetector,
    # 'LR_GCN': SnapshotDetector,
    # 'T_GCN2': SnapshotDetector,
    # 'A3T_GCN':SnapshotDetector,

    # Continuous-time temporal graph GNNs
    'JODIE': CTTGDetector,
    'DyRep': CTTGDetector,
    'TGN': CTTGDetector,
    'TGAT': CTTGDetector,
    'TCL': CTTGDetector,
    'CAWN': CTTGDetector,
    'GraphMixer': CTTGDetector,
    'DyGFormer': CTTGDetector,
    'FreeDyG': CTTGDetector,

}


def save_results(results, file_id):
    if not os.path.exists('results/'):
        os.mkdir('results/')
    if file_id is None:
        file_id = 0
        while os.path.exists('results/{}.xlsx'.format(file_id)):
            file_id += 1
    results.transpose().to_excel('results/{}.xlsx'.format(file_id))
    print('save to file ID: {}'.format(file_id))
    return file_id


def sample_param(model, dataset, t=0):
    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0}
    if t == 0:
        return model_config
    for k, v in param_space[model].items():
        model_config[k] = random.choice(v)
        
    return model_config


param_space = {}

param_space['GCN'] = {
    'hidden_dim': [16, 32,64],
    'num_layers': [1, 2, 3, 4],
    'dropout_rate': [0, 0.1, 0.2, 0.3],
    'lr': 10 ** np.linspace(-3, -1, 1000),
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['AddGraph'] = {
    'hidden_dim': [16, 32,64],
    'num_layers': [1, 2, 3, 4],
    'dropout_rate': [0, 0.1, 0.2, 0.3],
    'activation': ['ReLU', 'LeakyReLU', 'Tanh']
}

param_space['Node2vec'] = {

}


param_space['GC_LSTM'] = {
    'hidden_dim': [32,64],
    'K': [1,3,5,7]

}


param_space['DC_RNN'] = {

}
