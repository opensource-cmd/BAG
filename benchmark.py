import argparse
import time
from utils import *
import pandas
import os
import warnings
warnings.filterwarnings("ignore")
seed_list = list(range(3407, 10000, 10))

def set_seed(seed=3407):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--trials', type=int, default=10)
parser.add_argument('--models', type=str, default=None)
parser.add_argument('--datasets', type=str, default=None)
parser.add_argument('--anomaly_percents', type=str, default="0.01,0.05,0.1")
parser.add_argument('--loss', type=str, default="pairwise")
args = parser.parse_args()

columns = ['name']
new_row = {}
datasets = ['digg', 'uci', 'bit_alpha', 'bit_otc', 'as_topology',
            'email_dnc','epinions','elliptic', 'wiki', 'reddit', 'yelp', 'bgl']
ratios = [0.1, 0.05, 0.001]
models = model_detector_dict.keys()
anomaly_percents = [float(x) for x in args.anomaly_percents.split(',')]

if args.datasets is not None:
    if '-' in args.datasets:
        st, ed = args.datasets.split('-')
        datasets = datasets[int(st):int(ed)+1]
    else:
        datasets = [datasets[int(t)] for t in args.datasets.split(',')]
    print('Evaluated Datasets: ', datasets)

if args.models is not None:
    if '-' in args.models:
        st, ed = args.models.split('-')
        models = models[int(st):int(ed)+1] 
    else:
        models = [models[int(t)] for t in args.models.split(',')]
    print('Evaluated Baselines: ', models)
    print('Loss: ', args.loss)

# if args.anomaly_percent is not None:
#     anomaly_percent = str(args.anomaly_percent)
    
for dataset in datasets:
    for metric in ['AUROC mean', 'AUROC std', 'AUPRC mean', 'AUPRC std',
                   'RecK mean', 'RecK std', 'Time']:
        columns.append(dataset+'-'+metric)

results = pandas.DataFrame(columns=columns)
file_id = None

for model in models:
    model_result = {'name': model}
    for dataset_name in datasets:
        if dataset_name in ('reddit','wiki', 'yelp', 'bgl'):
            time_cost = 0
            data = Dataset(dataset_name)
            train_config = {
            'device': 'cuda',
            'epochs': 200,
            'patience': 50,
            'metric': 'AUROC',
            'loss': 'bce',
            'label': True,
            }
            model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0.1}
            
            auc_list, pre_list, rec_list, f1_list = [], [], [], []
            for t in range(args.trials):
                torch.cuda.empty_cache()
                print("Dataset {}, Model {}, Trial {}".format(dataset_name, model, t))
                
                data_path = os.path.join('data/static', dataset_name, f"{dataset_name}.pt") 
                snap_path = os.path.join('data/snapshot', dataset_name) 
                valid_snap_path = os.path.join(snap_path, f"{dataset_name}_valid.pt")
                
                if model in ('GCN', 'GraphSAGE','GIN','ChebNet','SGC','GT', 'GAT', 'Node2vec','Deepwalk'):
                    if os.path.exists(data_path):
                        static_graph = torch.load(data_path)
                    else:
                        static_graph = data.load_static_graph(anomaly_percent=None)
                    detector = model_detector_dict[model](train_config, model_config, static_graph)
                else:
                    if os.path.exists(valid_snap_path):
                        train_snapshots = torch.load(os.path.join(snap_path, f"{dataset_name}_train.pt"))
                        valid_snapshots = torch.load(valid_snap_path)
                        test_snapshots = torch.load(os.path.join(snap_path, f"{dataset_name}_test.pt"))
                    else:    
                        train_snapshots, test_snapshots, valid_snapshots = data.snapshot_setting(
                            anomaly_percent=0, n_clusters=42, random_seed=0)
                    detector = model_detector_dict[model](train_config, model_config, train_snapshots, valid_snapshots, test_snapshots)

                seed = seed_list[t]
                set_seed(seed)
                train_config['seed'] = seed
                                    
                st = time.time()
                test_score = detector.train()
                auc_list.append(test_score['AUROC'])
                pre_list.append(test_score['AUPRC'])
                rec_list.append(test_score['RecK'])
                f1_list.append(test_score['F1'])
                
                ed = time.time()
                time_cost += ed - st
                
                del detector
                model_result.update({
                    f"{dataset_name}-AUROC mean": np.mean(auc_list),
                    f"{dataset_name}-AUROC std": np.std(auc_list),
                    f"{dataset_name}-AUPRC mean": np.mean(pre_list),
                    f"{dataset_name}-AUPRC std": np.std(pre_list),
                    f"{dataset_name}-RecK mean": np.mean(rec_list),
                    f"{dataset_name}-RecK std": np.std(rec_list),
                    f"{dataset_name}-F1 mean": np.mean(f1_list),
                    f"{dataset_name}-F1 std": np.std(f1_list),
                    f"{dataset_name}-Time": time_cost / args.trials
                })
        else:
            for anomaly_percent in anomaly_percents:
                anomaly_percent_str = str(anomaly_percent)
                time_cost = 0
                if dataset_name in ('elliptic'):
                    train_config = {
                    'device': 'cuda',
                    'epochs': 500,
                    'patience': 50,
                    'metric': 'AUROC',
                    'loss': args.loss,
                    'label': False,
                    }
                    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0.1, 'hidden_dim':16}
                else:
                    data = Dataset(dataset_name)
                    train_config = {
                    'device': 'cuda',
                    'epochs': 200,
                    'patience': 100,
                    'metric': 'AUROC',
                    'loss': args.loss,
                    'label': False,
                    }
                    model_config = {'model': model, 'lr': 0.01, 'drop_rate': 0.1}
                
                auc_list, pre_list, rec_list, f1_list = [], [], [], []
                for t in range(args.trials):
                    torch.cuda.empty_cache()
                    print("Dataset {}, Anomaly Percent {}, Model {}, Trial {}".format(dataset_name,anomaly_percent, model, t))
                    
                    data_path = os.path.join('data/static', dataset_name, f"{dataset_name}_{anomaly_percent}.pt") 
                    snap_path = os.path.join('data/snapshot', dataset_name) 
                    valid_snap_path = os.path.join(snap_path, f"{dataset_name}_{anomaly_percent}_valid.pt")
                    
                    if model in ('GCN', 'GraphSAGE','GIN','ChebNet','SGC','GT', 'GAT', 'Node2vec','Deepwalk'):
                        if os.path.exists(data_path):
                            static_graph = torch.load(data_path)
                        else:
                            static_graph = data.train_test_split(anomaly_percent=anomaly_percent)
                        detector = model_detector_dict[model](train_config, model_config, static_graph)
                    else:
                        if os.path.exists(valid_snap_path):
                            train_snapshots = torch.load(os.path.join(snap_path, f"{dataset_name}_{anomaly_percent}_train.pt"))
                            valid_snapshot = torch.load(valid_snap_path)
                            test_snapshot = torch.load(os.path.join(snap_path, f"{dataset_name}_{anomaly_percent}_test.pt"))
                        else:    
                            train_snapshots, test_snapshot, valid_snapshot = data.snapshot_setting(
                                anomaly_percent=anomaly_percent, n_clusters=42, random_seed=0)
                        detector = model_detector_dict[model](train_config, model_config, train_snapshots, valid_snapshot, test_snapshot)

                    seed = seed_list[t]
                    set_seed(seed)
                    train_config['seed'] = seed
                                        
                    st = time.time()
                    test_score = detector.train()
                    auc_list.append(test_score['AUROC'])
                    pre_list.append(test_score['AUPRC'])
                    rec_list.append(test_score['RecK'])
                    f1_list.append(test_score['F1'])
                    
                    ed = time.time()
                    time_cost += ed - st
                    
                if dataset_name in ('elliptic'):
                    del detector
                else:
                    del detector, data
                
                model_result.update({
                    f"{dataset_name}-{anomaly_percent_str}-AUROC mean": np.mean(auc_list),
                    f"{dataset_name}-{anomaly_percent_str}-AUROC std": np.std(auc_list),
                    f"{dataset_name}-{anomaly_percent_str}-AUPRC mean": np.mean(pre_list),
                    f"{dataset_name}-{anomaly_percent_str}-AUPRC std": np.std(pre_list),
                    f"{dataset_name}-{anomaly_percent_str}-RecK mean": np.mean(rec_list),
                    f"{dataset_name}-{anomaly_percent_str}-RecK std": np.std(rec_list),
                    f"{dataset_name}-{anomaly_percent_str}-F1 mean": np.mean(f1_list),
                    f"{dataset_name}-{anomaly_percent_str}-F1 std": np.std(f1_list),
                    f"{dataset_name}-{anomaly_percent_str}-Time": time_cost / args.trials
                })

    model_result = pandas.DataFrame(model_result, index=[0])
    results = pandas.concat([results, model_result])
    file_id = save_results(results, file_id)
    print(results)