# BAG: Benchmarking Anomaly Detection on Dynamic Graphs

### This is the official implementation of the following paper:  BAG: Benchmarking Anomaly Detection on Dynamic Graphs

## Data Preprocess
Run ```preprocess_data/preprocess_data.py``` for pre-processing the '.txt' datasets.
For example, to preprocess the *bitcoin_otc* dataset, we can run the following commands:
```{bash}
cd preprocess_data/
python preprocess_data.py  --dataset_name bitcoin_otc
```
## Model Training

For those dataset need to inject anomaly, Run ```benchmark.py```.
For example, to train model *JODIE* on *bitcoin_otc* dataset with anomaly ratio 0.01, we can run the following comands:
```{bash}
python benchmark.py --dataset_name bitcoin_otc --model_name JODIE --num_runs 5 --gpu 0 --anomaly_ratio 0.01
```

For those dataset with anomaly label, for example, to train model *JODIE* on *BGL* dataset, we can run the following comands:
```{bash}
python benchmark.py --dataset_name BGL --model_name JODIE --num_runs 5 --gpu 0 --val_ratio 0.2 --test_ratio 0.3
```
