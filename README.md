
## Introduction
**[submit to TKDE ]** 

<p align="center">
<img src="./intro.pdf" width="800">
</p>
## Authors

anonymity

## Requirements
- numba==0.54.1
- numpy==1.19.2
- pandas==1.2.2
- scikit_learn==1.1.1
- torch==1.7.1
- torchdiffeq==0.2.2
- tqdm==4.59.0

## Model training
 ```./run_icews14.sh```
 ```./run_icews05-15.sh```
  ```./run_icews18.sh```
 ```./run_gdelt.sh```
 ```./run_social_TKG.sh```

## Model Testing

default_weight and test_model_path need to be set according to yourself training situation

```python test.py -d ICEWS14 --pos_dim 60 --embed_dim 600 --temporal_bias 0.01 --solver euler --step_size 0.125 --bs 128 --gpu 0 --seed 0 --default_weight 0.3 --test_model_path your_model_path```

```python test.py -d ICEWS05_15 --pos_dim 60 --embed_dim 600 --temporal_bias 0.001 --solver euler --step_size 0.125 --bs 128 --gpu 0 --seed 0 --default_weight 0.3 --test_model_path your_model_path```

```python test.py -d ICEWS18 --pos_dim 60 --embed_dim 600 --temporal_bias 0.001 --solver euler --step_size 0.125 --bs 128 --gpu 0 --seed 0 --default_weight 0.3 --test_model_path your_model_path```

```python main.py -d GDELT --pos_dim 60 --embed_dim 600 --temporal_bias 0.001 --solver euler --step_size 0.125 --bs 512 --128 0 --seed 0 --default_weight 0.3 --test_model_path your_model_path```

```python main.py -d social_TKG --pos_dim 60 --embed_dim 600 --temporal_bias 0.001 --solver rk4 --step_size 0.25 --bs 512 --gpu 0 --seed 0 --default_weight 0.3 --test_model_path your_model_path```
