#!/bin/bash

python main.py -d socialTKG --pos_dim 60 --embed_dim 600 \
--temporal_bias 0.01 --solver euler --step_size 0.125 --bs 128 \
--gpu 3 --seed 0 --default_weight 0.3 \
--base_capacity 2 --flexible_capacity 0.6 --path_encode ODE \
