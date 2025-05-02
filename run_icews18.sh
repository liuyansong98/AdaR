#!/bin/bash

python main.py -d ICEWS18 --pos_dim 60 --embed_dim 600 \
--temporal_bias 0.001 --solver euler --step_size 0.125 --bs 128 \
--gpu 2 --seed 0 --default_weight 0.3 \
--base_capacity 8 --flexible_capacity 0.6 --path_encode ODE \
