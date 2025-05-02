#!/bin/bash

python main.py -d ICEWS14 --pos_dim 60 --embed_dim 600 \
--temporal_bias 0.01 --solver euler --step_size 0.125 --bs 128 \
--gpu 3 --seed 0 --drop_out 0.15 \
--base_capacity 3 --flexible_capacity 0.3 --path_encode ODE \
