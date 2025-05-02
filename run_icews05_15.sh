#!/bin/bash

python main.py -d ICEWS05_15 --pos_dim 60 --embed_dim 600 \
--temporal_bias 0.001 --solver euler --step_size 0.125 --bs 128 \
--gpu 6 --seed 0 --drop_out 0.15 \
--base_capacity 8 --flexible_capacity 0.6 --path_encode ODE \
