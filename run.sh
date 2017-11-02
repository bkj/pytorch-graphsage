#!/bin/bash

# run.sh

# --
# Small

time ./train.py \
    --problem-path ./data/reddit/problem.h5 \
    --aggregator-class mean

# >>

time python ./train.py \
    --problem-path ./data/reddit/sparse-problem.h5 \
    --aggregator-class mean \
    --sampler-class sparse_uniform_neighbor_sampler

# <<

time ./train.py \
    --problem-path ./data/cora/problem.h5 \
    --aggregator-class mean

time ./train.py \
    --problem-path ./data/pokec/problem.h5 \
    --aggregator-class mean

python ./train.py \
    --problem-path ./data/example_data/problem.h5 \
    --aggregator-class mean
