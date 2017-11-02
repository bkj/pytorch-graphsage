#!/bin/bash

# run.sh

# --
# Small

time ./train.py \
    --problem-path ./data/reddit/problem.h5 \
    --aggregator-class mean

# >>
# Test sparse sampler for reddit
time python ./train.py \
    --problem-path ./data/reddit/sparse-problem.h5 \
    --aggregator-class mean \
    --sampler-class sparse_uniform_neighbor_sampler

time python ./train.py \
    --problem-path ./data/reddit/sparse-full-problem.h5 \
    --aggregator-class mean \
    --sampler-class sparse_uniform_neighbor_sampler

# <<

time ./train.py \
    --problem-path ./data/cora/problem.h5 \
    --aggregator-class mean

time ./train.py \
    --problem-path ./data/pokec/problem.h5 \
    --aggregator-class mean \
    --prep-class node_embedding \
    --epochs 3

# >>
# Sparse sampler for pokec

time ./train.py \
    --problem-path ./data/pokec/problem.h5 \
    --aggregator-class mean \
    --prep-class node_embedding \
    --sampler-class sparse_uniform_neighbor_sampler \
    --epochs 3

# <<

python ./train.py \
    --problem-path ./data/example_data/problem.h5 \
    --aggregator-class mean
