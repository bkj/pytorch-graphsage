#!/bin/bash

# pokec.sh

time python ./train.py --problem-path ./data/pokec/sparse-problem.h5 \
    --aggregator-class mean \
    --sampler-class sparse_uniform_neighbor_sampler \
    --prep-class node_embedding --epochs 3


time python ./train.py --problem-path ./data/pokec/problem.h5 \
    --aggregator-class mean \
    --prep-class node_embedding --epochs 3

# {'epoch': 2, 'train_metric': 3.1292589, 'val_metric': 3.7892995, 'time': 147.32675504684448}

