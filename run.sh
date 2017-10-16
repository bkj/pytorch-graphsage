#!/bin/bash

# run.sh

python -m graphsage.supervised_train \
    --train_prefix ./data/example_data/ppi \
    --model graphsage_mean \
    --sigmoid


python -m graphsage.supervised_train \
    --train_prefix ./data/reddit/reddit \
    --model graphsage_mean \
    --sigmoid

# --
# Pytorch 

python -m graphsage.train \
    --train_prefix ./data/example_data/ppi \
    --model graphsage_mean \
    --sigmoid
