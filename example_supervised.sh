#!/bin/bash

# example_supervised.dh

python -m graphsage.supervised_train \
    --train_prefix ./data/reddit/reddit \
    --model graphsage_gcn \
    --sigmoid
