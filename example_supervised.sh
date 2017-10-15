#!/bin/bash

# example_supervised.dh

# --
# GCN

python -m graphsage.supervised_train \
    --train_prefix ./data/reddit/reddit \
    --model graphsage_mean \
    --sigmoid

# Full validation stats: loss= 0.01206 f1_micro= 0.92721 f1_macro= 0.87990 time= 2.08346

# --
# Graphsage_mean

python -m graphsage.supervised_train \
    --train_prefix ./data/example_data/ppi \
    --model graphsage_mean \
    --sigmoid

# Full validation stats: loss= 0.00894 f1_micro= 0.94603 f1_macro= 0.91576 time= 1.90992

# --
# Graphsage_seq

python -m graphsage.supervised_train \
    --train_prefix ./data/reddit/reddit \
    --model graphsage_seq \
    --sigmoid \
    --learning_rate 0.001