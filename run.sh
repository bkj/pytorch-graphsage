#!/bin/bash

# run.sh

# --
# Supervised

python -m train \
    --data-path ./data/reddit/ \
    --aggregator-class mean

# --
# Unsupervised

python -m utrain \
    --data-path ./data/reddit/ \
    --aggregator-class mean