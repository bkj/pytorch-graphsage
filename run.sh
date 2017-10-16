#!/bin/bash

# run.sh

# --
# Small

python -m train \
    --data-path ./data/example_data/ \
    --aggregator mean \
    --multiclass

python -m train \
    --data-path ./data/reddit/ \
    --aggregator mean \
    --multiclass


