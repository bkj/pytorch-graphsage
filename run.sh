#!/bin/bash

# run.sh

# --
# Small

python -m train \
    --data-path ./data/example_data/ \
    --aggregator-class mean \
    --multiclass

python -m train \
    --data-path ./data/reddit/ \
    --aggregator-class mean


