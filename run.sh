#!/bin/bash

# run.sh

# --
# Small

python -m train \
    --data-path ./data/example_data/ \
    --aggregator-class lstm \
    --multiclass

python -m train \
    --data-path ./data/reddit/ \
    --aggregator-class mean


