#!/bin/bash

# run.sh

# --
# Small

python -m train \
    --data-path ./data/example_data/ \
    --aggregator-class attention \
    --multiclass

python -m train \
    --data-path ./data/reddit/ \
    --aggregator-class attention \
    --lr-init 0.001 \
    --lr-schedule linear


