#!/bin/bash

# run.sh

# --
# Small

python -m train \
    --data-path ./data/reddit/ \
    --aggregator-class mean \
    --lr-init 0.005


