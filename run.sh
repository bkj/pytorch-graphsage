#!/bin/bash

# run.sh

# --
# Small

python -m train \
    --data-path ./data/reddit/ \
    --aggregator-class attention \
    --lr-init 0.005 \
    --lr-schedule linear

