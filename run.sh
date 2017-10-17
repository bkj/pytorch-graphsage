#!/bin/bash

# run.sh

# --
# Small

python ./train.py \
    --problem-path ./data/reddit/problem.h5 \
    --aggregator-class mean

python ./train.py \
    --problem-path ./data/example_data/problem.h5 \
    --aggregator-class mean
