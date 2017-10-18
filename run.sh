#!/bin/bash

# run.sh

# --
# Small

time python ./train.py \
    --problem-path ./data/reddit/problem.h5 \
    --aggregator-class mean

time python ./train.py \
    --problem-path ./data/cora/problem.h5 \
    --aggregator-class mean


python ./train.py \
    --problem-path ./data/example_data/problem.h5 \
    --aggregator-class mean
