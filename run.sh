#!/bin/bash

# run.sh

# --
# Small

time python ./train.py \
    --problem-path ./data/reddit/problem.h5 \
    --aggregator-class mean

time python ./train.py \
    --problem-path ./data/reddit/problem.h5 \
    --aggregator-class mean \
    --prep-class embedding 


python ./train.py \
    --problem-path ./data/example_data/problem.h5 \
    --aggregator-class mean
