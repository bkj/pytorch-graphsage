#!/bin/bash

# pokec.sh

time python ./train.py --problem-path ./data/pokec/problem.h5 --aggregator-class mean \
    --prep-class node_embedding --epochs 3
