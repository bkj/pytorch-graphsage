#!/bin/bash

# example_data (ppi)
python convert.py \
    --inpath ../data/example_data/ \
    --task multilabel_classification

# reddit
python convert.py \
    --inpath ../data/reddit/ \
    --task classification
