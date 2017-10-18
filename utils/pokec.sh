#!/bin/bash

# pokec.sh

time python ./train.py --problem-path ./data/pokec/problem.h5 --aggregator-class max_pool --prep-class node_embedding
# {'epoch': 9, 'train_metric': 1.3477343, 'val_metric': 4.0626483}
# 11m

time python ./train.py --problem-path ./data/pokec/problem.h5 --aggregator-class mean_pool --prep-class node_embedding
# {'epoch': 9, 'train_metric': 1.377148, 'val_metric': 3.955126}
# 12m

time python ./train.py --problem-path ./data/pokec/problem.h5 --aggregator-class mean --prep-class node_embedding
# {'epoch': 9, 'train_metric': 1.2615484, 'val_metric': 3.9707122}
# 8m

time python ./train.py --problem-path ./data/pokec/problem.h5 --aggregator-class attention --prep-class node_embedding
# {'epoch': 9, 'train_metric': 2.1607544, 'val_metric': 4.5276971}
# 11m