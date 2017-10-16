#!/usr/bin/env python

"""
    supervised_train.py
"""

from __future__ import division
from __future__ import print_function

import os
import time
import sklearn
import numpy as np
import argparse
from sklearn import metrics

import torch
from torch.autograd import Variable

from graphsage.data_loader import NodeDataLoader
from graphsage.pytorch_models import SupervisedGraphsage

# --
# Helpers

def set_seeds(seed=0):
    np.random.seed(seed)
    _ = torch.manual_seed(seed)
    _ = torch.cuda.manual_seed(seed)

def to_numpy(x):
    if isinstance(x, Variable):
        return to_numpy(x.data)
    
    return x.cpu().numpy() if x.is_cuda else x.numpy()

def uniform_neighbor_sampler(ids, adj, n_samples=-1):
    # !! Should do "random gather" instead
    tmp = adj[ids]
    tmp = tmp[:,torch.randperm(tmp.size(1)).cuda()]
    return tmp[:,:n_samples]


def calc_f1(y_true, y_pred, sigmoid=True):
    if not sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = (y_pred > 0).astype(int)
    
    return {
        "micro" : metrics.f1_score(y_true, y_pred, average="micro"),
        "macro" : metrics.f1_score(y_true, y_pred, average="macro"),
    }


# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--train_prefix', required=True)
    parser.add_argument('--model', default='graphsage_mean')
    
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--model_size', default="small")
    
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--dropout', default=0.0)
    parser.add_argument('--weight_decay', default=0.0)
    parser.add_argument('--max_degree', default=128)
    parser.add_argument('--samples_1', default=25)
    parser.add_argument('--samples_2', default=10)
    parser.add_argument('--dim_1', default=128)
    parser.add_argument('--dim_2', default=128)
    parser.add_argument('--batch_size', default=512)
    parser.add_argument('--sigmoid', action="store_true")
    parser.add_argument('--identity_dim', default=0)
    
    parser.add_argument('--validate_iter', default=5000)
    parser.add_argument('--validate_batch_size', default=256)
    parser.add_argument('--gpu', default=1)
    parser.add_argument('--print_every', default=5)
    
    args = parser.parse_args()
    
    all_models = [
        'graphsage_mean',
        # 'gcn',
        # 'graphsage_seq',
        # 'graphsage_maxpool',
        # 'graphsage_meanpool',
    ]
    assert args.model in all_models, 'Error: model name unrecognized.'
    
    return args

if __name__ == "__main__":
    
    set_seeds(456)
    args = parse_args()
    
    # --
    # IO
    
    cache_path = '%s-%d-%d-iterator-cache.h5' % (args.train_prefix, args.batch_size, args.max_degree)
    if not os.path.exists(cache_path):
        data_loader = NodeDataLoader(
            data_path=args.train_prefix,
            batch_size=args.batch_size,
            max_degree=args.max_degree,
        )
        
        data_loader.save(cache_path)
    else:
        data_loader = NodeDataLoader(cache_path=cache_path)
    
    train_adj_ = Variable(torch.LongTensor(data_loader.train_adj.astype(int))).cuda()
    val_adj_ = Variable(torch.LongTensor(data_loader.val_adj.astype(int))).cuda()
    features_ = Variable(torch.FloatTensor(data_loader.features)).cuda()
    
    # --
    # Define model
    
    model = SupervisedGraphsage(**{
        "input_dim"     : data_loader.features.shape[1],
        "num_classes"   : data_loader.num_classes,
        "layer_infos" : {
            1 : {"sample_fn" : uniform_neighbor_sampler, "n_samples" : args.samples_1, "output_dim" : args.dim_1},
            2 : {"sample_fn" : uniform_neighbor_sampler, "n_samples" : args.samples_2, "output_dim" : args.dim_2},
        },
        "learning_rate" : args.learning_rate,
        "weight_decay"  : args.weight_decay,
    }).cuda()
    
    print(model)
    
    # --
    # Train
    
    set_seeds(891)
    
    total_steps = 0
    for epoch in range(args.epochs):
        # Train
        for iter_, train_batch in enumerate(data_loader.iterate(mode='train', shuffle=True)):
            
            ids = Variable(torch.LongTensor(train_batch['batch'])).cuda()
            labels = Variable(torch.FloatTensor(train_batch['labels'])).cuda()
            preds, loss = model.train_step(ids, features_, train_adj_, labels)
        
            if not total_steps % args.print_every:
                train_f1 = calc_f1(to_numpy(labels), to_numpy(preds))
                print({
                    "epoch" : epoch,
                    "iter" : iter_,
                    "train_f1" : train_f1,
                    # "val_f1" : val_f1,
                })
            
            total_steps += 1
        
        # Evaluate
        preds, labels = [], []
        for eval_batch in data_loader.iterate(mode='val', shuffle=False):
            ids = Variable(torch.LongTensor(eval_batch['batch'])).cuda()
            preds.append(to_numpy(model(ids, features_, val_adj_)))
            labels.append(eval_batch['labels'])
        
        print(calc_f1(np.vstack(labels), np.vstack(preds)))



