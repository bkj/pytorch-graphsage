#!/usr/bin/env python

"""
    supervised_train.py
"""

from __future__ import division
from __future__ import print_function

import os
import sys
import time
import sklearn
import numpy as np
import argparse
from sklearn import metrics

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from models import GSSupervised
from helpers import set_seeds, to_numpy
from data_loader import NodeDataLoader
from nn_modules import aggregator_lookup, prep_lookup

# --
# Helpers

def uniform_neighbor_sampler(ids, adj, n_samples=-1):
    tmp = adj[ids]
    perm = torch.randperm(tmp.size(1))
    if adj.is_cuda:
        perm = perm.cuda()
    
    tmp = tmp[:,perm]
    return tmp[:,:n_samples]


def evaluate(model, data_loader, mode='val', multiclass=True):
    preds, labels = [], []
    for eval_batch in data_loader.iterate(mode='val', shuffle=False):
        ids = Variable(torch.LongTensor(eval_batch['batch'])).cuda()
        preds.append(to_numpy(model(ids, features_, val_adj_)))
        labels.append(eval_batch['labels'])
    
    return calc_f1(np.vstack(labels), np.vstack(preds), multiclass=args.multiclass)


def calc_f1(y_true, y_pred, multiclass=True):
    if not multiclass:
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
    
    parser.add_argument('--data-path', required=True)
    parser.add_argument('--max-degree', default=128)
    parser.add_argument('--batch-size', default=512)
    parser.add_argument('--no-cuda', action="store_true")
    
    # Optimization params
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--learning-rate', default=0.01)
    parser.add_argument('--weight-decay', default=0.0)
    
    # Architecture params
    parser.add_argument('--aggregator-class', default='mean')
    parser.add_argument('--prep-class', default='identity')
    parser.add_argument('--multiclass', action='store_true')
    parser.add_argument('--n-samples', default='25,10')
    parser.add_argument('--output-dims', default='128,128')
    
    # Logging
    parser.add_argument('--log-interval', default=5)
    
    # --
    # Validate args
    
    args = parser.parse_args()
    
    args.cuda = not args.no_cuda
    assert args.prep_class in prep_lookup.keys(), 'Error: aggregator_class not recognized.'
    assert args.aggregator_class in aggregator_lookup.keys(), 'Error: prep_class not recognized.'
    
    return args

if __name__ == "__main__":
    
    set_seeds(456)
    args = parse_args()
    
    # --
    # IO
    
    cache_path = os.path.join(args.data_path, 'iterator-cache-%d-%d.h5' % (args.batch_size, args.max_degree))
    if not os.path.exists(cache_path):
        data_loader = NodeDataLoader(
            data_path=args.data_path,
            batch_size=args.batch_size,
            max_degree=args.max_degree,
        )
        
        data_loader.save(cache_path)
    else:
        data_loader = NodeDataLoader(cache_path=cache_path)
    
    # --
    # Define model
    
    n_samples = map(int, args.n_samples.split(','))
    output_dims = map(int, args.output_dims.split(','))
    
    model = GSSupervised(**{
        "input_dim"   : data_loader.features.shape[1],
        "num_classes" : data_loader.num_classes,
        "prep_class" : prep_lookup[args.prep_class],
        "aggregator_class" : aggregator_lookup[args.aggregator_class],
        "layer_specs" : [
            {
                "sample_fn" : uniform_neighbor_sampler,
                "n_samples" : n_samples[0], 
                "output_dim" : output_dims[0],
                "activation" : F.relu,
            },
            {
                "sample_fn" : uniform_neighbor_sampler,
                "n_samples" : n_samples[1], 
                "output_dim" : output_dims[1],
                "activation" : lambda x: x,
            },
        ],
        "learning_rate" : args.learning_rate,
        "weight_decay"  : args.weight_decay,
    })
    
    print(model, file=sys.stderr)
    
    features_  = Variable(torch.FloatTensor(data_loader.features))
    train_adj_ = Variable(torch.LongTensor(data_loader.train_adj.astype(int)))
    val_adj_   = Variable(torch.LongTensor(data_loader.val_adj.astype(int)))
    
    if args.cuda:
        model      = model.cuda()
        train_adj_ = train_adj_.cuda()
        val_adj_   = val_adj_.cuda()
        features_  = features_.cuda()
    
    # --
    # Train
    
    set_seeds(891)
    
    val_f1 = None
    for epoch in range(args.epochs):
        # Train
        _ = model.train()
        for iter_, train_batch in enumerate(data_loader.iterate(mode='train', shuffle=True)):
            
            ids = Variable(torch.LongTensor(train_batch['batch']))
            labels = Variable(torch.FloatTensor(train_batch['labels']))
            
            if args.cuda:
                ids, labels = ids.cuda(), labels.cuda()
            
            preds = model.train_step(ids, features_, train_adj_, labels)
            
            if not iter_ % args.log_interval:
                train_f1 = calc_f1(to_numpy(labels), to_numpy(preds), multiclass=args.multiclass)
                print({
                    "epoch"    : epoch,
                    "iter"     : iter_,
                    "train_f1" : train_f1,
                    "val_f1"   : val_f1,
                })
            
        
        # Evaluate
        _ = model.eval()
        val_f1 = evaluate(model, data_loader, mode='val', multiclass=args.multiclass)
    
    print({"val_f1" : val_f1})


