#!/usr/bin/env python

"""
    train.py
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

from lr import LRSchedule

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
    preds, acts = [], []
    for (ids, targets) in data_loader.iterate(mode=mode, shuffle=False):
        preds.append(to_numpy(model(ids, data_loader.features, data_loader.adj)))
        acts.append(to_numpy(targets))
    
    acts = np.vstack(acts)
    preds = np.vstack(preds)
    return calc_f1(acts, preds, multiclass=args.multiclass)


def calc_f1(y_true, y_pred, multiclass=True):
    if multiclass:
        y_pred = (y_pred > 0).astype(int)
    else:
        y_pred = np.argmax(y_pred, axis=1)
    
    return {
        "micro" : metrics.f1_score(y_true, y_pred, average="micro"),
        "macro" : metrics.f1_score(y_true, y_pred, average="macro"),
    }


# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data-path', type=str, required=True)
    parser.add_argument('--max-degree', type=int, default=128)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--no-cuda', action="store_true")
    
    # Optimization params
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr-init', type=float, default=0.01)
    parser.add_argument('--lr-schedule', type=str, default='constant')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    
    # Architecture params
    parser.add_argument('--aggregator-class', type=str, default='mean')
    parser.add_argument('--prep-class', type=str, default='identity')
    parser.add_argument('--multiclass', action='store_true')
    parser.add_argument('--n-samples', type=str, default='25,10')
    parser.add_argument('--output-dims', type=str, default='128,128')
    
    # Logging
    parser.add_argument('--log-interval', default=10, type=int)
    parser.add_argument('--seed', default=123, type=int)
    
    # --
    # Validate args
    
    args = parser.parse_args()
    
    args.cuda = not args.no_cuda
    assert args.prep_class in prep_lookup.keys(), 'Error: aggregator_class not recognized.'
    assert args.aggregator_class in aggregator_lookup.keys(), 'Error: prep_class not recognized.'
    
    return args

if __name__ == "__main__":
    args = parse_args()
    
    set_seeds(args.seed)
    
    # --
    # IO
    
    cache_path = os.path.join(args.data_path, 'iterator-cache-%d.h5' % args.max_degree)
    if not os.path.exists(cache_path):
        data_loader = NodeDataLoader(
            data_path=args.data_path,
            batch_size=args.batch_size,
            max_degree=args.max_degree,
            multiclass=args.multiclass,
            cuda=args.cuda,
        )
        
        data_loader.save(cache_path)
    else:
        data_loader = NodeDataLoader(cache_path=cache_path)
        data_loader.batch_size = args.batch_size
    
    # --
    # Define model
    
    n_samples = map(int, args.n_samples.split(','))
    output_dims = map(int, args.output_dims.split(','))
    model = GSSupervised(**{
        "prep_class"       : prep_lookup[args.prep_class],
        "aggregator_class" : aggregator_lookup[args.aggregator_class],
        
        "input_dim"        : data_loader.feature_dim,
        "num_classes"      : data_loader.num_classes,
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
        "lr_init" : args.lr_init,
        "lr_schedule" : args.lr_schedule,
        "weight_decay"  : args.weight_decay,
    })
    
    if args.cuda:
        model = model.cuda()
    
    print(model, file=sys.stderr)
    
    loss_fn = F.multilabel_soft_margin_loss if args.multiclass else F.cross_entropy
    
    # --
    # Train
    
    set_seeds(args.seed ** 2)
    
    val_f1 = None
    for epoch in range(args.epochs):
        # Train
        _ = model.train()
        for ids, targets, epoch_progress in data_loader.iterate(mode='train', shuffle=True):
            
            model.set_progress((epoch + epoch_progress) / args.epochs)
            preds = model.train_step(
                ids=ids, 
                features=data_loader.features,
                adj=data_loader.train_adj,
                targets=targets,
                loss_fn=loss_fn
            )
            
            if not iter_ % args.log_interval:
                train_f1 = calc_f1(to_numpy(targets), to_numpy(preds), multiclass=args.multiclass)
                print({
                    "epoch" : epoch,
                    "epoch_progress" : epoch_progress,
                    "train_f1" : train_f1,
                    "val_f1" : val_f1,
                })
        
        # Evaluate
        _ = model.eval()
        val_f1 = evaluate(model, data_loader, mode='val', multiclass=args.multiclass)
    
    print({"val_f1" : val_f1})
    print({"test_f1" : evaluate(model, data_loader, mode='test', multiclass=args.multiclass)})

