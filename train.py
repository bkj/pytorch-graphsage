#!/usr/bin/env python

"""
    train.py
"""

from __future__ import division
from __future__ import print_function

import sys
import argparse
import numpy as np

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from models import GSSupervised
from problem import NodeProblem
from helpers import set_seeds, to_numpy
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


def evaluate(model, problem, mode='val'):
    assert mode in ['test', 'val']
    preds, acts = [], []
    for (ids, targets, _) in problem.iterate(mode=mode, shuffle=False):
        preds.append(to_numpy(model(ids, problem.feats, problem.adj, train=True)))
        acts.append(to_numpy(targets))
    
    return problem.metric_fn(np.vstack(acts), np.vstack(preds))

# --
# Args

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--problem-path', type=str, required=True)
    parser.add_argument('--no-cuda', action="store_true")
    
    # Optimization params
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr-init', type=float, default=0.01)
    parser.add_argument('--lr-schedule', type=str, default='constant')
    parser.add_argument('--weight-decay', type=float, default=0.0)
    
    # Architecture params
    parser.add_argument('--aggregator-class', type=str, default='mean')
    parser.add_argument('--prep-class', type=str, default='identity')
    parser.add_argument('--n-train-samples', type=str, default='25,10')
    parser.add_argument('--n-val-samples', type=str, default='25,10')
    parser.add_argument('--output-dims', type=str, default='128,128')
    
    # Logging
    parser.add_argument('--log-interval', default=10, type=int)
    parser.add_argument('--seed', default=123, type=int)
    parser.add_argument('--show-test', action="store_true")
    
    # --
    # Validate args
    
    args = parser.parse_args()
    args.cuda = not args.no_cuda
    assert args.prep_class in prep_lookup.keys(), 'parse_args: prep_class not in %s' % str(prep_lookup.keys())
    assert args.aggregator_class in aggregator_lookup.keys(), 'parse_args: aggregator_class not in %s' % str(aggregator_lookup.keys())
    assert args.batch_size > 1, 'parse_args: batch_size must be > 1'
    return args

if __name__ == "__main__":
    args = parse_args()
    set_seeds(args.seed)
    
    # --
    # Load problem
    
    problem = NodeProblem(problem_path=args.problem_path, cuda=args.cuda)
    
    # --
    # Define model
    
    n_train_samples = map(int, args.n_train_samples.split(','))
    n_val_samples = map(int, args.n_val_samples.split(','))
    output_dims = map(int, args.output_dims.split(','))
    model = GSSupervised(**{
        "prep_class" : prep_lookup[args.prep_class],
        "aggregator_class" : aggregator_lookup[args.aggregator_class],
        
        "input_dim" : problem.feats_dim,
        "n_nodes"   : problem.n_nodes,
        "n_classes" : problem.n_classes,
        "layer_specs" : [
            {
                "sample_fn" : uniform_neighbor_sampler,
                "n_train_samples" : n_train_samples[0],
                "n_val_samples" : n_val_samples[0],
                "output_dim" : output_dims[0],
                "activation" : F.relu,
            },
            {
                "sample_fn" : uniform_neighbor_sampler,
                "n_train_samples" : n_train_samples[1],
                "n_val_samples" : n_val_samples[1],
                "output_dim" : output_dims[1],
                "activation" : lambda x: x,
            },
        ],
        
        "lr_init" : args.lr_init,
        "lr_schedule" : args.lr_schedule,
        "weight_decay" : args.weight_decay,
    })
    
    if args.cuda:
        model = model.cuda()
    
    print(model, file=sys.stderr)
    
    # --
    # Train
    
    set_seeds(args.seed ** 2)
    
    val_f1 = None
    for epoch in range(args.epochs):
        
        # Train
        _ = model.train()
        for ids, targets, epoch_progress in problem.iterate(mode='train', shuffle=True, batch_size=args.batch_size):
            model.set_progress((epoch + epoch_progress) / args.epochs)
            preds = model.train_step(
                ids=ids, 
                feats=problem.feats,
                adj=problem.train_adj,
                targets=targets,
                loss_fn=problem.loss_fn,
            )
            print({
                "epoch_progress" : epoch_progress,
                "train_metric" : problem.metric_fn(to_numpy(targets), to_numpy(preds))
            })
            sys.stdout.flush()
        
        # Evaluate
        _ = model.eval()
        print('-- eval --', file=sys.stderr)
        print({
            "epoch" : epoch,
            "train_metric" : problem.metric_fn(to_numpy(targets), to_numpy(preds)),
            "val_metric" : evaluate(model, problem, mode='val'),
        })
        print('----------', file=sys.stderr)
        
    if args.show_test:
        print({"test_f1" : evaluate(model, problem, mode='test')})

