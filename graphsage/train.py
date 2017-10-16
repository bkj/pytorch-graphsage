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

class UniformNeighborSampler(object):
    def __init__(self, adj, **kwargs):
        self.adj = adj
        
    def __call__(self, ids, num_samples=-1):
        # Select rows by nodes
        tmp = self.adj[ids]
        # Select random columns, up to num_samples
        tmp = tmp[:,torch.randperm(tmp.size(1))]
        return tmp[:,:num_samples]

def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = y_pred.round().astype(int)
    
    return {
        "micro" : metrics.f1_score(y_true, y_pred, average="micro"),
        "macro" : metrics.f1_score(y_true, y_pred, average="macro"),
    }


def evaluate(model, data_loader, mode):
    preds, labels = [], []
    for eval_batch in data_loader.iterate(mode=mode, shuffle=False):
        X = Variable(torch.FloatTensor(eval_batch['batch']))
        preds = model(X)
        preds.append(to_numpy(preds))
        preds.append(eval_batch['labels'])
    
    return calc_f1(np.vstack(labels), np.vstack(preds))

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
    parser.add_argument('--samples_3', default=0)
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

set_seeds(123)

def select_model(args, sampler):
    return {
        "layer_infos" : filter(None, [
            {"layer_name" : "node", "sampler" : sampler, "n_samples" : args.samples_1, "output_dim" : args.dim_1},
            {"layer_name" : "node", "sampler" : sampler, "n_samples" : args.samples_2, "output_dim" : args.dim_2} if args.samples_2 != 0 else None,
            {"layer_name" : "node", "sampler" : sampler, "n_samples" : args.samples_3, "output_dim" : args.dim_2} if args.samples_3 != 0 else None,
        ]),
    }

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
    
    adj_ = Variable(torch.LongTensor(data_loader.train_adj.astype(int)))
    
    params = {
        "num_classes"   : data_loader.num_classes,
        "features"      : data_loader.features,
        "adj"           : adj_,
        "degrees"       : data_loader.degrees,
        "model_size"    : args.model_size,
        "sigmoid"       : args.sigmoid,
        "identity_dim"  : args.identity_dim,
        "learning_rate" : args.learning_rate,
        "weight_decay"  : args.weight_decay,
    }
    sampler = UniformNeighborSampler(adj_)
    params.update(select_model(args, sampler))
    
    model = SupervisedGraphsage(**params)


