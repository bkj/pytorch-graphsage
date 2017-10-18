#!/usr/bin/env python

"""
    convert.py
    
    Convert problems from `graphsage` format to something that loads faster
"""

from __future__ import print_function

import os
import sys
import h5py
import shutil
import cPickle
import argparse
import numpy as np
import ujson as json
import networkx as nx
from networkx.readwrite import json_graph
from sklearn.preprocessing import StandardScaler

# --
# Helpers

def parse_fold(x):
    if x['test']:
        return 'test'
    elif x['val']:
        return 'val'
    else:
        return 'train'



def make_adjacency(G, folds, max_degree, train=True):
    
    all_nodes = np.array(G.nodes())
    
    # Initialize w/ links to a dummy node
    n_nodes = len(all_nodes)
    adj = (np.zeros((n_nodes + 1, max_degree)) + n_nodes).astype(int)
    
    if train:
        # only look at nodes in training set
        all_nodes = all_nodes[folds == 'train']
    
    for node in all_nodes:
        neibs = np.array(G.neighbors(node))
        
        if train:
            neibs = neibs[folds[neibs] == 'train']
        
        if len(neibs) > 0:
            if len(neibs) > max_degree:
                neibs = np.random.choice(neibs, max_degree, replace=False)
            elif len(neibs) < max_degree:
                neibs = np.random.choice(neibs, max_degree, replace=True)
            
            adj[node, :] = neibs
    
    return adj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='./data/reddit/')
    parser.add_argument('--max-degree', type=int, default=128)
    parser.add_argument('--task', type=str, default='classification')
    args = parser.parse_args()
    
    assert args.task in ['classification', 'multilabel_classification']
    
    return args



if __name__ == "__main__":
    args = parse_args()
    outpath = os.path.join(args.inpath, 'problem.h5')
    
    if os.path.exists(outpath):
        print('backing up old problem.h5', file=sys.stderr)
        _ = shutil.move(outpath, outpath + '.bak')
    
    
    print('loading', file=sys.stderr)
    id2target = json.load(open(os.path.join(args.inpath, 'class_map.json')))
    id2idx    = json.load(open(os.path.join(args.inpath, 'id_map.json')))
    feats     = np.load(os.path.join(args.inpath, 'feats.npy'))
    G         = json_graph.node_link_graph(json.load(open(os.path.join(args.inpath, 'G.json'))))
    
    
    print('reordering')
    feats   = np.vstack([feats[id2idx[id]] for id in G.nodes()])
    targets = np.vstack([id2target[id] for id in G.nodes()])
    folds   = np.array([parse_fold(G.node[id]) for id in G.nodes()])
    G       = nx.convert_node_labels_to_integers(G)
    
    
    print('normalizing feats', file=sys.stderr)
    scaler = StandardScaler().fit(feats[folds == 'train'])
    feats = scaler.transform(feats)
    
    
    print('making adjacency lists', file=sys.stderr)
    adj = make_adjacency(G, folds, args.max_degree, train=False) # Adds dummy node
    train_adj = make_adjacency(G, folds, args.max_degree, train=True) # Adds dummy node
    feats = np.vstack([feats, np.zeros((feats.shape[1],))]) # Add feat for dummy node
    
    
    print('finishing up', file=sys.stderr)
    if args.task == 'classification':
        n_classes = len(np.unique(targets))
    elif args.task == 'multilabel_classification':
        n_classes = targets.shape[1]
    
    
    print('saving', file=sys.stderr)
    problem = {
        "task"      : args.task,
        "n_classes" : n_classes,
        "feats"     : feats,
        "train_adj" : train_adj,
        "adj"       : adj,
        "targets"   : targets,
        "folds"     : folds,
    }

    f = h5py.File(outpath)
    for k,v in problem.items():
        f[k] = v

    f.close()