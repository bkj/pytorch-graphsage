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

def parse_node(x):
    if x['test']:
        return 'test'
    elif x['val']:
        return 'val'
    else:
        return 'train'


def normalize(feats, idx2fold):
    train_ids = np.array([k for k,v in idx2fold.items() if v == 'train'])
    train_feats = feats[train_ids]
    scaler = StandardScaler()
    scaler.fit(train_feats)
    return scaler.transform(feats)


def make_adjacency(G, idx2fold, max_degree, train=True):
    
    n_nodes = max(idx2fold.keys()) + 1
    adj = np.zeros((n_nodes, max_degree))
    adj += n_nodes # Initialize w/ OOB entries
    adj = adj.astype(int)
    
    all_nodes = G.nodes()
    if train:
        all_nodes = filter(lambda x: idx2fold[x] != 'train', all_nodes)
    
    for node in all_nodes:
        neighbors = G.neighbors(node)
        
        if train:
            neighbors = filter(lambda x: idx2fold[x] != 'train', neighbors)
        
        if len(neighbors) > 0:
            if len(neighbors) > max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=False)
            elif len(neighbors) < max_degree:
                neighbors = np.random.choice(neighbors, max_degree, replace=True)
            
            adj[node, :] = np.array(neighbors)
    
    return adj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, required=True)
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
    class_map = json.load(open(os.path.join(args.inpath, 'class_map.json')))
    id2idx = json.load(open(os.path.join(args.inpath, 'id_map.json')))
    
    print('making graph', file=sys.stderr)
    G = json_graph.node_link_graph(json.load(open(os.path.join(args.inpath, 'G.json'))))
    G = nx.relabel_nodes(G, id2idx)
    
    print('making lists', file=sys.stderr)
    idx2id = dict([(v, k) for k,v in id2idx.items()])
    idx2class = dict([(id2idx[k], v) for k,v in class_map.items()])
    idx2fold = dict(map(lambda x: (x, parse_node(G.node[x])), G.nodes()))
    
    print('normalizing feats', file=sys.stderr)
    feats = normalize(np.load(os.path.join(args.inpath, 'feats.npy')), idx2fold)
    
    print('making adjacency lists', file=sys.stderr)
    adj = make_adjacency(G, idx2fold, args.max_degree, train=False)
    train_adj = make_adjacency(G, idx2fold, args.max_degree, train=True)
    
    print('finishing up', file=sys.stderr)
    if args.task == 'classification':
        n_classes = len(set(idx2class.values()))
    elif args.task == 'multilabel_classification':
        n_classes = len(idx2class[0])
    
    print('saving', file=sys.stderr)
    problem = {
        "idx2id"    : cPickle.dumps(idx2id),
        "idx2class" : cPickle.dumps(idx2class),
        "idx2fold"  : cPickle.dumps(idx2fold),
        "task"      : args.task,
        "n_classes" : n_classes,
        "feats"     : feats,
        "train_adj" : adj,
        "adj"       : train_adj,
    }
    
    f = h5py.File(outpath)
    for k,v in problem.items():
        f[k] = v

    f.close()