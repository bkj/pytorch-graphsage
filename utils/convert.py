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
import pandas as pd
import ujson as json
from tqdm import tqdm
import networkx as nx
from scipy.sparse import csr_matrix
from networkx.readwrite import json_graph
from sklearn.preprocessing import StandardScaler

assert int(nx.__version__.split('.')[0]) < 2, "networkx major version > 1"

# --
# Helpers

def parse_fold(x):
    if x['test']:
        return 'test'
    elif x['val']:
        return 'val'
    else:
        return 'train'

def validate_problem(problem):
    assert problem['adj'] is not None, "problem['adj'] is None"
    assert problem['train_adj'] is not None, "problem['train_adj'] is None"
    assert problem['targets'] is not None, "problem['targets'] is None"
    assert problem['folds'] is not None, "problem['folds'] is None"
    
    if problem['feats'] is not None:
        assert problem['feats'].shape[0] == problem['targets'].shape[0], "problem['feats'].shape[0] != (problem['targets'].shape[0]"
        assert problem['feats'].shape[0] == problem['folds'].shape[0], "problem['feats'].shape[0] != (problem['folds'].shape[0]"
        
        if 'sparse' in problem and not problem['sparse']:
            assert problem['adj'].shape[0] == problem['feats'].shape[0], "problem['adj'].shape[0] != problem['feats'].shape[0]"
    
    assert problem['adj'].shape[0] == problem['train_adj'].shape[0], "problem['adj'].shape[0] != problem['train_adj'].shape[0]"
    assert len(problem['targets'].shape) == 2, "len(problem['targets'].shape) != 2"
    return True

def save_problem(problem, outpath):
    assert validate_problem(problem)
    assert not os.path.exists(outpath), 'save_problem: %s already exists' % outpath
    
    f = h5py.File(outpath)
    for k,v in problem.items():
        if v is not None:
            f[k] = v
    
    f.close()


def make_adjacency(G, max_degree, sel=None):
    
    all_nodes = np.array(G.nodes())
    
    # Initialize w/ links to a dummy node
    n_nodes = len(all_nodes)
    adj = (np.zeros((n_nodes + 1, max_degree)) + n_nodes).astype(int)
    
    if sel is not None:
        # only look at nodes in training set
        all_nodes = all_nodes[sel]
    
    for node in tqdm(all_nodes):
        neibs = np.array(list(G.neighbors(node)))
        
        if sel is not None:
            neibs = neibs[sel[neibs]]
        
        if len(neibs) > 0:
            if len(neibs) > max_degree:
                neibs = np.random.choice(neibs, max_degree, replace=False)
            elif len(neibs) < max_degree:
                extra = np.random.choice(neibs, max_degree - neibs.shape[0], replace=True)
                neibs = np.concatenate([neibs, extra])
            
            adj[node, :] = neibs
    
    return adj

def make_sparse_adjacency(G, sel=None):
    
    all_nodes = np.array(G.nodes())
    
    r, c, v = [], [], []
    
    if sel is not None:
        all_nodes = all_nodes[sel]
    
    for node in tqdm(all_nodes):
        neibs = np.array(list(G.neighbors(node)))
        
        if sel is not None:
            neibs = neibs[sel[neibs]]
        
        if len(neibs) > 0:
            r.append(node + np.zeros(len(neibs)).astype(int) + 1) # Off-by-one
            c.append(np.arange(len(neibs)))
            v.append(neibs + 1) # Off-by-one
    
    return csr_matrix((
        np.hstack(v),
        (
            np.hstack(r),
            np.hstack(c),
        )
    ))

def spadj2edgelist(spadj):
    spadj_v = spadj.data
    spadj_r, spadj_c = spadj.nonzero()
    return np.vstack([spadj_v, spadj_r, spadj_c])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', type=str, default='./data/reddit/')
    parser.add_argument('--outpath', type=str)
    parser.add_argument('--max-degree', type=int, default=128)
    parser.add_argument('--task', type=str, default='classification')
    
    args = parser.parse_args()
    assert args.task in ['classification', 'multilabel_classification'], 'unknown args.task'
    if not args.outpath:
        args.outpath = os.path.join(args.inpath, 'problem.h5')
    
    return args


# if __name__ == "__main__":
args = parse_args()

if os.path.exists(args.outpath):
    print('backing up old problem.h5', file=sys.stderr)
    _ = shutil.move(args.outpath, args.outpath + '.bak')

print('loading <- %s' % args.inpath, file=sys.stderr)
id2target = json.load(open(os.path.join(args.inpath, 'class_map.json')))
id2idx    = json.load(open(os.path.join(args.inpath, 'id_map.json')))
feats     = np.load(os.path.join(args.inpath, 'feats.npy'))
G         = json_graph.node_link_graph(json.load(open(os.path.join(args.inpath, 'G.json'))))

walks = None
if os.path.exists(os.path.join(args.inpath, 'walks.txt')):
    walks = pd.read_csv(os.path.join(args.inpath, 'walks.txt'), header=None, sep='\t')
    walks = np.array(walks)

print('reordering')
feats   = np.vstack([feats[id2idx[str(id)]] for id in G.nodes()])
targets = np.vstack([id2target[str(id)] for id in G.nodes()])
folds   = np.array([parse_fold(G.node[id]) for id in G.nodes()])

old2new = dict(zip(G.nodes(), range(len(G.nodes()))))
if walks is not None:
    walks = np.vstack([(
        old2new.get(w[0], -1),
        old2new.get(w[1], -1),
    ) for w in walks])

walks = walks[~(walks == -1).any(axis=1)]

G = nx.convert_node_labels_to_integers(G)

print('normalizing feats', file=sys.stderr)
scaler = StandardScaler().fit(feats[folds == 'train'])
feats = scaler.transform(feats)

print('n_classes', file=sys.stderr)
if args.task == 'classification':
    n_classes = len(np.unique(targets))
elif args.task == 'multilabel_classification':
    n_classes = targets.shape[1]
elif 'regression' in args.task:
    n_classes = None

print('making adjacency lists', file=sys.stderr)
adj = make_adjacency(G, args.max_degree, sel=None) # Adds dummy node
train_adj = make_adjacency(G, args.max_degree, sel=(folds == 'train')) # Adds dummy node

aug_feats   = np.vstack([feats, np.zeros((feats.shape[1],))]) # Add feat for dummy node
aug_targets = np.vstack([targets, np.zeros((targets.shape[1],), dtype='int64')])
aug_folds   = np.hstack([folds, ['dummy']])

# >>

# !! What about edges w/ one node in?
# train_ids = set(np.where(folds == 'train')[0])
# tmp = [{
#     "e0" : e[0] in train_ids,
#     "e1" : e[1] in train_ids,
#     "train_removed" : G[e[0]][e[1]].get('train_removed'),
# } for e in G.edges()]

# tmp = pd.DataFrame(tmp)

# tmp.drop_duplicates()
# # e0     e1  train_removed
# # 0        True   True          False
# # 153024  False  False           True

non_train_nodes = set(np.where(folds != 'train')[0])
val_edges = np.vstack([e for e in G.edges() if e[0] in non_train_nodes or e[1] in non_train_nodes]) # !! Needs to be generalize.  What are we doing exactly?
context_pairs = np.vstack([walks, val_edges])
context_pairs_folds = np.concatenate([
    np.repeat('train', walks.shape[0]),
    np.repeat('val', val_edges.shape[0]),
])

perm = np.random.permutation(context_pairs.shape[0])
context_pairs = context_pairs[perm]
context_pairs_folds = context_pairs_folds[perm]

# <<

print('saving -> %s' % args.outpath, file=sys.stderr)
save_problem({
    "task"      : args.task,
    "n_classes" : n_classes,
    
    "adj"       : adj,
    "train_adj" : train_adj,
    
    "feats"     : aug_feats,
    "targets"   : aug_targets,
    "folds"     : aug_folds,
    
    "context_pairs" : context_pairs,
    "context_pairs_folds" : context_pairs_folds
}, args.outpath)

# # >>
# print('making sparse adjacency lists', file=sys.stderr)
# adj = make_sparse_adjacency(G, sel=None) # Adds dummy node
# train_adj = make_sparse_adjacency(G, sel=(folds == 'train')) # Adds dummy node

# aug_feats   = np.vstack([np.zeros((feats.shape[1],)), feats]) # Add feat for dummy node
# aug_targets = np.vstack([np.zeros((targets.shape[1],), dtype='int64'), targets])
# aug_folds   = np.hstack([['dummy'], folds])

# print('saving -> %s' % args.outpath, file=sys.stderr)
# save_problem({
#     "task"      : args.task,
#     "n_classes" : n_classes,

#     "sparse"    : True,
#     "adj"       : spadj2edgelist(adj),
#     "train_adj" : spadj2edgelist(train_adj),

#     "feats"     : aug_feats,
#     "targets"   : aug_targets,
#     "folds"     : aug_folds,
# }, './data/reddit/sparse-problem.h5')
# # <<