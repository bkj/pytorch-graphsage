# !! Need to update this

import h5py
import networkx as nx
import numpy as np
from scipy import sparse as sp
import pandas as pd

from convert import make_adjacency

def encode_onehot(labels):
    ulabels = set(labels)
    ulabels_dict = {c: np.identity(len(ulabels))[i, :] for i, c in enumerate(ulabels)}
    return np.array(map(ulabels_dict.get, labels), dtype=np.int32)


def load_data(path="./data/cora/", dataset="cora"):
    idx_features_labels = np.loadtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-2], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    
    edges_unordered = np.loadtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(map(idx_map.get, edges_unordered.flatten()), dtype=np.int32).reshape(edges_unordered.shape)
    
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    
    return features.todense(), adj, labels


def make_mask(idx, n):
    mask = np.zeros(n)
    mask[idx] = 1
    return mask == 1


def get_splits(y):
    idx_train = range(140)
    idx_val   = range(200, 500)
    idx_test  = range(500, 1500)
    
    y_train = np.zeros(y.shape, dtype=np.int32)
    y_train[idx_train] = y[idx_train]
    
    y_val = np.zeros(y.shape, dtype=np.int32)
    y_val[idx_val] = y[idx_val]
    
    y_test = np.zeros(y.shape, dtype=np.int32)
    y_test[idx_test] = y[idx_test]
    
    return y_train, y_val, y_test, idx_train, idx_val, idx_test

# --
# IO

feats, sparse_adj, targets = load_data(dataset='cora')
feats /= feats.sum(axis=1)
targets = targets.argmax(axis=1)
folds = ['train' for _ in range(140)] + ['val' for _ in range(200, 500)] + ['test' for _ in range(500, len(feats))]
folds = np.array(folds)

feats = feats[:folds.shape[0]]
targets = targets[:folds.shape[0]].reshape(-1, 1)

dense_adj = sparse_adj.todense()
dense_adj = dense_adj[:folds.shape[0]][:,:folds.shape[0]]
dense_adj += np.identity(dense_adj.shape[0])
edges     = np.vstack(np.where(dense_adj)).T
G         = nx.from_edgelist(edges)

train_adj = make_adjacency(G, folds, 128, train=True)
adj = make_adjacency(G, folds, 128, train=False)

outpath = './data/cora/problem.h5'
problem = {
    "task"      : 'classification',
    "n_classes" : np.unique(targets).shape[0],
    "feats"     : feats,
    "train_adj" : train_adj,
    "adj"       : adj,
    "targets"   : targets,
    "folds"     : folds,
}

assert feats.shape[0] == targets.shape[0]
assert feats.shape[0] == folds.shape[0]
assert adj.shape[0] == train_adj.shape[0]
assert adj.shape[0] == (feats.shape[0] + 1)
assert len(targets.shape) == 2

f = h5py.File(outpath)
for k,v in problem.items():
    f[k] = v

f.close()