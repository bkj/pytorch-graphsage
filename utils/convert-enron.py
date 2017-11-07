#!/usr/bin/env python

"""
    convert-enron.py
"""

import numpy as np
import pandas as pd
import networkx as nx
from scipy.io import loadmat
from convert import make_adjacency, save_problem


Z = loadmat('./data/enron/enron_adj.mat')
adj, labs = Z['A'], Z['label'].squeeze()

inds = np.argsort(labs)
adj = adj[inds][:,inds]
labs = labs[inds]
labs -= 1
labs = 1 - labs

# >>
from scipy.linalg import eig
from sklearn.svm import LinearSVC
from rsub import *
from matplotlib import pyplot as plt

U, s, Vh = svd(adj)
U = U[:,:16]
s = s[:16]

e = s * U

# e /= np.sqrt((e ** 2).sum(axis=1, keepdims=True))

res = []
for _ in tqdm(range(10000)):
    pos_train = np.random.choice(np.where(labs == 1)[0], 10, replace=False)
    neg_train = np.random.choice(np.where(labs == 0)[0], 20, replace=False)
    train_sel = np.concatenate([pos_train, neg_train])
    
    train_sel = np.array(pd.Series(np.arange(labs.shape[0])).isin(train_sel))
    
    e_train, e_val = e[train_sel], e[~train_sel]
    y_train, y_val = labs[train_sel], labs[~train_sel]
    
    svc = LinearSVC().fit(e_train, y_train)
    
    res.append(y_val[svc.decision_function(e_val).argsort()[::-1]])

res = np.vstack(res)

_ = plt.plot(res.mean(axis=0))
show_plot()

# <<

# --

G = nx.from_edgelist(np.vstack(np.where(adj)).T)

adj = make_adjacency(G, 100, sel=None) # Adds dummy node

targets = labs.reshape(-1, 1)

folds = ['val'] * targets.shape[0]

pos_train = np.random.choice(np.where(targets == 1)[0], 10, replace=False)
neg_train = np.random.choice(np.where(targets == 0)[0], 20, replace=False)
train_sel = np.concatenate([pos_train, neg_train])

for i in train_sel:
    folds[i] = 'train'

folds = np.array(folds)

aug_targets = np.vstack([targets, np.zeros((targets.shape[1],), dtype='int64')])
aug_folds   = np.hstack([folds, ['dummy']])

save_problem({
    "task"      : 'classification',
    "n_classes" : 2,
    
    "adj"       : adj,
    "train_adj" : adj,
    
    "feats"     : None,
    "targets"   : aug_targets,
    "folds"     : aug_folds,
}, './data/enron/problem.h5')
