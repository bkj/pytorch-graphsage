import os
import numpy as np
import pandas as pd
import networkx as nx
from convert import make_adjacency, save_problem

def latlon2cartesian(latlon, d=1):
    return np.array([
        d * np.cos(latlon[:,0]) * np.cos(latlon[:,1]), # x
        d * np.cos(latlon[:,0]) * np.sin(latlon[:,1]), # y
        d * np.sin(latlon[:,1]),                     # z
    ])




max_degree = 24
inpath = '../data/hwai/'

locs = pd.read_csv(os.path.join(inpath, 'user_locations.tsv'), sep='\t')
locs = locs.sort_values('user_id')

edges = np.load(os.path.join(inpath, 'filtered_edges.npy'))
edges = pd.DataFrame(edges)

assert np.all(locs.user_id.isin(edges))
assert locs.user_id.unique().shape[0] == locs.shape[0]

# --
# Fix IDs

locs['uid'] = np.arange(locs.shape[0])

edges = pd.merge(edges, locs[['user_id', 'uid']], left_on=0, right_on='user_id')
edges = edges[['uid', 1]]
edges.columns = (0, 1)

edges = pd.merge(edges, locs[['user_id', 'uid']], left_on=1, right_on='user_id')
edges = edges[[0, 'uid']]
edges.columns = (0, 1)

assert np.unique(edges).shape[0] == np.array(edges).max() + 1


targets = np.array(locs[['latitude', 'longitude']])
targets2 = latlon2cartesian(targets)

folds = np.random.choice(['train', 'val'], targets.shape[0], p=[0.8, 0.2])

G = nx.from_edgelist(np.array(edges))
adj = make_adjacency(G, folds, max_degree, train=False)

outpath = '../data/hwai/problem.h5'

save_problem({
    "task"      : 'cosine_regression',
    "n_classes" : 2,
    "feats"     : None,
    "train_adj" : adj,
    "adj"       : adj,
    "targets"   : targets,
    "folds"     : folds,
}, outpath)