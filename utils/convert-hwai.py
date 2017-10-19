import os
import numpy as np
import pandas as pd
import networkx as nx
from convert import make_adjacency, save_problem

def latlon2cartesian(latlon, d=1):
    latlon = np.radians(latlon)
    return np.array([
        d * np.cos(latlon[:,0]) * np.cos(-latlon[:,1]), # x
        d * np.cos(latlon[:,0]) * np.sin(-latlon[:,1]), # y
        d * np.sin(latlon[:,0]),                      # z
    ]).T

def cartesian2latlon(xyz):
    d = (xyz ** 2).sum(axis=1)
    latlon = np.array([
        np.pi / 2 - np.arccos(xyz[:,2] / d), # lat
        - np.arctan2(xyz[:,1], xyz[:,0]),           # lon
    ]).T
    return np.degrees(latlon)



# --
# IO

max_degree = 24
inpath = '../data/hwai/'

locs = pd.read_csv(os.path.join(inpath, 'user_locations.tsv'), sep='\t')
edges = np.load(os.path.join(inpath, 'filtered_edges.npy'))

# --
# Filter locs to just US

locs = locs.sort_values('user_id')

bbox = 'westlimit=-128.4; southlimit=24.3; eastlimit=-65.4; northlimit=51.5'
bbox = map(lambda x: float(x.split('=')[1]), bbox.split(';'))
bbox = dict(zip(['west', 'south', 'east', 'north'], bbox))

sel = ((locs.latitude < bbox['north']) &
    (locs.latitude > bbox['south']) &
    (locs.longitude > bbox['west']) &
    (locs.longitude < bbox['east']))

locs = locs[sel]

edges = edges[edges[0].isin(locs.user_id) & edges[1].isin(locs.user_id)]
locs = locs[locs.user_id.isin(np.unique(edges))]
assert locs.shape[0] == np.unique(edges).shape[0]
assert locs.user_id.unique().shape[0] == locs.shape[0], 'non-unique user in locs'

# Fix ids

locs['uid'] = np.arange(locs.shape[0])

edges = pd.DataFrame(edges)
edges = pd.merge(edges, locs[['user_id', 'uid']], left_on=0, right_on='user_id')
edges = edges[['uid', 1]]
edges.columns = (0, 1)

edges = pd.merge(edges, locs[['user_id', 'uid']], left_on=1, right_on='user_id')
edges = edges[[0, 'uid']]
edges.columns = (0, 1)

assert (np.unique(edges).shape[0] == np.array(edges).max() + 1), 'new ids not sequential + unique'

# --
# Fix targets

targets = np.array(locs[['latitude', 'longitude']])
cart_targets = latlon2cartesian(targets)
assert np.allclose((cart_targets ** 2).sum(axis=1), 1), 'not unit norm'
assert np.allclose((cartesian2latlon(cart_targets) % 360 - targets % 360), 0), 'not reversible'

# --
# Folds

folds = np.random.choice(['train', 'val'], targets.shape[0], p=[0.8, 0.2])

G = nx.from_edgelist(np.array(edges))
adj = make_adjacency(G, folds, max_degree, train=False)

outpath = '../data/hwai/problem.h5'

save_problem({
    "task"      : 'geo_regression',
    "n_classes" : 3,
    "feats"     : None,
    "train_adj" : adj,
    "adj"       : adj,
    "targets"   : cart_targets,
    "folds"     : folds,
}, outpath)