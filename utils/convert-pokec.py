import os
import sys
import h5py
import numpy as np
import pandas as pd
import networkx as nx
from convert import make_adjacency, save_problem

np.random.seed(123)

def load_ages(path):
    ages = pd.read_csv(path, header=None, sep='\t')
    ages.columns = ('id', 'age')
    
    ages = ages[ages.age != 'null']
    ages.age = ages.age.astype(int)
    ages = ages[ages.age > 0]
    
    return ages


max_degree = 128
inpath = '../data/pokec/'

# --
# Load data

ages = load_ages(os.path.join(inpath, 'soc-pokec-ages.tsv'))
edges = pd.read_csv(os.path.join(inpath, 'soc-pokec-relationships.txt'), header=None, sep='\t')
edges.columns = ('src', 'trg')

edges = edges[edges.src.isin(ages.id)]
edges = edges[edges.trg.isin(ages.id)]
ages  = ages[ages.id.isin(edges.src) | ages.id.isin(edges.trg)]

ages['uid'] = np.arange(ages.shape[0])

edges = pd.merge(edges, ages, left_on='src', right_on='id')
edges = edges[['uid', 'trg']]
edges.columns = ('src', 'trg')
edges = pd.merge(edges, ages, left_on='trg', right_on='id')
edges = edges[['src', 'uid']]
edges.columns = ('src', 'trg')

ages = ages[['uid', 'age']]

targets = np.array(ages.age).astype(float).reshape(-1, 1)
folds = np.random.choice(['train', 'val'], targets.shape[0], p=[0.5, 0.5])

G = nx.from_edgelist(np.array(edges))
adj = make_adjacency(G, folds, max_degree, train=False) # Adds dummy node


# --
# Write

outpath = '../data/pokec/problem.h5'
problem = {
    "task"      : 'regression_mae',
    "n_classes" : None,
    "feats"     : None,
    "train_adj" : adj,
    "adj"       : adj,
    "targets"   : targets,
    "folds"     : folds,
}


save_problem(problem, outpath)
