#!/usr/bin/env python

"""
    utils.py
"""

from __future__ import print_function

import os
import json
import numpy as np
from networkx.readwrite import json_graph


def load_data(prefix, normalize=True):
    # --
    # Load
    
    G = json_graph.node_link_graph(json.load(open(prefix + "-G.json")))
    class_map = json.load(open(prefix + "-class_map.json"))
    id_map = json.load(open(prefix + "-id_map.json"))
    
    feats = None
    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    
    # --
    # Format
    
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : str(n)
    
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)
    
    id_map = {conversion(k):int(v) for k,v in id_map.items()}
    class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}
    
    # Remove edges not in training dataset
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
    
    # Normalize features
    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
                
    return G, feats, id_map, class_map
