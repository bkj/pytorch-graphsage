#!/usr/bin/env python

"""
    data_loader.py
"""

from __future__ import division
from __future__ import print_function

import os
import sys
import h5py
import cPickle
import numpy as np
import ujson as json
from networkx.readwrite import json_graph

class NodeDataLoader(object):
    def __init__(self, data_path=None, cache_path=None, batch_size=100, max_degree=25):
        if data_path:
            print("Building NodeDataLoader from data", file=sys.stderr)
            self.__init(data_path, batch_size, max_degree)
        elif cache_path:
            print("Loading NodeDataLoader from cache", file=sys.stderr)
            self.__load(cache_path)
        else:
            raise Exception('NodeDataLoader: either `data_path` or `cache_path` must != None')
    
    def __init(self, data_path, batch_size, max_degree):
        G, features, id2idx, class_map = load_data(data_path)
        
        # Munge features
        if features is not None:
            features = np.vstack([features, np.zeros((features.shape[1],))])
        
        # Determine number of classes
        if isinstance(list(class_map.values())[0], list):
            self.num_classes = len(list(class_map.values())[0])
        else:
            self.num_classes = len(set(class_map.values()))
        
        # Construct adjacency matrices
        self.train_adj, self.degrees = self.construct_adj(G, id2idx, max_degree, train=True)
        self.val_adj, _ = self.construct_adj(G, id2idx, max_degree, train=False)
        
        # Construct node lists
        val_nodes   = [n for n in G.nodes() if G.node[n]['val']]
        test_nodes  = [n for n in G.nodes() if G.node[n]['test']]
        train_nodes = set(G.nodes()).difference(set(val_nodes + test_nodes))
        train_nodes = [n for n in train_nodes if self.degrees[id2idx[n]] > 0]
        
        self.nodes = {
            "train" : np.array(train_nodes),
            "val"   : np.array(val_nodes),
            "test"  : np.array(test_nodes),
        }
        
        self.batch_size  = batch_size
        self.features = features
        self.id2idx = id2idx
        self.class_map = class_map
    
    def __load(self, cache_path):
        f = h5py.File(cache_path)
        
        self.id2idx       = cPickle.loads(f['id2idx'].value)
        self.batch_size   = f['batch_size'].value
        self.class_map    = cPickle.loads(f['class_map'].value)
        self.num_classes  = f['num_classes'].value
        
        self.features = f['features'].value
        
        self.train_adj = f['train_adj'].value
        self.degrees   = f['degrees'].value
        self.val_adj   = f['val_adj'].value
        
        self.nodes = {
            "train" : cPickle.loads(f['nodes/train'].value),
            "val"   : cPickle.loads(f['nodes/val'].value),
            "test"  : cPickle.loads(f['nodes/test'].value),
        }
        
        f.close()
    
    def save(self, outpath):
        f = h5py.File(outpath)
        
        f['batch_size']  = self.batch_size
        f['num_classes'] = self.num_classes        
        f['id2idx']      = cPickle.dumps(self.id2idx)
        f['class_map']   = cPickle.dumps(self.class_map)
        f['features']    = self.features
        
        f['train_adj']   = self.train_adj
        f['degrees']     = self.degrees
        f['val_adj']     = self.val_adj
        
        f['nodes/train'] = cPickle.dumps(self.nodes['train'])
        f['nodes/val']   = cPickle.dumps(self.nodes['val'])
        f['nodes/test']  = cPickle.dumps(self.nodes['test'])
        
        f.close()
    
    def construct_adj(self, G, id2idx, max_degree, train=True):
        adj = np.zeros((len(id2idx) + 1, max_degree)) + len(id2idx)
        degrees = np.zeros(len(id2idx))
        
        for node_id in G.nodes():
            if train:
                if G.node[node_id]['test'] or G.node[node_id]['val']:
                    continue
                
                neighbors = np.array([id2idx[neighbor] for neighbor in G.neighbors(node_id)
                    if (not G[node_id][neighbor]['train_removed'])])
                
                degrees[id2idx[node_id]] = len(neighbors)
            else:
                neighbors = np.array([id2idx[neighbor] for neighbor in G.neighbors(node_id)])
            
            if len(neighbors) == 0:
                # If no neighbors, skip
                continue
            elif len(neighbors) > max_degree:
                # If degree is too large, downsample
                neighbors = np.random.choice(neighbors, max_degree, replace=False)
            elif len(neighbors) < max_degree:
                # If degree is too small, sample w/ replacement
                neighbors = np.random.choice(neighbors, max_degree, replace=True)
            
            adj[id2idx[node_id], :] = neighbors
        
        return adj, degrees
    
    def _make_label_vec(self, node):
        label = self.class_map[node]
        if isinstance(label, list):
            # Sparse categorical
            return np.array(label)
        else:
            # One hot
            label_vec = np.zeros((self.num_classes))
            class_ind = self.class_map[node]
            label_vec[class_ind] = 1
            return label_vec
    
    def iterate(self, mode, shuffle=False):
        nodes = self.nodes[mode]
        
        if shuffle:
            idx = np.random.permutation(len(nodes)).astype(int)
        else:
            idx = np.arange(len(nodes)).astype(int)
        
        for idx_chunk in np.array_split(idx, idx.shape[0] // self.batch_size + 1):
            batch_nodes = nodes[idx_chunk]
            yield {
                'batch_size' : len(batch_nodes),
                'batch' : [self.id2idx[n] for n in batch_nodes],
                'labels' : np.vstack([self._make_label_vec(node) for node in batch_nodes])
            }


def load_data(data_path, normalize=True):
    # --
    # Load
    
    G = json_graph.node_link_graph(json.load(open(os.path.join(data_path, "G.json"))))
    class_map = json.load(open(os.path.join(data_path, "class_map.json")))
    id_map = json.load(open(os.path.join(data_path, "id_map.json")))
    
    feats = None
    if os.path.exists(os.path.join(data_path, "feats.npy")):
        feats = np.load(os.path.join(data_path, "feats.npy"))
    
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
    
    
    id_map    = {conversion(k):int(v) for k,v in id_map.items()}
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
