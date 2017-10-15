#!/usr/bin/env python

"""
    minibatch.py
"""

from __future__ import division
from __future__ import print_function

import numpy as np

class NodeMinibatchIterator(object):
    def __init__(self, G, id2idx, label_map, num_classes, batch_size=100, max_degree=25, **kwargs):
        
        self.G            = G
        self.id2idx       = id2idx
        self.batch_size   = batch_size
        self.max_degree   = max_degree
        self.label_map    = label_map
        self.num_classes  = num_classes
        
        self.train_adj, self.degrees = self.construct_adj(train=True)
        self.val_adj, _ = self.construct_adj(train=False)
        
        val_nodes   = [n for n in self.G.nodes() if self.G.node[n]['val']]
        test_nodes  = [n for n in self.G.nodes() if self.G.node[n]['test']]
        train_nodes = set(G.nodes()).difference(set(val_nodes + test_nodes))
        train_nodes = [n for n in train_nodes if self.degrees[id2idx[n]] > 0]
        
        self.nodes = {
            "train" : np.array(train_nodes),
            "val" : np.array(val_nodes),
            "test" : np.array(test_nodes),
        }
    
    def construct_adj(self, train):
        adj = np.zeros((len(self.id2idx), self.max_degree)) - 1
        degrees = np.zeros(len(self.id2idx))
        
        for nodeid in self.G.nodes():
            if train:
                if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                    continue
                
                # ?? What is `train_removed`?
                neighbors = np.array([self.id2idx[neighbor] for neighbor in self.G.neighbors(nodeid)
                    if (not self.G[nodeid][neighbor]['train_removed'])])
                
                degrees[self.id2idx[nodeid]] = len(neighbors)
            else:
                neighbors = np.array([self.id2idx[neighbor] for neighbor in self.G.neighbors(nodeid)])
            
            if len(neighbors) == 0:
                # If no neighbors, skip
                continue
            elif len(neighbors) > self.max_degree:
                # If degree is too large, downsample
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            elif len(neighbors) < self.max_degree:
                # If degree is too small, sample w/ replacement
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            
            adj[self.id2idx[nodeid], :] = neighbors
        
        return adj, degrees
    
    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            # Sparse categorical
            return np.array(label)
        else:
            # One hot
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
            return label_vec
    
    def batch_feed_dict(self, batch_nodes, val=False):
        return {
            'batch_size' : len(batch_nodes),
            'batch' : [self.id2idx[n] for n in batch_nodes],
            'labels' : np.vstack([self._make_label_vec(node) for node in batch_nodes])
        }
        
    def get_eval_batch(self, size=None, mode='val'):
        nodes = self.nodes[mode]
        nodes = nodes if size is None else np.random.choice(nodes, size, replace=True)
        return self.batch_feed_dict(nodes)
    
    def iterate(self, mode, shuffle=False):
        nodes = self.nodes[mode]
        
        if shuffle:
            idx = np.random.permutation(len(nodes)).astype(int)
        else:
            idx = np.arange(len(nodes)).astype(int)
        
        for idx_chunk in np.array_split(idx, idx.shape[0] // self.batch_size + 1):
            yield self.batch_feed_dict(nodes[idx_chunk])
