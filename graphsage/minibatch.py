#!/usr/bin/env python

"""
    minibatch.py
"""

from __future__ import division
from __future__ import print_function

import numpy as np

class NodeMinibatchIterator(object):
    
    """ 
    This minibatch iterator iterates over nodes for supervised learning.

    G -- networkx graph
    id2idx -- dict mapping node ids to integer values indexing feature tensor
    placeholders -- standard tensorflow placeholders object for feeding
    label_map -- map from node ids to class values (integer or list)
    num_classes -- number of output classes
    batch_size -- size of the minibatches
    max_degree -- maximum size of the downsampled adjacency lists
    """
    def __init__(self, G, id2idx, 
            placeholders, label_map, num_classes, 
            batch_size=100, max_degree=25,
            **kwargs):
        
        self.G = G
        self.nodes = G.nodes()
        self.id2idx = id2idx
        self.placeholders = placeholders
        self.batch_size = batch_size
        self.max_degree = max_degree
        self.batch_num = 0
        self.label_map = label_map
        self.num_classes = num_classes
        
        self.adj, self.deg = self.construct_adj(train=True)
        self.val_adj, _ = self.construct_adj(train=False)
        
        self.val_nodes  = [n for n in self.G.nodes() if self.G.node[n]['val']]
        self.test_nodes = [n for n in self.G.nodes() if self.G.node[n]['test']]
        self.no_train_nodes_set = set(self.val_nodes + self.test_nodes)
        
        self.train_nodes = set(G.nodes()).difference(self.no_train_nodes_set)
        # don't train on nodes that only have edges to test set
        self.train_nodes = [n for n in self.train_nodes if self.deg[id2idx[n]] > 0]
    
    def construct_adj(self, train):
        adj = len(self.id2idx) * np.ones((len(self.id2idx) + 1 , self.max_degree))
        deg = np.zeros((len(self.id2idx),))
        
        for nodeid in self.G.nodes():
            if train:
                if self.G.node[nodeid]['test'] or self.G.node[nodeid]['val']:
                    continue
                
                neighbors = np.array([self.id2idx[neighbor] for neighbor in self.G.neighbors(nodeid)
                    if (not self.G[nodeid][neighbor]['train_removed'])])
                
                deg[self.id2idx[nodeid]] = len(neighbors)
            else:
                neighbors = np.array([self.id2idx[neighbor] for neighbor in self.G.neighbors(nodeid)])
            
            if len(neighbors) == 0:
                continue
            
            if len(neighbors) > self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=False)
            
            elif len(neighbors) < self.max_degree:
                neighbors = np.random.choice(neighbors, self.max_degree, replace=True)
            
            adj[self.id2idx[nodeid], :] = neighbors
        
        return adj, deg
    
    def _make_label_vec(self, node):
        label = self.label_map[node]
        if isinstance(label, list):
            label_vec = np.array(label)
        else:
            label_vec = np.zeros((self.num_classes))
            class_ind = self.label_map[node]
            label_vec[class_ind] = 1
        
        return label_vec
    
    def batch_feed_dict(self, batch_nodes, val=False):
        labels = np.vstack([self._make_label_vec(node) for node in batch_nodes])
        return {
            self.placeholders['batch_size'] : len(batch_nodes),
            self.placeholders['batch']: [self.id2idx[n] for n in batch_nodes],
            self.placeholders['labels']: labels
        }, labels
        
    def get_eval_batch(self, size=None, test=False):
        nodes = self.test_nodes if test else self.val_nodes
        nodes = nodes if size is None else np.random.choice(nodes, size, replace=True)
        return self.batch_feed_dict(nodes)
        
    def get_train_batch(self):
        start = self.batch_num * self.batch_size
        self.batch_num += 1
        nodes = self.train_nodes[start : start + self.batch_size]
        return self.batch_feed_dict(nodes)
        
    def shuffle(self):
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0
        
    @property
    def is_done(self):
        return self.batch_num * self.batch_size > len(self.train_nodes) - self.batch_size 
