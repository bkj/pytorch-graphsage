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

import torch
from torch.autograd import Variable

from helpers import to_numpy

class NodeDataLoader(object):
    def __init__(self, data_path=None, cache_path=None, batch_size=512, max_degree=25, 
        multiclass=False, cuda=True):
        
        if data_path:
            print("Building NodeDataLoader from data", file=sys.stderr)
            self.__init(data_path, batch_size, max_degree, multiclass, cuda)
        elif cache_path:
            print("Loading NodeDataLoader from cache", file=sys.stderr)
            self.__load(cache_path)
        else:
            raise Exception('NodeDataLoader: either `data_path` or `cache_path` must != None')
    
    def __init(self, data_path, batch_size, max_degree, multiclass, cuda):
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
        self.train_adj = self.make_adjacency(G, id2idx, max_degree, train=True)
        self.adj = self.make_adjacency(G, id2idx, max_degree, train=False)
        
        # Construct node lists
        val_nodes   = [n for n in G.nodes() if G.node[n]['val']]
        test_nodes  = [n for n in G.nodes() if G.node[n]['test']]
        train_nodes = list(set(G.nodes()).difference(set(val_nodes + test_nodes)))
        
        self.nodes = {
            "train" : np.array(train_nodes),
            "val"   : np.array(val_nodes),
            "test"  : np.array(test_nodes),
        }
        
        self.batch_size  = batch_size
        self.id2idx      = id2idx
        self.class_map   = class_map
        self.features    = features
        self.feature_dim = features.shape[1]
        self.multiclass  = multiclass
        self.cuda        = cuda
        self.__to_torch()
        
    def __to_torch(self):
        self.train_adj = Variable(torch.LongTensor(self.train_adj))
        self.adj = Variable(torch.LongTensor(self.adj))
        self.features = Variable(torch.FloatTensor(self.features))
        
        if self.cuda:
            self.train_adj = self.train_adj.cuda()
            self.adj = self.adj.cuda()
            self.features = self.features.cuda()
    
    def __load(self, cache_path):
        f = h5py.File(cache_path)
        
        self.id2idx       = cPickle.loads(f['id2idx'].value)
        self.batch_size   = f['batch_size'].value
        self.class_map    = cPickle.loads(f['class_map'].value)
        self.num_classes  = f['num_classes'].value
        self.features     = f['features'].value
        
        self.feature_dim  = f['feature_dim'].value
        self.multiclass   = f['multiclass'].value
        self.cuda         = f['cuda'].value
        
        self.train_adj    = f['train_adj'].value
        self.adj          = f['adj'].value
        
        self.nodes = {
            "train" : cPickle.loads(f['nodes/train'].value),
            "val"   : cPickle.loads(f['nodes/val'].value),
            "test"  : cPickle.loads(f['nodes/test'].value),
        }
        
        f.close()
        
        self.__to_torch()
    
    def save(self, outpath):
        f = h5py.File(outpath)
        
        f['batch_size']  = self.batch_size
        f['num_classes'] = self.num_classes        
        f['id2idx']      = cPickle.dumps(self.id2idx)
        f['class_map']   = cPickle.dumps(self.class_map)
        f['features']    = to_numpy(self.features)
        
        f['feature_dim'] = self.feature_dim
        f['multiclass']  = self.multiclass
        f['cuda']        = self.cuda
        
        f['train_adj']   = to_numpy(self.train_adj)
        f['adj']         = to_numpy(self.adj)
        
        f['nodes/train'] = cPickle.dumps(self.nodes['train'])
        f['nodes/val']   = cPickle.dumps(self.nodes['val'])
        f['nodes/test']  = cPickle.dumps(self.nodes['test'])
        
        f.close()
    
    def make_adjacency(self, G, id2idx, max_degree, train=True):
        adj = (np.zeros((len(id2idx) + 1, max_degree)) + len(id2idx)).astype(int)
        
        for node_id in G.nodes():
            if train:
                if G.node[node_id]['test'] or G.node[node_id]['val']:
                    continue
                
                neighbors = np.array([id2idx[neighbor] for neighbor in G.neighbors(node_id)
                    if (not G[node_id][neighbor]['train_removed'])])
                
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
        
        return adj
    
    def _make_label_vec(self, node):
        """ force 2d array """
        label = self.class_map[node]
        if isinstance(label, list):
            return np.array(label)
        else:
            return np.array([label])
    
    def iterate(self, mode, shuffle=False):
        nodes = self.nodes[mode]
        
        if shuffle:
            idx = np.random.permutation(len(nodes)).astype(int)
        else:
            idx = np.arange(len(nodes)).astype(int)
        
        n_chunks = idx.shape[0] // self.batch_size + 1
        for chunk_id, chunk in enumerate(np.array_split(idx, n_chunks)):
            # Get batch
            ids = [self.id2idx[n] for n in nodes[chunk]]
            targets = np.vstack([self._make_label_vec(node) for node in nodes[chunk]])
            
            # Convert to torch
            ids = Variable(torch.LongTensor(ids))
            if self.multiclass:
                targets = Variable(torch.FloatTensor(targets))
            else:
                targets = Variable(torch.LongTensor(targets))
            
            if self.cuda:
                ids, targets = ids.cuda(), targets.cuda()
            
            yield ids, targets, chunk_id / n_chunks



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
