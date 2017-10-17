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
import pandas as pd
import ujson as json
from networkx.readwrite import json_graph

import torch
from torch.autograd import Variable

from helpers import to_numpy

class DataLoader(object):

    def __init__(self, data_path=None, cache_path=None, **kwargs):
        
        if data_path:
            print("Building DataLoader from data", file=sys.stderr)
            self._init(data_path, **kwargs)
        elif cache_path:
            print("Loading DataLoader from cache", file=sys.stderr)
            self._load(cache_path)
        else:
            raise Exception('DataLoader: either `data_path` or `cache_path` must != None')
    
    def _to_torch(self):
        self.train_adj = Variable(torch.LongTensor(self.train_adj))
        self.adj = Variable(torch.LongTensor(self.adj))
        self.feats = Variable(torch.FloatTensor(self.feats))
        
        if self.cuda:
            self.train_adj = self.train_adj.cuda()
            self.adj = self.adj.cuda()
            self.feats = self.feats.cuda()
    
    def _make_adjacency(self, G, id2idx, max_degree, train=True):
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


class NodeDataLoader(DataLoader):
    def _init(self, data_path, **kwargs):
        G, self.feats, self.id2idx, self.id2class, _ = load_data(data_path)
        
        # Construct adjacency matrices
        self.train_adj = self._make_adjacency(G, self.id2idx, kwargs['max_degree'], train=True)
        self.adj = self._make_adjacency(G, self.id2idx, kwargs['max_degree'], train=False)
        
        # Train/test split nodes and edges
        val_nodes   = [n for n in G.nodes() if G.node[n]['val']]
        test_nodes  = [n for n in G.nodes() if G.node[n]['test']]
        train_nodes = list(set(G.nodes()).difference(set(val_nodes + test_nodes)))
        
        self.nodes = {
            "train" : np.array(train_nodes),
            "val"   : np.array(val_nodes),
            "test"  : np.array(test_nodes),
        }
        
        # Munge self.feats
        if self.feats is not None:
            self.feats = np.vstack([self.feats, np.zeros((self.feats.shape[1],))])
        
        # Determine number of classes
        if isinstance(list(self.id2class.values())[0], list):
            self.num_classes = len(list(self.id2class.values())[0])
        else:
            self.num_classes = len(set(self.id2class.values()))
        
        # Save properties
        self.batch_size  = kwargs['batch_size']
        self.feat_dim    = self.feats.shape[1]
        self.multiclass  = kwargs['multiclass']
        self.cuda        = kwargs['cuda']
        self._to_torch()
    
    def _load(self, cache_path):
        f = h5py.File(cache_path)
        
        self.feats       = f['feats'].value
        self.id2idx      = cPickle.loads(f['id2idx'].value)
        self.id2class    = cPickle.loads(f['id2class'].value)
        
        self.num_classes = f['num_classes'].value
        self.batch_size  = f['batch_size'].value
        self.feat_dim    = f['feat_dim'].value
        self.multiclass  = f['multiclass'].value
        self.cuda        = f['cuda'].value
        
        self.train_adj   = f['train_adj'].value
        self.adj         = f['adj'].value
        
        self.nodes = {
            "train" : cPickle.loads(f['nodes/train'].value),
            "val"   : cPickle.loads(f['nodes/val'].value),
            "test"  : cPickle.loads(f['nodes/test'].value),
        }
        
        f.close()
        
        self._to_torch()
    
    def save(self, outpath):
        f = h5py.File(outpath)
        
        f['feats']       = to_numpy(self.feats)
        f['id2idx']      = cPickle.dumps(self.id2idx)
        f['id2class']    = cPickle.dumps(self.id2class)
        
        f['batch_size']  = self.batch_size
        f['num_classes'] = self.num_classes
        f['feat_dim']    = self.feat_dim
        f['multiclass']  = self.multiclass
        f['cuda']        = self.cuda
        
        f['train_adj']   = to_numpy(self.train_adj)
        f['adj']         = to_numpy(self.adj)
        
        f['nodes/train'] = cPickle.dumps(self.nodes['train'])
        f['nodes/val']   = cPickle.dumps(self.nodes['val'])
        f['nodes/test']  = cPickle.dumps(self.nodes['test'])
        
        f.close()
    
    def _make_label_vec(self, node):
        """ force 2d array """
        label = self.id2class[node]
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


class CPairDataLoader(DataLoader):
    def _init(self, data_path, **kwargs):
        G, self.feats, self.id2idx, _, context_pairs = load_data(data_path)
        
        # Construct adjacency matrices
        self.train_adj = self._make_adjacency(G, self.id2idx, kwargs['max_degree'], train=True)
        self.adj = self._make_adjacency(G, self.id2idx, kwargs['max_degree'], train=False)
        
        self.cpairs = {
            "train" : context_pairs,
            "val" : [e for e in G.edges() if G[e[0]][e[1]]['train_removed']]
        }
        
        # Munge self.feats
        if self.feats is not None:
            self.feats = np.vstack([self.feats, np.zeros((self.feats.shape[1],))])
                
        # Save properties
        self.batch_size  = kwargs['batch_size']
        self.feat_dim    = self.feats.shape[1]
        self.cuda        = kwargs['cuda']
        self._to_torch()
    
    def _load(self, cache_path):
        f = h5py.File(cache_path)
        
        self.feats        = f['feats'].value
        self.id2idx       = cPickle.loads(f['id2idx'].value)
        
        self.batch_size  = f['batch_size'].value
        self.feat_dim    = f['feat_dim'].value
        self.cuda        = f['cuda'].value
        
        self.train_adj   = f['train_adj'].value
        self.adj         = f['adj'].value
        
        self.cpairs = {
            "train" : cPickle.loads(f['cpairs/train'].value),
            "val"   : cPickle.loads(f['cpairs/val'].value),
        }
        
        f.close()
        
        self._to_torch()
    
    def save(self, outpath):
        f = h5py.File(outpath)
        
        f['feats']       = to_numpy(self.feats)
        f['id2idx']      = cPickle.dumps(self.id2idx)
        
        f['batch_size']  = self.batch_size
        f['feat_dim']    = self.feat_dim
        f['cuda']        = self.cuda
        
        f['train_adj']   = to_numpy(self.train_adj)
        f['adj']         = to_numpy(self.adj)
        
        f['cpairs/train'] = cPickle.dumps(self.cpairs['train'])
        f['cpairs/val']   = cPickle.dumps(self.cpairs['val'])
        
        f.close()
    
    def iterate(self, mode, shuffle=False):
        cpairs = self.cpairs[mode]
        
        if shuffle:
            idx = np.random.permutation(len(cpairs)).astype(int)
        else:
            idx = np.arange(len(cpairs)).astype(int)
        
        n_chunks = idx.shape[0] // self.batch_size + 1
        for chunk_id, chunk in enumerate(np.array_split(idx, n_chunks)):
            # Get batch
            ids1, ids2 = zip(*[(self.id2idx[n1], self.id2idx[n2]) for n1,n2 in cpairs[chunk]])
            
            # Convert to torch
            ids1 = Variable(torch.LongTensor(ids1))
            ids2 = Variable(torch.LongTensor(ids2))
            
            if self.cuda:
                ids1, ids2 = ids1.cuda(), ids2.cuda()
            
            yield ids1, ids2, chunk_id / n_chunks


def load_data(data_path, normalize=True):
    # --
    # Load
    
    G = json_graph.node_link_graph(json.load(open(os.path.join(data_path, "G.json"))))
    id2class = json.load(open(os.path.join(data_path, "class_map.json")))
    id2idx = json.load(open(os.path.join(data_path, "id_map.json")))
    
    feats = None
    if os.path.exists(os.path.join(data_path, "feats.npy")):
        feats = np.load(os.path.join(data_path, "feats.npy"))
    
    context_pairs = None
    if os.path.exists(os.path.join(data_path, 'walks.txt')):
        context_pairs = pd.read_csv(os.path.join(data_path, 'walks.txt'), header=None, sep='\t')
    
    # --
    # Format
    
    if isinstance(G.nodes()[0], int):
        conversion = lambda n : int(n)
    else:
        conversion = lambda n : str(n)
    
    if isinstance(list(id2class.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)
    
    id2idx   = {conversion(k):int(v) for k,v in id2idx.items()}
    id2class = {conversion(k):lab_conversion(v) for k,v in id2class.items()}
    
    # Remove edges not in training dataset
    for edge in G.edges():
        if (G.node[edge[0]]['val'] or G.node[edge[1]]['val'] or G.node[edge[0]]['test'] or G.node[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False
    
    # Normalize feats
    if normalize and feats is not None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id2idx[n] for n in G.nodes() if not G.node[n]['val'] and not G.node[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    # Format context_pairs (if they exist)
    if context_pairs is not None:
        context_pairs[0] = context_pairs[0].apply(conversion)
        context_pairs[1] = context_pairs[1].apply(conversion)
        context_pairs = np.array(context_pairs)
    
    return G, feats, id2idx, id2class, context_pairs
