#!/usr/bin/env python

"""
    data_loader_2.py
"""

from __future__ import division
from __future__ import print_function

import os
import sys
import cPickle
import h5py
import numpy as np
import ujson as json

import torch
from torch.autograd import Variable

from helpers import to_numpy

def read_problem(problem_path):
    f = h5py.File(problem_path)
    problem = {
        "idx2fold"  : cPickle.loads(f['folds'].value),
        "task"      : f['task'].value,
        "n_classes" : f['n_classes'].value,
        "feats"     : f['feats'].value,
        "train_adj" : f['adj'].value,
        "adj"       : f['train_adj'].value,
    }
    f.close()
    return problem

class NodeDataLoader(object):
    def __init__(self, problem_path, cuda=True):
        
        problem = load_problem(problem_path)
        
        idx2fold  = problem['folds']
        task      = problem['task']
        n_classes = problem['n_classes']
        feats     = problem['feats']
        train_adj = problem['adj']
        adj       = problem['train_adj']
        
        self.feats_dim = feats.shape[1]
        self.cuda = cuda
        
        self.nodes = {
            "train" : np.array([idx for idx,fold in meta['idx2fold'].items() if fold == 'train']),
            "val"   : np.array([idx for idx,fold in meta['idx2fold'].items() if fold == 'val']),
            "test"  : np.array([idx for idx,fold in meta['idx2fold'].items() if fold == 'test']),
        }
        
        self.__to_torch()
    
    def __to_torch(self):
        self.train_adj = Variable(torch.LongTensor(self.train_adj))
        self.adj = Variable(torch.LongTensor(self.adj))
        self.feats = Variable(torch.FloatTensor(self.feats))
        
        if self.cuda:
            self.train_adj = self.train_adj.cuda()
            self.adj = self.adj.cuda()
            self.feats = self.feats.cuda()
    
    def _make_label_vec(self, node):
        """ force 2d array """
        label = self.class_map[node]
        if isinstance(label, list):
            return np.array(label)
        else:
            return np.array([label])
    
    def iterate(self, mode, batch_size=512, shuffle=False):
        nodes = self.nodes[mode]
        
        if shuffle:
            idx = np.random.permutation(nodes.shape[0])
        else:
            idx = np.arange(nodes.shape[0])
        
        n_chunks = idx.shape[0] // batch_size + 1
        for chunk_id, chunk in enumerate(np.array_split(idx, n_chunks)):
            # Get batch
            mids    = nodes[chunk]
            targets = np.vstack([self._make_label_vec(mid) for mid in mids])
            
            # Convert to torch
            mids = Variable(torch.LongTensor(mids))
            
            if self.task == 'multiclass_classification':
                targets = Variable(torch.FloatTensor(targets))
            elif self.task == 'classification':
                targets = Variable(torch.LongTensor(targets))
            else:
                raise Exception('NodeDataLoader: unknown task')
            
            if self.cuda:
                mids, targets = mids.cuda(), targets.cuda()
            
            yield mids, targets, chunk_id / n_chunks
