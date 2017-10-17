#!/usr/bin/env python

"""
    problem.py
"""

from __future__ import division
from __future__ import print_function

import os
import sys
import h5py
import cPickle
import numpy as np
from sklearn import metrics

import torch
from torch.autograd import Variable
from torch.nn import functional as F

# --
# Helper classes

class ProblemLosses:
    @staticmethod
    def multilabel_classification(preds, targets):
        return F.multilabel_soft_margin_loss(preds, targets)
    
    @staticmethod
    def classification(preds, targets):
        return F.cross_entropy(preds, targets)


class ProblemMetrics:
    @staticmethod
    def multilabel_classification(y_true, y_pred):
        y_pred = (y_pred > 0).astype(int)
        return {
            "micro" : metrics.f1_score(y_true, y_pred, average="micro"),
            "macro" : metrics.f1_score(y_true, y_pred, average="macro"),
        }
    
    @staticmethod
    def classification(y_true, y_pred):
        y_pred = np.argmax(y_pred, axis=1)
        return {
            "micro" : metrics.f1_score(y_true, y_pred, average="micro"),
            "macro" : metrics.f1_score(y_true, y_pred, average="macro"),
        }

# --
# Helpers

def load_problem(problem_path):
    f = h5py.File(problem_path)
    problem = {
        "idx2class" : cPickle.loads(f['idx2class'].value),
        "idx2fold"  : cPickle.loads(f['idx2fold'].value),
        "task"      : f['task'].value,
        "n_classes" : f['n_classes'].value,
        "feats"     : f['feats'].value,
        "train_adj" : f['adj'].value, # !! Got flipped
        "adj"       : f['train_adj'].value,
    }
    f.close()
    return problem

# --
# Problem definition

class NodeProblem(object):
    def __init__(self, problem_path, cuda=True):
        
        problem = load_problem(problem_path)
        
        self.idx2class = problem['idx2class']
        self.task      = problem['task']
        self.n_classes = problem['n_classes']
        self.feats     = problem['feats']
        self.train_adj = problem['train_adj']
        self.adj       = problem['adj']
        
        self.feats_dim = self.feats.shape[1]
        self.cuda = cuda
        
        self.loss_fn = getattr(ProblemLosses, self.task)
        self.metric_fn = getattr(ProblemMetrics, self.task)
        
        self.nodes = {
            "train" : np.array([idx for idx,fold in problem['idx2fold'].items() if fold == 'train']),
            "val"   : np.array([idx for idx,fold in problem['idx2fold'].items() if fold == 'val']),
            "test"  : np.array([idx for idx,fold in problem['idx2fold'].items() if fold == 'test']),
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
        label = self.idx2class[node]
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
            
            if self.task == 'multilabel_classification':
                targets = Variable(torch.FloatTensor(targets))
            elif self.task == 'classification':
                targets = Variable(torch.LongTensor(targets))
            else:
                raise Exception('NodeDataLoader: unknown task: %s' % self.task)
            
            if self.cuda:
                mids, targets = mids.cuda(), targets.cuda()
            
            yield mids, targets, chunk_id / n_chunks
