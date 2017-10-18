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
        # return {
        #     "micro" : metrics.f1_score(y_true, y_pred, average="micro"),
        #     "macro" : metrics.f1_score(y_true, y_pred, average="macro"),
        # }
        return (y_pred == y_true.squeeze()).mean()

# --
# Problem definition

class NodeProblem(object):
    def __init__(self, problem_path, cuda=True):
        
        print('NodeProblem: loading started')
        
        f = h5py.File(problem_path)
        self.task      = f['task'].value
        self.n_classes = f['n_classes'].value
        self.feats     = f['feats'].value
        self.folds     = f['folds'].value
        self.targets   = f['targets'].value
        self.adj       = f['adj'].value
        self.train_adj = f['train_adj'].value
        f.close()
        
        self.__validate_problem()
        
        self.feats_dim = self.feats.shape[1]
        self.n_nodes   = self.adj.shape[0]
        self.cuda      = cuda
        self.__to_torch()
        
        self.nodes = {
            "train" : np.where(self.folds == 'train')[0],
            "val"   : np.where(self.folds == 'val')[0],
            "test"  : np.where(self.folds == 'test')[0],
        }
        
        # # >>
        # Drop some nodes from "train" nodes -- semi-supervised
        # alpha = 0.10
        # self.nodes['train'] = np.random.choice(
        #     self.nodes['train'], 
        #     size=int(self.nodes['train'].shape[0] * alpha)
        # )
        # print("self.nodes['train'].shape[0]", self.nodes['train'].shape[0])
        # # <<
        
        self.loss_fn = getattr(ProblemLosses, self.task)
        self.metric_fn = getattr(ProblemMetrics, self.task)
        
        print('NodeProblem: loading finished')
        
    def __validate_problem(self):
        """ validate input format """
        assert self.feats.shape[0] == self.targets.shape[0]
        assert self.feats.shape[0] == self.folds.shape[0]
        assert self.adj.shape[0] == self.train_adj.shape[0]
        assert self.adj.shape[0] == (self.feats.shape[0] + 1)
        assert len(self.targets.shape) == 2
    
    def __to_torch(self):
        self.train_adj = Variable(torch.LongTensor(self.train_adj))
        self.adj = Variable(torch.LongTensor(self.adj))
        self.feats = Variable(torch.FloatTensor(self.feats))
        
        if self.cuda:
            self.train_adj = self.train_adj.cuda()
            self.adj = self.adj.cuda()
            self.feats = self.feats.cuda()
    
    def __batch_to_torch(self, mids, targets):
        """ convert batch to torch """
        mids = Variable(torch.LongTensor(mids))
        
        if self.task == 'multilabel_classification':
            targets = Variable(torch.FloatTensor(targets))
        elif self.task == 'classification':
            targets = Variable(torch.LongTensor(targets))
        else:
            raise Exception('NodeDataLoader: unknown task: %s' % self.task)
        
        if self.cuda:
            mids, targets = mids.cuda(), targets.cuda()
        
        return mids, targets
    
    def iterate(self, mode, batch_size=512, shuffle=False):
        nodes = self.nodes[mode]
        
        idx = np.arange(nodes.shape[0])
        if shuffle:
            idx = np.random.permutation(idx)
        
        n_chunks = idx.shape[0] // batch_size + 1
        for chunk_id, chunk in enumerate(np.array_split(idx, n_chunks)):
            mids = nodes[chunk]
            targets = self.targets[mids]
            mids, targets = self.__batch_to_torch(mids, targets)
            yield mids, targets, chunk_id / n_chunks
