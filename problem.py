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
from scipy import sparse
from sklearn import metrics
from scipy.sparse import csr_matrix

import torch
from torch.autograd import Variable
from torch.nn import functional as F

from helpers import to_numpy

# --
# Helper classes

class ProblemLosses:
    @staticmethod
    def multilabel_classification(preds, targets):
        return F.multilabel_soft_margin_loss(preds, targets)
    
    @staticmethod
    def classification(preds, targets):
        return F.cross_entropy(preds, targets)
        
    @staticmethod
    def regression_mae(preds, targets):
        return F.l1_loss(preds, targets)


class ProblemMetrics:
    @staticmethod
    def multilabel_classification(y_true, y_pred):
        y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
        
        y_pred = (y_pred > 0).astype(int)
        return {
            "micro" : float(metrics.f1_score(y_true, y_pred, average="micro")),
            "macro" : float(metrics.f1_score(y_true, y_pred, average="macro")),
        }
    
    @staticmethod
    def classification(y_true, y_pred):
        y_true, y_pred = to_numpy(y_true).squeeze(), to_numpy(y_pred).squeeze()
        
        pred_class = np.argmax(y_pred, axis=1)
        
        try:
            roc = float(metrics.roc_auc_score(y_true, y_pred[:,1]))
        except:
            roc = None
        
        return {
            "micro" : float(metrics.f1_score(y_true, pred_class, average="micro")),
            "macro" : float(metrics.f1_score(y_true, pred_class, average="macro")),
            "roc"   : roc,
        }
    
    @staticmethod
    def regression_mae(y_true, y_pred):
        y_true, y_pred = to_numpy(y_true), to_numpy(y_pred)
        
        return float(np.abs(y_true - y_pred).mean())


# --
# Problem definition

def parse_csr_matrix(x):
    v, r, c = x
    return csr_matrix((v, (r, c)))

class NodeProblem(object):
    def __init__(self, problem_path, cuda=True):
        
        print('NodeProblem: loading started', file=sys.stderr)
        
        f = h5py.File(problem_path)
        self.task      = f['task'].value
        self.n_classes = f['n_classes'].value if 'n_classes' in f else 1 # !!
        self.feats     = f['feats'].value if 'feats' in f else None
        self.folds     = f['folds'].value
        self.targets   = f['targets'].value
        if 'sparse' in f and f['sparse'].value:
            self.adj = parse_csr_matrix(f['adj'].value)
            self.train_adj = parse_csr_matrix(f['train_adj'].value)
        else:
            self.adj = f['adj'].value
            self.train_adj = f['train_adj'].value
            
        f.close()
        
        self.feats_dim = self.feats.shape[1] if self.feats is not None else None
        self.n_nodes   = self.adj.shape[0]
        self.cuda      = cuda
        self.__to_torch()
        
        self.nodes = {
            "train" : np.where(self.folds == 'train')[0],
            "val"   : np.where(self.folds == 'val')[0],
            "test"  : np.where(self.folds == 'test')[0],
        }
        
        self.loss_fn = getattr(ProblemLosses, self.task)
        self.metric_fn = getattr(ProblemMetrics, self.task)
        
        print('NodeProblem: loading finished', file=sys.stderr)
    
    def __to_torch(self):
        if not sparse.issparse(self.adj):
            self.adj = Variable(torch.LongTensor(self.adj))
            self.train_adj = Variable(torch.LongTensor(self.train_adj))
            if self.cuda:
                self.adj = self.adj.cuda()
                self.train_adj = self.train_adj.cuda()
        
        if self.feats is not None:
            self.feats = Variable(torch.FloatTensor(self.feats))
            if self.cuda:
                self.feats = self.feats.cuda()
    
    def __batch_to_torch(self, mids, targets):
        """ convert batch to torch """
        mids = Variable(torch.LongTensor(mids))
        
        if self.task == 'multilabel_classification':
            targets = Variable(torch.FloatTensor(targets))
            assert(len(targets.size()) == 2), 'multilabel_classification targets must be 2d'
        elif self.task == 'classification':
            targets = Variable(torch.LongTensor(targets))
            assert(targets.size(1) == 1), 'classification targets must be 1d'
        elif 'regression' in self.task:
            targets = Variable(torch.FloatTensor(targets))
            assert(len(targets.size()) == 1), 'regression targets must be 1d'
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

# --
# Unsupervised

def bce_with_logits(logits, target):
    # !! Is this right?
    max_val = (-logits).clamp(min=0)
    return logits - logits * target + max_val + ((-max_val).exp() + (-logits - max_val).exp()).log()


class UnsupervisedLosses:
    @staticmethod
    def xent(anc_emb, pos_emb, neg_emb, neg_alpha=0.01):
        
        pos_sim = (anc_emb * pos_emb).sum(dim=1)
        neg_sim = torch.mm(anc_emb, neg_emb.t())
        
        pos_loss = bce_with_logits(logits=pos_sim, target=1).sum()
        neg_loss = bce_with_logits(logits=neg_sim, target=0).sum()
        
        loss = pos_loss + neg_alpha * neg_loss
        return loss / anc_emb.size(0)


class UnsupervisedMetrics:
    @staticmethod
    def xent(anc_emb, pos_emb, neg_emb, neg_alpha=0.01):
        
        pos_sim = (anc_emb * pos_emb).sum(dim=1)
        neg_sim = torch.mm(anc_emb, neg_emb.t())
        
        pos_loss = bce_with_logits(logits=pos_sim, target=1).sum()
        neg_loss = bce_with_logits(logits=neg_sim, target=0).sum()
        
        loss = pos_loss + neg_alpha * neg_loss
        loss = loss / anc_emb.size(0)
        return float(to_numpy(loss)[0])


class EdgeProblem(object):
    def __init__(self, problem_path, cuda=True):
        
        print('EdgeProblem: loading started')
        
        f = h5py.File(problem_path)
        self.task      = f['task'].value
        # self.n_classes = f['n_classes'].value if 'n_classes' in f else 1 # !!
        self.final_dim = 128
        self.feats     = f['feats'].value if 'feats' in f else None
        self.folds     = f['folds'].value
        # self.targets   = f['targets'].value
        
        # >>
        self.context_pairs = f['context_pairs'].value
        self.context_pairs_folds = f['context_pairs_folds'].value
        # <<
        
        if 'sparse' in f and f['sparse'].value:
            self.adj = parse_csr_matrix(f['adj'].value)
            self.train_adj = parse_csr_matrix(f['train_adj'].value)
        else:
            self.adj = f['adj'].value
            self.train_adj = f['train_adj'].value
        
        f.close()
        
        self.feats_dim = self.feats.shape[1] if self.feats is not None else None
        self.n_nodes   = self.adj.shape[0]
        self.cuda      = cuda
        self.__to_torch()
        
        self.nodes = {
            "train" : np.where(self.folds == 'train')[0],
            "val"   : np.where(self.folds == 'val')[0],
            "test"  : np.where(self.folds == 'test')[0],
        }
        
        # >>
        self.folds = {
            "train" : np.where(self.context_pairs_folds == 'train')[0],
            "val"   : np.where(self.context_pairs_folds == 'val')[0],
            "test"  : np.where(self.context_pairs_folds == 'test')[0],
        }
        # <<
        
        self.loss_fn = getattr(UnsupervisedLosses, 'xent')
        self.metric_fn = getattr(UnsupervisedMetrics, 'xent')
        
        print('EdgeProblem: loading finished')
    
    def __to_torch(self):
        if not sparse.issparse(self.adj):
            self.adj = Variable(torch.LongTensor(self.adj))
            self.train_adj = Variable(torch.LongTensor(self.train_adj))
            if self.cuda:
                self.adj = self.adj.cuda()
                self.train_adj = self.train_adj.cuda()
        
        if self.feats is not None:
            self.feats = Variable(torch.FloatTensor(self.feats))
            if self.cuda:
                self.feats = self.feats.cuda()
    
    def __batch_to_torch(self, mids):
        """ convert batch to torch """
        mids = Variable(torch.LongTensor(mids))
        
        if self.cuda:
            mids = mids.cuda()
        
        return mids
    
    def _negative_sampling(self, anc_ids, n=None):
        """
            simplest 'negative sampling':
                just randomly pick nodes, weighted by number of times they appear in random walks
            
            (original version samples according to degree ** 0.75 -- IDK what that does)
        """
        idx = np.random.choice(self.context_pairs.shape[0], n if n else anc_ids.shape[0], replace=True)
        return self.context_pairs[idx,0]
    
    def iterate_edges(self, mode, batch_size=512, neg_batch_size=20, shuffle=False):
        context_pairs = self.context_pairs[self.folds[mode]]
        
        idx = np.arange(context_pairs.shape[0])
        if shuffle:
            idx = np.random.permutation(idx)
        
        n_chunks = idx.shape[0] // batch_size + 1
        for chunk_id, chunk in enumerate(np.array_split(idx, n_chunks)):
            cpairs = context_pairs[chunk]
            anc_ids, pos_ids = cpairs[:,0], cpairs[:,1]
            neg_ids = self._negative_sampling(anc_ids, n=neg_batch_size)
            
            anc_ids = self.__batch_to_torch(anc_ids)
            pos_ids = self.__batch_to_torch(pos_ids)
            neg_ids = self.__batch_to_torch(neg_ids)
            yield anc_ids, pos_ids, neg_ids, chunk_id / n_chunks
    
    def iterate_nodes(self, mode, batch_size=512, shuffle=False):
        nodes = self.nodes[mode]
        
        idx = np.arange(nodes.shape[0])
        if shuffle:
            idx = np.random.permutation(idx)
        
        n_chunks = idx.shape[0] // batch_size + 1
        for chunk_id, chunk in enumerate(np.array_split(idx, n_chunks)):
            mids = nodes[chunk]
            mids = self.__batch_to_torch(mids)
            yield mids, chunk_id / n_chunks
