#!/usr/bin/env python

"""
    nn_modules.py
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable

import numpy as np
from scipy import sparse
from helpers import to_numpy

# --
# Samplers

class UniformNeighborSampler(object):
    """
        Samples from a "dense 2D edgelist", which looks like
        
            [
                [1, 2, 3, ..., 1],
                [1, 3, 3, ..., 3],
                ...
            ]
        
        stored as torch.LongTensor. 
        
        This relies on a preprocessing step where we sample _exactly_ K neighbors
        for each node -- if the node has less than K neighbors, we upsample w/ replacement
        and if the node has more than K neighbors, we downsample w/o replacement.
        
        This seems like a "definitely wrong" thing to do -- but it runs pretty fast, and
        I don't know what kind of degradation it causes in practice.
    """
    
    def __init__(self, adj):
        self.adj = adj
    
    def __call__(self, ids, n_samples=-1):
        tmp = self.adj[ids]
        perm = torch.randperm(tmp.size(1))
        if ids.is_cuda:
            perm = perm.cuda()
        
        tmp = tmp[:,perm]
        return tmp[:,:n_samples]


class SparseUniformNeighborSampler(object):
    """
        Samples from "sparse 2D edgelist", which looks like
        
            [
                [0, 0, 0, 0, ..., 0],
                [1, 2, 3, 0, ..., 0],
                [1, 3, 0, 0, ..., 0],
                ...
            ]
        
        stored as a scipy.sparse.csr_matrix.
        
        The first row is a "dummy node", so there's an "off-by-one" issue vs `feats`.
        Have to increment/decrement by 1 in a couple of places.  In the regular
        uniform sampler, this "dummy node" is at the end.
        
        Ideally, obviously, we'd be doing this sampling on the GPU.  But it does not
        appear that torch.sparse.LongTensor can support this ATM.
    """
    def __init__(self, adj,):
        assert sparse.issparse(adj), "SparseUniformNeighborSampler: not sparse.issparse(adj)"
        self.adj = adj
        
        idx, partial_degrees = np.unique(adj.nonzero()[0], return_counts=True)
        self.degrees = np.zeros(adj.shape[0]).astype(int)
        self.degrees[idx] = partial_degrees
        
    def __call__(self, ids, n_samples=128):
        assert n_samples > 0, 'SparseUniformNeighborSampler: n_samples must be set explicitly'
        is_cuda = ids.is_cuda
        
        ids = to_numpy(ids)
        
        tmp = self.adj[ids]
        
        sel = np.random.choice(self.adj.shape[1], (ids.shape[0], n_samples))
        sel = sel % self.degrees[ids].reshape(-1, 1)
        tmp = tmp[
            np.arange(ids.shape[0]).repeat(n_samples).reshape(-1),
            np.array(sel).reshape(-1)
        ]
        tmp = np.asarray(tmp).squeeze() 
        
        tmp = Variable(torch.LongTensor(tmp))
        
        if is_cuda:
            tmp = tmp.cuda()
        
        return tmp


sampler_lookup = {
    "uniform_neighbor_sampler" : UniformNeighborSampler,
    "sparse_uniform_neighbor_sampler" : SparseUniformNeighborSampler,
}

# --
# Preprocessers

class IdentityPrep(nn.Module):
    def __init__(self, input_dim, n_nodes=None):
        """ Example of preprocessor -- doesn't do anything """
        super(IdentityPrep, self).__init__()
        self.input_dim = input_dim
    
    @property
    def output_dim(self):
        return self.input_dim
    
    def forward(self, ids, feats, layer_idx=0):
        return feats


class NodeEmbeddingPrep(nn.Module):
    def __init__(self, input_dim, n_nodes, embedding_dim=64):
        """ adds node embedding """
        super(NodeEmbeddingPrep, self).__init__()
        
        self.n_nodes = n_nodes
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(num_embeddings=n_nodes + 1, embedding_dim=embedding_dim)
        self.fc = nn.Linear(embedding_dim, embedding_dim) # Affine transform, for changing scale + location
    
    @property
    def output_dim(self):
        if self.input_dim:
            return self.input_dim + self.embedding_dim
        else:
            return self.embedding_dim
    
    def forward(self, ids, feats, layer_idx=0):
        if layer_idx > 0:
            embs = self.embedding(ids)
        else:
            # Don't look at node's own embedding for prediction, or you'll probably overfit a lot
            embs = self.embedding(Variable(ids.clone().data.zero_() + self.n_nodes))
        
        embs = self.fc(embs)
        if self.input_dim:
            return torch.cat([feats, embs], dim=1)
        else:
            return embs


class LinearPrep(nn.Module):
    def __init__(self, input_dim, n_nodes, output_dim=32):
        """ adds node embedding """
        super(LinearPrep, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        self.output_dim = output_dim
    
    def forward(self, ids, feats, layer_idx=0):
        return self.fc(feats)


prep_lookup = {
    "identity" : IdentityPrep,
    "node_embedding" : NodeEmbeddingPrep,
    "linear" : LinearPrep,
}

# --
# Aggregators

class AggregatorMixin(object):
    @property
    def output_dim(self):
        tmp = torch.zeros((1, self.output_dim_))
        return self.combine_fn([tmp, tmp]).size(1)


class MeanAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MeanAggregator, self).__init__()
        
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)
        
        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn
    
    def forward(self, x, neibs):
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1)) # !! Careful
        agg_neib = agg_neib.mean(dim=1) # Careful
        
        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.activation:
            out = self.activation(out)
        
        return out


class PoolAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, pool_fn, activation, hidden_dim=512, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(PoolAggregator, self).__init__()
        
        self.mlp = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU()
        ])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)
        
        self.output_dim_ = output_dim
        self.activation = activation
        self.pool_fn = pool_fn
        self.combine_fn = combine_fn
    
    def forward(self, x, neibs):
        h_neibs = self.mlp(neibs)
        agg_neib = h_neibs.view(x.size(0), -1, h_neibs.size(1))
        agg_neib = self.pool_fn(agg_neib)
        
        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.activation:
            out = self.activation(out)
        
        return out


class MaxPoolAggregator(PoolAggregator):
    def __init__(self, input_dim, output_dim, activation, hidden_dim=512, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MaxPoolAggregator, self).__init__(**{
            "input_dim" : input_dim,
            "output_dim" : output_dim,
            "pool_fn" : lambda x: x.max(dim=1)[0],
            "activation" : activation,
            "hidden_dim" : hidden_dim,
            "combine_fn" : combine_fn,
        })


class MeanPoolAggregator(PoolAggregator):
    def __init__(self, input_dim, output_dim, activation, hidden_dim=512, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MeanPoolAggregator, self).__init__(**{
            "input_dim" : input_dim,
            "output_dim" : output_dim,
            "pool_fn" : lambda x: x.mean(dim=1),
            "activation" : activation,
            "hidden_dim" : hidden_dim,
            "combine_fn" : combine_fn,
        })


class LSTMAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation, 
        hidden_dim=512, bidirectional=False, combine_fn=lambda x: torch.cat(x, dim=1)):
        
        super(LSTMAggregator, self).__init__()
        assert not hidden_dim % 2, "LSTMAggregator: hiddem_dim % 2 != 0"
        
        self.lstm = nn.LSTM(input_dim, hidden_dim // (1 + bidirectional), bidirectional=bidirectional, batch_first=True)
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(hidden_dim, output_dim, bias=False)
        
        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn
    
    def forward(self, x, neibs):
        x_emb = self.fc_x(x)
        
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib, _ = self.lstm(agg_neib)
        agg_neib = agg_neib[:,-1,:] # !! Taking final state, but could do something better (eg attention)
        neib_emb = self.fc_neib(agg_neib)
        
        out = self.combine_fn([x_emb, neib_emb])
        if self.activation:
            out = self.activation(out)
        
        return out


class AttentionAggregator(nn.Module, AggregatorMixin):
    def __init__(self, input_dim, output_dim, activation, hidden_dim=32, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(AttentionAggregator, self).__init__()
        
        self.att = nn.Sequential(*[
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        ])
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)
        
        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn
    
    def forward(self, x, neibs):
        # Compute attention weights
        neib_att = self.att(neibs)
        x_att    = self.att(x)
        neib_att = neib_att.view(x.size(0), -1, neib_att.size(1))
        x_att    = x_att.view(x_att.size(0), x_att.size(1), 1)
        ws       = F.softmax(torch.bmm(neib_att, x_att).squeeze())
        
        # Weighted average of neighbors
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = torch.sum(agg_neib * ws.unsqueeze(-1), dim=1)
        
        out = self.combine_fn([self.fc_x(x), self.fc_neib(agg_neib)])
        if self.activation:
            out = self.activation(out)
        
        return out


aggregator_lookup = {
    "mean" : MeanAggregator,
    "max_pool" : MaxPoolAggregator,
    "mean_pool" : MeanPoolAggregator,
    "lstm" : LSTMAggregator,
    "attention" : AttentionAggregator,
}
