#!/usr/bin/env python

"""
    nn_modules.py
"""

import torch
from torch import nn

# --
# Preprocessers

class IdentityPrep(nn.Module):
    def __init__(self, input_dim):
        """ Example of preprocessor -- doesn't do anything """
        super(IdentityPrep, self).__init__()
        self.input_dim = input_dim
        
    @property
    def output_dim(self):
        return self.input_dim
    
    def forward(self, ids, features, adj):
        return features

prep_lookup = {
    "identity" : IdentityPrep,
}

# --
# Aggregators

class MeanAggregator(nn.Module):
    def __init__(self, input_dim, output_dim, activation, combine_fn=lambda x: torch.cat(x, dim=1)):
        super(MeanAggregator, self).__init__()
        
        self.fc_x = nn.Linear(input_dim, output_dim, bias=False)
        self.fc_neib = nn.Linear(input_dim, output_dim, bias=False)
        
        self.output_dim_ = output_dim
        self.activation = activation
        self.combine_fn = combine_fn
    
    @property
    def output_dim(self):
        tmp = torch.zeros((1, self.output_dim_))
        return self.combine_fn([tmp, tmp]).size(1)
        
    def forward(self, x, neibs):
        x_emb = self.fc_x(x)
        
        # !! Be careful w/ dimensions here -- messed it up the first time
        agg_neib = neibs.view(x.size(0), -1, neibs.size(1))
        agg_neib = agg_neib.mean(dim=1)
        neib_emb = self.fc_neib(agg_neib)
        
        out = self.combine_fn([x_emb, neib_emb])
        if self.activation:
            out = self.activation(out)
        
        return out

aggregator_lookup = {
    "mean" : MeanAggregator,
}