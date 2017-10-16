#!/usr/bin/env python

"""
    models.py
"""

from __future__ import division
from __future__ import print_function

from functools import partial

import torch
from torch import nn

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

# --
# Model

class GSSupervised(nn.Module):
    def __init__(self, input_dim, num_classes, aggregator, layer_specs, learning_rate, weight_decay):
        super(GSSupervised, self).__init__()
        
        # --
        # Define network
        
        agg_layers = []
        for spec in layer_specs:
            agg = aggregator_lookup[aggregator](
                input_dim=input_dim,
                output_dim=spec['output_dim'],
                activation=spec['activation'],
            )
            agg_layers.append(agg)
            input_dim = agg.output_dim # May not be the same as spec['output_dim']
        
        self.agg_layers = nn.Sequential(*agg_layers)
        self.fc = nn.Linear(input_dim, num_classes, bias=True)
        
        # --
        # Define samplers
        
        self.sampler_fns = [partial(s['sample_fn'], n_samples=s['n_samples']) for s in layer_specs]
        
        # --
        # Optimizer
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    def _sample(self, ids, features, adj):
        all_feats = [features[ids]]
        for sampler_fn in self.sampler_fns:
            ids = sampler_fn(ids=ids, adj=adj).contiguous().view(-1)
            all_feats.append(features[ids])
        
        return all_feats
    
    def forward(self, ids, features, adj):
        # Collect features for points in neighborhoods of ids
        all_feats = self._sample(ids, features, adj)
        
        # Sequentially apply layers, per original (little weird, IMO)
        # Each iteration reduces length of array by one
        for agg_layer in self.agg_layers.children():
            all_feats = [agg_layer(all_feats[k], all_feats[k + 1]) for k in range(len(all_feats) - 1)]
        
        assert len(all_feats) == 1, "len(all_feats) != 1"
        
        out = F.normalize(all_feats[0], dim=1)
        return self.fc(out)
    
    def train_step(self, ids, features, adj, labels):
        self.optimizer.zero_grad()
        preds = self(ids, features, adj)
        loss = F.multilabel_soft_margin_loss(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return preds

