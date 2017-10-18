#!/usr/bin/env python

"""
    models.py
"""

from __future__ import division
from __future__ import print_function

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from lr import LRSchedule

# --
# Model

class GSSupervised(nn.Module):
    def __init__(self, input_dim, n_classes, layer_specs, 
        aggregator_class, prep_class, lr_init=0.01, weight_decay=0.0,
        lr_schedule='constant', epochs=10):
        super(GSSupervised, self).__init__()
        
        # --
        # Define network
        
        self.prep = prep_class(input_dim=input_dim)
        
        input_dim = self.prep.output_dim
        
        agg_layers = []
        for spec in layer_specs:
            agg = aggregator_class(
                input_dim=input_dim,
                output_dim=spec['output_dim'],
                activation=spec['activation'],
            )
            agg_layers.append(agg)
            input_dim = agg.output_dim # May not be the same as spec['output_dim']
        
        self.agg_layers = nn.Sequential(*agg_layers)
        self.fc = nn.Linear(input_dim, n_classes, bias=True)
        
        # --
        # Setup samplers
        
        self.train_sample_fns = [partial(s['sample_fn'], n_samples=s['n_train_samples']) for s in layer_specs]
        self.val_sample_fns = [partial(s['sample_fn'], n_samples=s['n_val_samples']) for s in layer_specs]
        
        # --
        # Define optimizer
        
        self.lr_scheduler = partial(getattr(LRSchedule, lr_schedule), lr_init=lr_init)
        self.lr = self.lr_scheduler(0.0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=weight_decay)
    
    def _sample(self, ids, feats, adj, train):
        sample_fns = self.train_sample_fns if train else self.val_sample_fns
        
        all_feats = [feats[ids]]
        for sampler_fn in sample_fns:
            ids = sampler_fn(ids=ids, adj=adj).contiguous().view(-1)
            all_feats.append(feats[ids])
        
        return all_feats
    
    def forward(self, ids, feats, adj, train=True):
        # Prep feats
        feats = self.prep(ids, feats, adj)
        
        # Collect feats for points in neighborhoods of ids
        all_feats = self._sample(ids, feats, adj, train=train)
        
        # Sequentially apply layers, per original (little weird, IMO)
        # Each iteration reduces length of array by one
        for agg_layer in self.agg_layers.children():
            all_feats = [agg_layer(all_feats[k], all_feats[k + 1]) for k in range(len(all_feats) - 1)]
        
        assert len(all_feats) == 1, "len(all_feats) != 1"
        
        out = F.normalize(all_feats[0], dim=1) # ??
        return self.fc(out)
    
    def set_progress(self, progress):
        self.lr = self.lr_scheduler(progress)
        LRSchedule.set_lr(self.optimizer, self.lr)
    
    def train_step(self, ids, feats, adj, targets, loss_fn):
        self.optimizer.zero_grad()
        preds = self(ids, feats, adj)
        loss = loss_fn(preds, targets.squeeze())
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        self.optimizer.step()
        return preds

