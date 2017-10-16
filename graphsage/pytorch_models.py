#!/usr/bin/env python

"""
    pytorch_models.py
"""

import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from graphsage.pytorch_aggregators import *

# aggs = {
#     "mean" : MeanAggregator,
#     "seq" : SeqAggregator,
#     "meanpool" : MeanPoolingAggregator,
#     "maxpool" : MaxPoolingAggregator,
#     "gcn" : GCNAggregator,
# }

class SupervisedGraphsage(nn.Module):
    def __init__(self, input_dim, num_classes, layer_infos, learning_rate, weight_decay, concat=True):
        
        super(SupervisedGraphsage, self).__init__()
        
        self.layer_infos = layer_infos
        
        # --
        # Define network
        
        dim_mult = 2 if concat else 1
        
        # "early"
        self.hop2_self = nn.Linear(input_dim, layer_infos[2]['output_dim'])
        self.hop1_self = nn.Linear(input_dim, layer_infos[1]['output_dim'])
        
        self.hop2_neib = nn.Linear(input_dim, layer_infos[2]['output_dim'])
        self.hop1_neib = nn.Linear(2 * layer_infos[2]['output_dim'], layer_infos[1]['output_dim'])
        
        # "final"
        self.fc = nn.Linear(dim_mult * layer_infos[1]['output_dim'], num_classes)
        
        # --
        # Optimizer
        
        self.opt = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
        # dims = [(0 if features is None else features.shape[1]) + identity_dim]
        # dims.extend([layer_infos[i]['output_dim'] for i in range(len(layer_infos))])
        
        # self.node_embeddings = None
        # if identity_dim > 0:
        #    self.node_embeddings = nn.Embedding(num_embeddings=adj.size(0), embedding_dim=identity_dim)
        
        # if features is None: 
        #     assert identity_dim == 0, "Must have a positive value for identity feature dimension if no input features given."
        #     features = node_embeddings
        # else:
        #     features = Variable(torch.FloatTensor(features), requires_grad=False)
        #     if not node_embeddings is None:
        #         features = torch.cat([node_embeddings, features], dim=1)
    
    def forward(self, ids, features, adj):
        
        # Collect features
        all_feats = {0 : features[ids]}
        for i in range(1, len(self.layer_infos) + 1):
            ids = self.layer_infos[i]['sample_fn'](ids=ids, adj=adj, n_samples=self.layer_infos[i]['n_samples'])
            ids = ids.contiguous().view(-1)
            all_feats[i] = features[ids]
        
        # Predict, from furthest to nearest
        agg_neib = all_feats[2].view(all_feats[1].size(0), all_feats[2].size(1), -1)
        agg_neib = agg_neib.mean(dim=-1)
        h2 = torch.cat([self.hop2_self(all_feats[1]), self.hop2_neib(agg_neib)], dim=1)
        h2 = F.relu(h2)
        
        agg_neib = h2.view(all_feats[0].size(0), h2.size(1), -1)
        agg_neib = agg_neib.mean(dim=-1)
        h1 = torch.cat([self.hop1_self(all_feats[0]), self.hop1_neib(agg_neib)], dim=-1)
        
        # h = F.normalize(h1, dim=1)
        h = h1
        return self.fc(h)
    
    def train_step(self, ids, features, adj, labels):
        self.opt.zero_grad()
        preds = self(ids, features, adj)
        loss = F.multilabel_soft_margin_loss(preds, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.parameters(), 5)
        self.opt.step()
        return preds, loss.data[0]
        
    
        # # Aggregate
        # aggregated, aggregators = self.aggregate(
        #     samples=samples, 
        #     input_features=features,
        #     dims=dims,
        #     num_samples=[layer_info.num_samples for layer_info in self.layer_infos],
        #     support_sizes=support_sizes,
        #     concat=concat,
        #     model_size=model_size
        # )
        
        # # Normalize
        # normed_aggregated = tf.nn.l2_normalize(aggregated, 1)
        
        # # Predict
        # dim_mult = 2 if concat else 1
        # fc = Dense(
        #     dim_mult * dims[-1],
        #     num_classes,
        #     dropout=self.dropout,
        #     act=lambda x : x,
        # )
        # node_predictions = fc(normed_aggregated)
        
        # # --
        # # Define loss
        
        # self.loss = 0
        
        # # regularization
        # for aggregator in aggregators:
        #     for var in aggregator.vars.values():
        #         self.loss += weight_decay * tf.nn.l2_loss(var)
        
        # for var in fc.vars.values():
        #     self.loss += weight_decay * tf.nn.l2_loss(var)
       
        # # classification
        # if sigmoid:
        #     self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #             logits=node_predictions,
        #             labels=placeholders['labels']))
        # else:
        #     self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #             logits=node_predictions,
        #             labels=placeholders['labels']))
        
        # # --
        # # Gradients
        
        # self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # grads_and_vars = self.optimizer.compute_gradients(self.loss)
        # clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) for grad, var in grads_and_vars]
        # self.grad, _ = clipped_grads_and_vars[0]
        # self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        
        # # --
        # # Predictions
        
        # if sigmoid:
        #     self.preds = tf.nn.sigmoid(node_predictions)
        # else:
        #     self.preds = tf.nn.softmax(node_predictions)
    
    def aggregate(self, samples, input_features, dims, num_samples, support_sizes, concat, name=None, model_size="small"):
        
        hidden = [tf.nn.embedding_lookup([input_features], node_samples) for node_samples in samples]
        
        aggregators = []
        for layer in range(len(num_samples)):
            if layer == len(num_samples) - 1:
                act = lambda x: x
            else:
                act = tf.nn.relu
            
            dim_mult = 2 if concat and (layer != 0) else 1
            
            aggregator = self.aggregator(
                input_dim=dim_mult * dims[layer],
                output_dim=dims[layer+1],
                act=act,
                dropout=self.dropout,
                name=name,
                concat=concat,
                model_size=model_size,
            )
            
            aggregators.append(aggregator)
            
            next_hidden = []
            for hop in range(len(num_samples) - layer):
                neib_dims = [
                    self.batch_size * support_sizes[hop],
                    num_samples[len(num_samples) - hop - 1],
                    dim_mult * dims[layer]
                ]
                next_hidden.append(aggregator((hidden[hop], tf.reshape(hidden[hop + 1], neib_dims))))
            
            hidden = next_hidden
        
        return hidden[0], aggregators