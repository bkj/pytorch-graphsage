#!/usr/bin/env python

"""
    supervised_models.py
"""

import tensorflow as tf
from collections import namedtuple

from graphsage.layers import Dense
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, \
    MeanPoolingAggregator, SeqAggregator, GCNAggregator

aggs = {
    "mean" : MeanAggregator,
    "seq" : SeqAggregator,
    "meanpool" : MeanPoolingAggregator,
    "maxpool" : MaxPoolingAggregator,
    "gcn" : GCNAggregator,
}

SAGEInfo = namedtuple("SAGEInfo", [
    'layer_name',
    'neib_sampler',
    'num_samples',
    'output_dim',
])

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        
        self.name = name
        
        self.logging = kwargs.get('logging', False)
        
        self.vars = {}



class SupervisedGraphsage(Model):
    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos, learning_rate, weight_decay, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid=False, identity_dim=0, **kwargs):
        
        super(SupervisedGraphsage, self).__init__(**kwargs)
        
        self.aggregator   = aggs[aggregator_type]
        self.batch_size   = placeholders["batch_size"]
        self.dropout      = placeholders["dropout"]
        self.layer_infos  = layer_infos
        
        dims = [(0 if features is None else features.shape[1]) + identity_dim]
        dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        
        node_embeddings = None
        if identity_dim > 0:
           node_embeddings = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            
            features = node_embeddings
        else:
            features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not node_embeddings is None:
                features = tf.concat([node_embeddings, features], axis=1)
        
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        
        # Sample
        samples1, support_sizes1 = self.sample(placeholders["batch"], self.layer_infos)
        
        # Aggregate
        self.outputs1, self.aggregators = self.aggregate(
            samples=samples1, 
            input_features=[features],
            dims=dims,
            num_samples=[layer_info.num_samples for layer_info in self.layer_infos],
            support_sizes=support_sizes1,
            concat=concat,
            model_size=model_size
        )
        
        # Normalize
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        
        # Predict
        dim_mult = 2 if concat else 1
        self.node_pred = Dense(
            dim_mult * dims[-1],
            num_classes,
            dropout=self.dropout,
            act=lambda x : x,
        )
        self.node_preds = self.node_pred(self.outputs1)
        
        # --
        # Define loss
        
        # regularization
        self.loss = 0
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += weight_decay * tf.nn.l2_loss(var)
        
        for var in self.node_pred.vars.values():
            self.loss += weight_decay * tf.nn.l2_loss(var)
       
        # classification
        if sigmoid:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=placeholders['labels']))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=placeholders['labels']))
            
        tf.summary.scalar('loss', self.loss)
        
        # --
        # Gradients
        
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        
        # --
        # Predictions
        
        if sigmoid:
            self.preds = tf.nn.sigmoid(self.node_preds)
        else:
            self.preds = tf.nn.softmax(self.node_preds)
            
        
    def sample(self, inputs, layer_infos):
        samples = [inputs]
        support_size = 1
        support_sizes = [support_size]
        for k in range(len(layer_infos)):
            t = len(layer_infos) - k - 1
            support_size *= layer_infos[t].num_samples
            node = layer_infos[t].neib_sampler((samples[k], layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [support_size * self.batch_size,]))
            support_sizes.append(support_size)
        
        return samples, support_sizes
    
    def aggregate(self, samples, input_features, dims, num_samples, support_sizes,
            name=None, concat=False, model_size="small"):
        
        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        
        aggregators = []
        for layer in range(len(num_samples)):
            if layer == len(num_samples) - 1:
                act = lambda x: x
            else:
                act = tf.nn.relu
            
            dim_mult = 2 if concat and (layer != 0) else 1
            
            aggregator = self.aggregator(
                dim_mult * dims[layer],
                dims[layer+1],
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