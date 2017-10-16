#!/usr/bin/env python

"""
    supervised_models.py
"""

import tensorflow as tf
from collections import namedtuple

from graphsage.layers import Layer, Dense
from graphsage.inits import glorot, zeros

SAGEInfo = namedtuple("SAGEInfo", [
    'layer_name',
    'neib_sampler',
    'num_samples',
    'output_dim',
])

class MeanAggregator(Layer):
    def __init__(self, input_dim, output_dim, act=tf.nn.relu, **kwargs):
        
        super(MeanAggregator, self).__init__(**kwargs)
        
        self.act = act
        
        with tf.variable_scope(self.name + '_vars'):
            self.vars['neib_weights'] = glorot([input_dim, output_dim], name='neib_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim], name='self_weights')
        
    def _call(self, inputs):
        self_vecs, neib_vecs = inputs
        
        neib_means = tf.reduce_mean(neib_vecs, axis=1)
        
        output = tf.concat([
            tf.matmul(self_vecs, self.vars["self_weights"]),
            tf.matmul(neib_means, self.vars['neib_weights'])
        ], axis=1)
       
        return self.act(output)


class SupervisedGraphsage(object):
    def __init__(self, num_classes, placeholders, features, adj,
            layer_infos, learning_rate, weight_decay, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid=False, identity_dim=0, **kwargs):
        
        name = self.__class__.__name__.lower()
        self.name = name
        
        self.aggregator  = MeanAggregator
        self.batch_size  = placeholders["batch_size"]
        self.layer_infos = layer_infos
        
        dims = [50, 128, 128]
        
        features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
    
        # Sample
        samples = [placeholders["batch"]] # Node itself
        support_size = 1 # Node itself
        support_sizes = [support_size] # Node itself
        
        inds = range(len(layer_infos))
        for i, j in zip(inds, reversed(inds)):
            support_size *= layer_infos[j].num_samples
            support_sizes.append(support_size)
            node = layer_infos[j].neib_sampler(ids=samples[i], num_samples=layer_infos[j].num_samples)
            node = tf.reshape(node, [support_size * self.batch_size,])
            samples.append(node)
        
        # Aggregate
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        hidden = [tf.nn.embedding_lookup([features], node_samples) for node_samples in samples]
        
        agg2 = self.aggregator(
            input_dim=dims[0],
            output_dim=dims[1],
            act=tf.nn.relu,
        )
        agg1 = self.aggregator(
            input_dim=2 * dims[1],
            output_dim=dims[2],
            act=lambda x: x,
        )
        
        agg_size = [
            self.batch_size * support_sizes[0],
            num_samples[-1],
            dims[0],
        ]
        print('agg_size', agg_size)
        h1 = agg2((hidden[0], tf.reshape(hidden[1], agg_size)))
        
        agg_size = [
            self.batch_size * support_sizes[1],
            num_samples[-2],
            dims[0],
        ]
        print('agg_size', agg_size)
        h2 = agg2((hidden[1], tf.reshape(hidden[2], agg_size)))
        
        agg_size = [
            self.batch_size * support_sizes[0],
            num_samples[-1],
            2 * dims[1],
        ]
        aggregated = agg1((h1, tf.reshape(h2, agg_size)))
        
        # Normalize
        normed_aggregated = tf.nn.l2_normalize(aggregated, 1)
        
        # Predict
        fc = Dense(2 * dims[-1], num_classes)
        node_predictions = fc(normed_aggregated)
        
        # --
        # Define loss
        
        self.loss = 0
        
        # regularization
        for layer in [agg1, agg2, fc]:
            for var in layer.vars.values():
                self.loss += weight_decay * tf.nn.l2_loss(var)
       
        # classification
        self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                logits=node_predictions,
                labels=placeholders['labels']))
        
        # --
        # Gradients
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        
        # --
        # Predictions
        
        self.preds = tf.nn.sigmoid(node_predictions)
