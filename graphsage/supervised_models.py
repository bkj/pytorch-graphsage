#!/usr/bin/env python

"""
    supervised_models.py
"""

import tensorflow as tf
from collections import namedtuple

from graphsage.layers import Dense
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, \
    MeanPoolingAggregator, SeqAggregator, GCNAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

aggs = {
    "mean": MeanAggregator,
    "seq": SeqAggregator,
    "meanpool": MeanPoolingAggregator,
    "maxpool": MaxPoolingAggregator,
    "gcn": GCNAggregator,
}

SAGEInfo = namedtuple("SAGEInfo", [
    'layer_name', # name of the layer (to get feature embedding etc.)
    'neigh_sampler', # callable neigh_sampler constructor
    'num_samples',
    'output_dim' # the output (i.e., hidden) dimension
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
        self.placeholders = {}
        
        self.layers = []
        self.activations = []
        
        self.inputs = None
        self.outputs = None
        
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None


class SupervisedGraphsage(Model):
    def __init__(self, num_classes,
            placeholders, features, adj, degrees,
            layer_infos, concat=True, aggregator_type="mean", 
            model_size="small", sigmoid_loss=False, identity_dim=0, **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''
        
        super(SupervisedGraphsage, self).__init__(**kwargs)
        
        self.aggregator = aggs[aggregator_type]
        self.inputs1    = placeholders["batch"]
        self.model_size = model_size
        self.adj_info   = adj
        
        self.embeds = None
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            
            self.features = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
        
        self.degrees      = degrees
        self.concat       = concat
        self.batch_size   = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos  = layer_infos
        
        dims = [(0 if features is None else features.shape[1]) + identity_dim]
        dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        
        # Optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        
        # Sample
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        
        # Aggregate
        self.outputs1, self.aggregators = self.aggregate(
            samples=samples1, 
            input_features=[self.features],
            dims=dims,
            num_samples=[layer_info.num_samples for layer_info in self.layer_infos],
            support_sizes=support_sizes1,
            concat=self.concat,
            model_size=self.model_size
        )
        
        # Normalize
        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)
        
        # Predict
        dim_mult = 2 if self.concat else 1
        self.node_pred = Dense(
            dim_mult * dims[-1],
            num_classes,
            dropout=self.placeholders['dropout'],
            act=lambda x : x
        )
        self.node_preds = self.node_pred(self.outputs1)
        
        # --
        # Define loss
        
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
        # classification loss
        if sigmoid_loss:
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        else:
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
            
        tf.summary.scalar('loss', self.loss)
        
        # --
        # Gradients
        
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        
        # --
        # Predictions
        
        if sigmoid_loss:
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
            node = layer_infos[t].neigh_sampler((samples[k], layer_infos[t].num_samples))
            samples.append(tf.reshape(node, [support_size * self.batch_size,]))
            support_sizes.append(support_size)
        
        return samples, support_sizes
    
    def aggregate(self, samples, input_features, dims, num_samples, support_sizes, batch_size=None,
            name=None, concat=False, model_size="small"):
        """ 
        
        At each layer, aggregate hidden representations of neighbors to compute the hidden representations  at next layer.
        
        Args:
            samples: a list of samples of variable hops away for convolving at each layer of the
                network. Length is the number of layers + 1. Each is a vector of node indices.
            input_features: the input features for each sample of various hops away.
            dims: a list of dimensions of the hidden representations from the input layer to the
                final layer. Length is the number of layers + 1.
            num_samples: list of number of samples for each layer.
            support_sizes: the number of nodes to gather information from for each layer.
            batch_size: the number of inputs (different for batch inputs and negative samples).
        
        Returns:
            The hidden representation at the final layer for all nodes in batch
        """
        
        batch_size = batch_size if batch_size else self.batch_size
            
        # length: number of layers + 1
        hidden = [tf.nn.embedding_lookup(input_features, node_samples) for node_samples in samples]
        
        aggregators = []
        for layer in range(len(num_samples)):
            dim_mult = 2 if concat and (layer != 0) else 1
            # aggregator at current layer
            if layer == len(num_samples) - 1:
                aggregator = self.aggregator(dim_mult*dims[layer], dims[layer+1], act=lambda x : x,
                        dropout=self.placeholders['dropout'], 
                        name=name, concat=concat, model_size=model_size)
            else:
                aggregator = self.aggregator(dim_mult*dims[layer], dims[layer+1],
                        dropout=self.placeholders['dropout'], 
                        name=name, concat=concat, model_size=model_size)
            aggregators.append(aggregator)
            
            # hidden representation at current layer for all support nodes that are various hops away
            
            next_hidden = []
            # as layer increases, the number of support nodes needed decreases
            for hop in range(len(num_samples) - layer):
                dim_mult = 2 if concat and (layer != 0) else 1
                neigh_dims = [batch_size * support_sizes[hop], 
                              num_samples[len(num_samples) - hop - 1], 
                              dim_mult*dims[layer]]
                h = aggregator((hidden[hop], tf.reshape(hidden[hop + 1], neigh_dims)))
                next_hidden.append(h)
            
            hidden = next_hidden
        
        return hidden[0], aggregators