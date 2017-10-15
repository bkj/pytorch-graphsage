#!/usr/bin/env python

"""
    aggregators.py
"""

import tensorflow as tf

from .layers import Layer, Dense
from .inits import glorot, zeros

class MeanAggregator(Layer):
    """ Aggregates via mean followed by matmul and non-linearity. """
    def __init__(self, input_dim, output_dim, neib_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, 
            name=None, concat=False, **kwargs):
        
        super(MeanAggregator, self).__init__(**kwargs)
        
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        if neib_input_dim is None:
            neib_input_dim = input_dim
            
        if name is not None:
            name = '/' + name
        else:
            name = ''
            
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neib_weights'] = glorot([neib_input_dim, output_dim], name='neib_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim], name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')
        
    def _call(self, inputs):
        self_vecs, neib_vecs = inputs
        
        neib_vecs  = tf.nn.dropout(neib_vecs, 1-self.dropout)
        self_vecs  = tf.nn.dropout(self_vecs, 1-self.dropout)
        neib_means = tf.reduce_mean(neib_vecs, axis=1)
       
        # [nodes] x [out_dim]
        from_neibs = tf.matmul(neib_means, self.vars['neib_weights'])
        
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
         
        if not self.concat:
            output = tf.add_n([from_self, from_neibs])
        else:
            output = tf.concat([from_self, from_neibs], axis=1)
            
        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class GCNAggregator(Layer):
    """
        Aggregates via mean followed by matmul and non-linearity.
        Same matmul parameters are used self vector and neibbor vectors.
    """
    def __init__(self, input_dim, output_dim, neib_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        
        super(GCNAggregator, self).__init__(**kwargs)
        
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        self.input_dim  = input_dim
        self.output_dim = output_dim
        
        if neib_input_dim is None:
            neib_input_dim = input_dim
            
        if name is not None:
            name = '/' + name
        else:
            name = ''
            
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['weights'] = glorot([neib_input_dim, output_dim],name='neib_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')
                
        
    def _call(self, inputs):
        self_vecs, neib_vecs = inputs
        
        neib_vecs = tf.nn.dropout(neib_vecs, 1-self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, 1-self.dropout)
        means = tf.reduce_mean(tf.concat([neib_vecs, tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)
       
        # [nodes] x [out_dim]
        output = tf.matmul(means, self.vars['weights'])
        
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class MaxPoolingAggregator(Layer):
    """ Aggregates via max-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neib_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(MaxPoolingAggregator, self).__init__(**kwargs)
        
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        
        if neib_input_dim is None:
            neib_input_dim = input_dim
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neib_input_dim = neib_input_dim
        
        if name is not None:
            name = '/' + name
        else:
            name = ''
            
        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024
            
        self.mlp_layers = [
            Dense(
                input_dim=neib_input_dim,
                output_dim=hidden_dim,
                act=tf.nn.relu,
                dropout=dropout,
                sparse_inputs=False,
            )
        ]

        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neib_weights'] = glorot([hidden_dim, output_dim], name='neib_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim], name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')
                
        
    def _call(self, inputs):
        self_vecs, neib_h = inputs
        
        dims = tf.shape(neib_h)
        batch_size, num_neighbors = dims[:2]
        
        # [nodes * sampled neibbors] x [hidden_dim]
        h_reshaped = tf.reshape(neib_h, (batch_size * num_neighbors, self.neib_input_dim))
        for mlp_layer in self.mlp_layers:
            h_reshaped = mlp_layer(h_reshaped)
        
        neib_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neib_h = tf.reduce_max(neib_h, axis=1)
        
        from_neibs = tf.matmul(neib_h, self.vars['neib_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neibs])
        else:
            output = tf.concat([from_self, from_neibs], axis=1)
            
        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class MeanPoolingAggregator(Layer):
    """ Aggregates via mean-pooling over MLP functions.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neib_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        
        super(MeanPoolingAggregator, self).__init__(**kwargs)
        
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        
        if neib_input_dim is None:
            neib_input_dim = input_dim
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neib_input_dim = neib_input_dim
        
        if name is not None:
            name = '/' + name
        else:
            name = ''
            
        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024
            
        self.mlp_layers = [
            Dense(
                input_dim=neib_input_dim,
                output_dim=hidden_dim,
                act=tf.nn.relu,
                dropout=dropout,
                sparse_inputs=False,
            )
        ]
        
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neib_weights'] = glorot([hidden_dim, output_dim], name='neib_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim], name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')
                
        
    def _call(self, inputs):
        self_vecs, neib_h = inputs
        
        dims = tf.shape(neib_h)
        batch_size, num_neighbors= dims[:2]
        
        # [nodes * sampled neibbors] x [hidden_dim]
        h_reshaped = tf.reshape(neib_h, (batch_size * num_neighbors, self.neib_input_dim))
        
        for mlp_layer in self.mlp_layers:
            h_reshaped = mlp_layer(h_reshaped)
        
        neib_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        neib_h = tf.reduce_mean(neib_h, axis=1)
        
        from_neibs = tf.matmul(neib_h, self.vars['neib_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neibs])
        else:
            output = tf.concat([from_self, from_neibs], axis=1)
        
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)


class TwoMaxLayerPoolingAggregator(Layer):
    """ Aggregates via pooling over two MLP functions. """
    def __init__(self, input_dim, output_dim, model_size="small", neib_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None, concat=False, **kwargs):
        super(TwoMaxLayerPoolingAggregator, self).__init__(**kwargs)
        
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        
        if neib_input_dim is None:
            neib_input_dim = input_dim
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neib_input_dim = neib_input_dim
        
        if name is not None:
            name = '/' + name
        else:
            name = ''
            
        if model_size == "small":
            hidden_dim_1 = self.hidden_dim_1 = 512
            hidden_dim_2 = self.hidden_dim_2 = 256
        elif model_size == "big":
            hidden_dim_1 = self.hidden_dim_1 = 1024
            hidden_dim_2 = self.hidden_dim_2 = 512
            
        self.mlp_layers = [
            Dense(
                input_dim=neib_input_dim,
                output_dim=hidden_dim_1,
                act=tf.nn.relu,
                dropout=dropout,
                sparse_inputs=False,
            ),
            Dense(
                input_dim=hidden_dim_1,
                output_dim=hidden_dim_2,
                act=tf.nn.relu,
                dropout=dropout,
                sparse_inputs=False,
            ),
        ]
        
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neib_weights'] = glorot([hidden_dim_2, output_dim], name='neib_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim], name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')
                
    def _call(self, inputs):
        self_vecs, neib_h = inputs
        
        dims = tf.shape(neib_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        # [nodes * sampled neibbors] x [hidden_dim]
        h_reshaped = tf.reshape(neib_h, (batch_size * num_neighbors, self.neib_input_dim))
        
        for mlp_layer in self.mlp_layers:
            h_reshaped = mlp_layer(h_reshaped)
        
        neib_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim_2))
        neib_h = tf.reduce_max(neib_h, axis=1)
        
        from_neibs = tf.matmul(neib_h, self.vars['neib_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neibs])
        else:
            output = tf.concat([from_self, from_neibs], axis=1)
            
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

class SeqAggregator(Layer):
    """ Aggregates via a standard LSTM.
    """
    def __init__(self, input_dim, output_dim, model_size="small", neib_input_dim=None,
            dropout=0., bias=False, act=tf.nn.relu, name=None,  concat=False, **kwargs):
        super(SeqAggregator, self).__init__(**kwargs)
        
        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat
        
        if neib_input_dim is None:
            neib_input_dim = input_dim
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neib_input_dim = neib_input_dim
        
        if name is not None:
            name = '/' + name
        else:
            name = ''
            
        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256
            
        with tf.variable_scope(self.name + name + '_vars'):
            self.vars['neib_weights'] = glorot([hidden_dim, output_dim], name='neib_weights')
            self.vars['self_weights'] = glorot([input_dim, output_dim],name='self_weights')
            if self.bias:
                self.vars['bias'] = zeros([self.output_dim], name='bias')
                
        self.cell = tf.contrib.rnn.BasicLSTMCell(self.hidden_dim)
        
    def _call(self, inputs):
        self_vecs, neib_vecs = inputs
        
        dims = tf.shape(neib_vecs)
        batch_size = dims[0]
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        used = tf.sign(tf.reduce_max(tf.abs(neib_vecs), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.maximum(length, tf.constant(1.))
        length = tf.cast(length, tf.int32)

        with tf.variable_scope(self.name) as scope:
            try:
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neib_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
            except ValueError:
                scope.reuse_variables()
                rnn_outputs, rnn_states = tf.nn.dynamic_rnn(
                        self.cell, neib_vecs,
                        initial_state=initial_state, dtype=tf.float32, time_major=False,
                        sequence_length=length)
        
        batch_size = tf.shape(rnn_outputs)[0]
        max_len = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        neib_h = tf.gather(flat, index)
        
        from_neibs = tf.matmul(neib_h, self.vars['neib_weights'])
        from_self = tf.matmul(self_vecs, self.vars["self_weights"])
         
        output = tf.add_n([from_self, from_neibs])
        
        if not self.concat:
            output = tf.add_n([from_self, from_neibs])
        else:
            output = tf.concat([from_self, from_neibs], axis=1)
            
        # bias
        if self.bias:
            output += self.vars['bias']
       
        return self.act(output)

