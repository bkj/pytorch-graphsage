from __future__ import division
from __future__ import print_function

import tensorflow as tf

from graphsage.inits import zeros

flags = tf.app.flags
FLAGS = flags.FLAGS

_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]

class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'model_size'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        
        self.name = name
        self.vars = {}
        
    def __call__(self, inputs):
        with tf.name_scope(self.name):
            return self._call(inputs)


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, dropout=0., act=tf.nn.relu, placeholders=None, sparse_inputs=False, **kwargs):
        
        super(Dense, self).__init__(**kwargs)
        
        self.input_dim   = input_dim
        self.output_dim  = output_dim
        self.act         = act
        
        # helper variable for sparse dropout
        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights'] = tf.get_variable(
                'weights',
                shape=(input_dim, output_dim),
                dtype=tf.float32, 
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=tf.contrib.layers.l2_regularizer(FLAGS.weight_decay)
            )
            
            self.vars['bias'] = zeros([output_dim], name='bias')
                
    def _call(self, x):
        output = tf.matmul(x, self.vars['weights'])
        output += self.vars['bias']
        return output
