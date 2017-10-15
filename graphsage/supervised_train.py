#!/usr/bin/env python

"""
    supervised_train.py
"""

from __future__ import division
from __future__ import print_function

import os
import time
import sklearn
import numpy as np
import tensorflow as tf
from sklearn import metrics

from graphsage.data_loader import NodeDataLoader
from graphsage.supervised_models import SupervisedGraphsage, SAGEInfo

# --
# Helpers

def set_seeds(seed=0):
    np.random.seed(seed)
    tf.set_random_seed(seed)


class UniformNeighborSampler(object):
    def __init__(self, adj, **kwargs):
        self.adj = adj
        
    def __call__(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj, ids)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        return tf.slice(adj_lists, [0,0], [-1, num_samples])


def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred = y_pred.round().astype(int)
    
    return {
        "micro" : metrics.f1_score(y_true, y_pred, average="micro"),
        "macro" : metrics.f1_score(y_true, y_pred, average="macro"),
    }


def evaluate(sess, model, data_loader, placeholders, mode):
    preds, labels = [], []
    for eval_batch in data_loader.iterate(mode=mode, shuffle=False):
        preds.append(sess.run(model.preds, feed_dict={
            placeholders['batch'] : eval_batch['batch'],
            placeholders['batch_size'] : eval_batch['batch_size'],
            placeholders['labels'] : eval_batch['labels'],
        }))
        labels.append(eval_batch['labels'])
    
    return calc_f1(np.vstack(labels), np.vstack(preds))

# --
# Params

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

set_seeds(123)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")

#core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')  
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string('model_size', "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')

# left to default values in main experiments 
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

all_models = [
    'graphsage_mean',
    'gcn',
    'graphsage_seq',
    'graphsage_maxpool',
    'graphsage_meanpool',
]


def select_model(model, sampler):
    if model == 'graphsage_mean':
        return {
            "layer_infos" : filter(None, [
                SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2) if FLAGS.samples_2 != 0 else None,
                SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2) if FLAGS.samples_3 != 0 else None,
            ]),
        }
        
    elif model == 'gcn':
        return {
            "aggregator_type" : "gcn",
            "concat" : False,
            "layer_infos" : [
                SAGEInfo("node", sampler, FLAGS.samples_1, 2 * FLAGS.dim_1),
                SAGEInfo("node", sampler, FLAGS.samples_2, 2 * FLAGS.dim_2)
            ],
        }
        
    elif model == 'graphsage_seq':
        return {
            "aggregator_type" : "seq",
            "layer_infos" : [
                SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)
            ],
        }
        
    elif model == 'graphsage_maxpool':
        return {
            "aggregator_type" : "pool",
            "layer_infos" : [
                SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)
            ],
        }
        
    elif model == 'graphsage_meanpool':
        return {
            "aggregator_type" : "meanpool",
            "layer_infos" : [
                SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)
            ],
        }


if __name__ == "__main__":
    
    set_seeds(456)
    
    assert FLAGS.model in all_models, 'Error: model name unrecognized.'
    
    # --
    # IO
    
    cache_path = '%s-%d-%d-iterator-cache.h5' % (FLAGS.train_prefix, FLAGS.batch_size, FLAGS.max_degree)
    if not os.path.exists(cache_path):
        data_loader = NodeDataLoader(
            data_path=FLAGS.train_prefix,
            batch_size=FLAGS.batch_size,
            max_degree=FLAGS.max_degree,
        )
        
        data_loader.save(cache_path)
    else:
        data_loader = NodeDataLoader(cache_path=cache_path)
    
    # --
    # Define model + sampler
    
    adj_ = tf.Variable(tf.constant(data_loader.train_adj, dtype=tf.int32), trainable=False, name="adj_")
    
    placeholders = {
        'labels'     : tf.placeholder(tf.float32, shape=(None, data_loader.num_classes), name='labels'),
        'batch'      : tf.placeholder(tf.int32, shape=(None), name='batch'),
        'dropout'    : tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    
    params = {
        "num_classes"   : data_loader.num_classes,
        "placeholders"  : placeholders,
        "features"      : data_loader.features,
        "adj"           : adj_,
        "degrees"       : data_loader.degrees,
        "model_size"    : FLAGS.model_size,
        "sigmoid"       : FLAGS.sigmoid,
        "identity_dim"  : FLAGS.identity_dim,
        "learning_rate" : FLAGS.learning_rate,
        "weight_decay"  : FLAGS.weight_decay,
    }
    sampler = UniformNeighborSampler(adj_)
    params.update(select_model(FLAGS.model, sampler))
    model = SupervisedGraphsage(**params)
    
    # --
    # TF stuff
    
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())
    
    load_train_adj = tf.assign(adj_, data_loader.train_adj)
    load_val_adj = tf.assign(adj_, data_loader.val_adj)
    
    # --
    # Train 
    
    set_seeds(891)
    
    total_steps = 0
    for epoch in range(FLAGS.epochs): 
        for iter_, train_batch in enumerate(data_loader.iterate(mode='train', shuffle=True)):
            
            # Training step
            _, _, train_preds = sess.run([model.opt_op, model.loss, model.preds], feed_dict={
                placeholders['batch'] : train_batch['batch'],
                placeholders['batch_size'] : train_batch['batch_size'],
                placeholders['labels'] : train_batch['labels'],
                placeholders['dropout'] : FLAGS.dropout
            })
            
            # Validation performance
            if iter_ % FLAGS.validate_iter == 0:
                sess.run(load_val_adj.op)
                val_batch = data_loader.sample_eval_batch(size=FLAGS.validate_batch_size, mode='val')
                val_f1 = calc_f1(val_batch['labels'], sess.run(model.preds, feed_dict={
                    placeholders['batch'] : val_batch['batch'],
                    placeholders['batch_size'] : val_batch['batch_size'],
                    placeholders['labels'] : val_batch['labels'],
                }))
                sess.run(load_train_adj.op)
            
            # Logging
            if total_steps % FLAGS.print_every == 0:
                train_f1 = calc_f1(train_batch['labels'], train_preds)
                print({
                    "epoch" : epoch,
                    "iter" : iter_,
                    "train_f1" : train_f1,
                    "val_f1" : val_f1,
                })
            
            total_steps += 1
    
    # --
    # Eval
    
    sess.run(load_val_adj.op)
    print({"val_f1" : evaluate(sess, model, data_loader, placeholders, mode='test')})
    print({"test_f1" : evaluate(sess, model, data_loader, placeholders, mode='val')})
