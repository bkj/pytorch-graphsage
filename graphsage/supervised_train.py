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

from graphsage.utils import load_data
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.supervised_models import SupervisedGraphsage, SAGEInfo

# --
# Helpers

class UniformNeighborSampler(object):
    def __init__(self, adj_info, **kwargs):
        self.adj_info = adj_info
        
    def __call__(self, inputs):
        ids, num_samples = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        adj_lists = tf.transpose(tf.random_shuffle(tf.transpose(adj_lists)))
        return tf.slice(adj_lists, [0,0], [-1, num_samples])


# --
# Params

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

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
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 1, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")

all_models = [
    'graphsage_mean',
    'gcn',
    'graphsage_seq',
    'graphsage_maxpool',
    'graphsage_meanpool',
]

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def calc_f1(y_true, y_pred):
    if not FLAGS.sigmoid:
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def log_dir():
    log_dir = FLAGS.base_log_dir + "/sup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.4f}/".format(
        model=FLAGS.model,
        model_size=FLAGS.model_size,
        lr=FLAGS.learning_rate
    )
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    return log_dir


def train(train_data, test_data=None):
    
    total_steps   = 0
    
    G             = train_data[0]
    features      = train_data[1]
    id_map        = train_data[2]
    context_pairs = train_data[3]
    class_map     = train_data[4]
    
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
    else:
        num_classes = len(set(class_map.values()))
        
    # pad with dummy zero vector
    if features is not None:
        features = np.vstack([features, np.zeros((features.shape[1],))])
    
    placeholders = {
        'labels'     : tf.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch'      : tf.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout'    : tf.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.placeholder(tf.int32, name='batch_size'),
    }
    
    minibatch = NodeMinibatchIterator(
        G,
        id_map,
        placeholders,
        class_map,
        num_classes,
        batch_size=FLAGS.batch_size,
        max_degree=FLAGS.max_degree, 
        context_pairs=context_pairs
    )
    
    adj_info = tf.Variable(tf.constant(minibatch.adj, dtype=tf.int32), trainable=False, name="adj_info")
    
    sampler = UniformNeighborSampler(adj_info)
    
    if FLAGS.model == 'graphsage_mean':
        layer_infos = filter(None, [
            SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2) if FLAGS.samples_2 != 0 else None,
            SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2) if FLAGS.samples_3 != 0 else None,
        ])
        
        model = SupervisedGraphsage(
            num_classes,
            placeholders,
            features,
            adj_info,
            minibatch.deg,
            layer_infos=layer_infos,
            model_size=FLAGS.model_size,
            sigmoid_loss=FLAGS.sigmoid,
            identity_dim=FLAGS.identity_dim,
            logging=True,
        )
    
    elif FLAGS.model == 'gcn':
        layer_infos = [
            SAGEInfo("node", sampler, FLAGS.samples_1, 2 * FLAGS.dim_1),
            SAGEInfo("node", sampler, FLAGS.samples_2, 2 * FLAGS.dim_2)
        ]
        
        model = SupervisedGraphsage(
            num_classes,
            placeholders,
            features,
            adj_info,
            minibatch.deg,
            layer_infos=layer_infos,
            model_size=FLAGS.model_size,
            sigmoid_loss=FLAGS.sigmoid,
            identity_dim=FLAGS.identity_dim,
            logging=True,
            
            concat=False,
            aggregator_type="gcn",
        )

    elif FLAGS.model == 'graphsage_seq':
        layer_infos = [
            SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)
        ]
        
        model = SupervisedGraphsage(
            num_classes,
            placeholders,
            features,
            adj_info,
            minibatch.deg,
            layer_infos=layer_infos,
            model_size=FLAGS.model_size,
            sigmoid_loss=FLAGS.sigmoid,
            identity_dim=FLAGS.identity_dim,
            logging=True,
            
            aggregator_type="seq",
        )

    elif FLAGS.model == 'graphsage_maxpool':
        layer_infos = [
            SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)
        ]
        
        model = SupervisedGraphsage(
            num_classes,
            placeholders,
            features,
            adj_info,
            minibatch.deg,
            layer_infos=layer_infos,
            model_size=FLAGS.model_size,
            sigmoid_loss=FLAGS.sigmoid,
            identity_dim=FLAGS.identity_dim,
            logging=True,
            
            aggregator_type="pool",
        )

    elif FLAGS.model == 'graphsage_meanpool':
        layer_infos = [
            SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)
        ]
        
        model = SupervisedGraphsage(
            num_classes,
            placeholders,
            features,
            adj_info,
            minibatch.deg,
            layer_infos=layer_infos,
            model_size=FLAGS.model_size,
            sigmoid_loss=FLAGS.sigmoid,
            identity_dim=FLAGS.identity_dim,
            logging=True,
            
            aggregator_type="meanpool",
        )
    
    # Initialize session
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
     
    # Init variables
    sess.run(tf.global_variables_initializer())
    
    # Train model
    avg_time = 0.0
    
    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.val_adj)
    for epoch in range(FLAGS.epochs): 
        minibatch.shuffle()
        
        iter = 0
        print('Epoch: %04d' % (epoch + 1))
        while not minibatch.is_done:
            # Construct feed dictionary
            feed_dict, labels = minibatch.get_train_batch()
            feed_dict.update({
                placeholders['dropout']: FLAGS.dropout
            })
            
            t = time.time()
            
            # Training step
            outs = sess.run([merged, model.opt_op, model.loss, model.preds], feed_dict=feed_dict)
            
            if iter % FLAGS.validate_iter == 0:
                sess.run(val_adj_info.op)
                
                val_feed_dict, val_labels = minibatch.get_eval_batch(size=FLAGS.validate_batch_size, test=False)
                val_node_outs = sess.run([model.preds, model.loss], feed_dict=val_feed_dict)
                val_f1_mic, val_f1_mac = calc_f1(val_labels, val_node_outs[0])
                
                sess.run(train_adj_info.op)
            
            if total_steps % FLAGS.print_every == 0:
                train_f1_mic, train_f1_mac = calc_f1(labels, outs[-1])
                print(
                    "Iter:", '%04d' % iter, 
                    "train_f1_mic=", "{:.5f}".format(train_f1_mic),
                    "train_f1_mac=", "{:.5f}".format(train_f1_mac),
                    "val_f1_mic=", "{:.5f}".format(val_f1_mic),
                    "val_f1_mac=", "{:.5f}".format(val_f1_mac),
                )
            
            iter += 1
            total_steps += 1
    
    sess.run(val_adj_info.op)
    
    # Final validation performance
    val_feed_dict, val_labels = minibatch.get_eval_batch(size=None, test=False)
    val_node_outs = sess.run([model.preds, model.loss], feed_dict=val_feed_dict)
    val_f1_mic, val_f1_mac = calc_f1(val_labels, val_node_outs[0])
    print("\nval_f1_mic={:.5f} val_f1_mac={:.5f}".format(val_f1_mic, val_f1_mac))
    
    # Final test performance
    test_feed_dict, test_labels = minibatch.get_eval_batch(size=None, test=True)
    test_node_outs = sess.run([model.preds, model.loss], feed_dict=test_feed_dict)
    test_f1_mic, test_f1_mac = calc_f1(test_labels, test_node_outs[0])
    print("\ntest_f1_mic={:.5f} test_f1_mac={:.5f}".format(test_f1_mic, test_f1_mac))


if __name__ == '__main__':
    assert FLAGS.model in all_models, 'Error: model name unrecognized.'
    print("Loading training data..")
    train_data = load_data(FLAGS.train_prefix)
    train(train_data)
