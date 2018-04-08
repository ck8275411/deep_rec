#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
import math
__mtime__ = '2018/3/13'


def full_connect(inputs, weights_shape, biases_shape, is_train=True):
    weights = tf.get_variable("weights",
                              weights_shape,
                              initializer=tf.random_normal_initializer(stddev=(0.1 / math.sqrt(float(10)))),
                              trainable=is_train)
    biases = tf.get_variable("biases",
                             biases_shape,
                             initializer=tf.random_normal_initializer(stddev=0.1),
                             trainable=is_train)
    layer = tf.matmul(inputs, weights) + biases

    return layer


def full_connect_activation(inputs, weights_shape, biases_shape, activation='relu'):
    if activation == 'relu':
        layer = tf.nn.relu(full_connect(inputs, weights_shape, biases_shape))
    elif activation == 'tanh':
        layer = tf.nn.tanh(full_connect(inputs, weights_shape, biases_shape))
    return layer


def get_nn_layer(nn_inputs, input_size, nn_layer_shape, ctivation='relu'):
        with tf.variable_scope("nn_layer_input"):
            layer_output = full_connect_activation(nn_inputs,
                                             [input_size, nn_layer_shape[0]],
                                             [nn_layer_shape[0]])
        for i in range(len(nn_layer_shape) - 1):
            with tf.variable_scope("layer{}".format(i)):
                layer_output = full_connect_activation(layer_output,
                                                 [nn_layer_shape[i], nn_layer_shape[i + 1]],
                                                 [nn_layer_shape[i + 1]])
        with tf.variable_scope("output"):
            layer_output = full_connect(layer_output,
                                        [nn_layer_shape[-1], 1],
                                        [1])
        return layer_output


def get_lr_layer(sparse_features,
                 lr_weights_shape,
                 lr_biases_shape):
        with tf.variable_scope("lr_layer"):
            lr_weights = tf.get_variable("weights",
                                         lr_weights_shape,
                                         initializer=tf.random_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases",
                                     lr_biases_shape,
                                     initializer=tf.random_normal_initializer(stddev=0.1))
        return tf.sparse_tensor_dense_matmul(sparse_features, lr_weights) + biases


def get_fm_layer(sparse_features,
                 sparse_features_square,
                 fm_weights_shape,
                 embedding_dim):
    with tf.variable_scope("fm_layer", reuse=tf.AUTO_REUSE):
        fm_layer_weights = tf.get_variable("weights",
                                           fm_weights_shape,
                                           initializer=tf.random_normal_initializer(
                                               stddev=(0.1 / math.sqrt(float(embedding_dim))))
                                           )
    return 0.5 * tf.reduce_sum(
        tf.pow(tf.sparse_tensor_dense_matmul(sparse_features, fm_layer_weights), 2) -
        tf.sparse_tensor_dense_matmul(sparse_features_square, tf.pow(fm_layer_weights, 2)), 1,
        keepdims=True)
