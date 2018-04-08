#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import input_reader
from utils import layers
import tensorflow as tf
import math

__mtime__ = '2018/3/13'


class DeepFM:
    def __init__(self,
                 n_epoch,
                 batch_size,
                 embedding_dim,
                 nn_layer_shape,
                 feature_size,
                 num_parallel=10,
                 activation='relu',
                 learning_rate=0.001,
                 optimizer='adam',
                 steps_to_logout=1000):
        self.n_epoch = n_epoch
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.nn_layer_shape = nn_layer_shape
        self.num_parallel = num_parallel
        self.activation = activation
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.steps_to_logout = steps_to_logout
        self.feature_size = feature_size
        self.lr_weights_shape = [self.feature_size, 1]
        self.lr_biases_shape = [1]
        self.fm_weights_shape = [self.feature_size, self.embedding_dim]

    def get_nn_inputs(self, sparse_features):
        dense_features  = tf.sparse_tensor_to_dense(sparse_features, default_value=0.0)
        dense_features = tf.reshape(dense_features, shape=[-1, self.feature_size, 1])
        with tf.variable_scope("fm_layer", reuse=tf.AUTO_REUSE):
            fm_layer_embeddings = tf.get_variable("weights",
                                                  self.fm_weights_shape)

        nn_inputs = tf.reshape(tf.multiply(fm_layer_embeddings, dense_features),
                               [-1, self.feature_size * self.embedding_dim])
        return nn_inputs, self.feature_size * self.embedding_dim

    def train_op(self, batch_parsed_features):
        batch_labels = tf.reshape(tf.to_float(batch_parsed_features["label"]), [-1])
        sparse_ids = batch_parsed_features["indices"]
        sparse_values = batch_parsed_features["values"]

        sparse_features = tf.sparse_merge(sparse_ids, sparse_values, vocab_size=self.feature_size)
        sparse_values_square = tf.SparseTensor(sparse_values.indices, tf.pow(sparse_values.values, 2),
                                               sparse_values.dense_shape)
        sparse_features_square = tf.sparse_merge(sparse_ids, sparse_values_square, vocab_size=self.feature_size)

        # 获取LR输出
        lr_layer = layers.get_lr_layer(sparse_features,
                                       self.lr_weights_shape,
                                       self.lr_biases_shape)
        # 获取FM向量输出
        fm_layer = layers.get_fm_layer(sparse_features,
                                       sparse_features_square,
                                       self.fm_weights_shape,
                                       self.embedding_dim)
        # 获取对稀疏输入处理，获取dnn的输入
        nn_inputs, input_size = self.get_nn_inputs(sparse_features)
        nn_layer = layers.get_nn_layer(nn_inputs, input_size, self.nn_layer_shape)
        deepfm_output = tf.reshape(lr_layer + fm_layer + nn_layer, [-1])
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels, logits=deepfm_output)
        loss = tf.reduce_mean(cross_entropy)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=global_step)
        #train auc
        field_deepfm_output_sigmoid = tf.sigmoid(deepfm_output)
        auc = tf.metrics.auc(batch_labels, field_deepfm_output_sigmoid)
        tf.summary.scalar('auc1', auc[0])
        tf.summary.scalar('auc2', auc[1])
        tf.summary.scalar('loss', loss)
        return train_step, loss, global_step, auc

    def fit(self, train_path):
        next_element = input_reader.get_input(train_path,
                                              self.num_parallel,
                                              self.batch_size,
                                              self.n_epoch)
        train_logit = self.train_op(next_element)
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer())
        with tf.Session() as sess:
            sess.run(init_op)
            #合并到Summary中
            merged = tf.summary.merge_all()
            #选定可视化存储目录
            writer = tf.summary.FileWriter('./summary/deepfm', graph=tf.get_default_graph())
            try:
                while True:
                    train_step, loss, global_step, auc = sess.run(train_logit)
                    if global_step % self.steps_to_logout == 0:
                        result = sess.run(merged)
                        writer.add_summary(result, global_step)
            except tf.errors.OutOfRangeError:
                print("End of dataset")
        writer.close()
