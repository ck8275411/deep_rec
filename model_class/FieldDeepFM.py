#!/usr/bin/env python
# -*- coding: utf-8 -*-

from utils import input_reader
from utils import layers
import tensorflow as tf
import math

__mtime__ = '2018/3/13'


class FieldDeepFM:
    def __init__(self,
                 n_epoch,
                 batch_size,
                 embedding_dim,
                 nn_layer_shape,
                 feature_size,
                 feature_field_file,
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
        self.field_embedding_masks, self.fields_num = self.get_feature_field(feature_field_file)

        self.lr_weights_shape = [self.feature_size, 1]
        self.lr_biases_shape = [1]
        self.fm_weights_shape = [self.feature_size, self.embedding_dim]

    def get_feature_field(self, feature_field_file):
        feature_fields = []
        field_names = {}
        _fields = []

        for line in open(feature_field_file, "r"):
            tokens = line.strip('\r').strip('\n').split(' ')
            #特征id从0开始编码，所以feature_fields数组中的下表就可以表示特征id，值表示特征所属的field
            feature_fields.append(tokens[0])
            #保存去重后的field id
            field_names[tokens[0]] = 1
        for field_name in field_names.keys():
            #遍历所有field
            field = []
            for j in range(self.feature_size):
                '''
                遍历所有特征id
                如果j大于len(feature_fields)，则直接置0
                如果第j个特征的field等于当前field，则置1.0
                如果第j个特征的field不等于当前field，则置0.0
                '''
                if j >= len(feature_fields):
                    field.append([0.0])
                elif feature_fields[j] == field_name:
                    field.append([1.0])
                else:
                    field.append([0.0])
            _fields.append(field)
        return _fields, len(field_names)

    def get_field_embeddings(self, sparse_features):
        input_layers = []
        k = 0
        with tf.variable_scope("fm_layer", reuse=tf.AUTO_REUSE):
            fm_layer_embeddings = tf.get_variable("weights",
                                                  self.fm_weights_shape)
            for i in range(self.fields_num):
                fm_field_embeddings = tf.multiply(fm_layer_embeddings, self.field_embedding_masks[i])
                # 计算每个field的feature_cnt
                sparse_x_feature_cnt = tf.maximum(tf.sparse_tensor_dense_matmul(sparse_features, self.field_embedding_masks[i]), 1.0)
                sparse_x_field_embedded = tf.sparse_tensor_dense_matmul(sparse_features, fm_field_embeddings)
                # 计算embedding计算的均值
                sparse_x_field_embedded = tf.divide(sparse_x_field_embedded, sparse_x_feature_cnt)
                input_layers.append(sparse_x_field_embedded)
                k += self.embedding_dim
            nn_inputs = tf.concat(input_layers, 1)
        return nn_inputs, k

    def train_op(self, batch_parsed_features):
        global_steps = tf.Variable(0, trainable=False)

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
        # 获取Field Embedding输出
        field_embeddings, input_size = self.get_field_embeddings(sparse_features)
        nn_layer = layers.get_nn_layer(field_embeddings, input_size, self.nn_layer_shape)
        field_deepfm_output = tf.reshape(lr_layer + fm_layer + nn_layer, [-1])
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=batch_labels,
                                                                logits=field_deepfm_output)
        #train loss
        loss = tf.reduce_mean(cross_entropy)
        train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(loss, global_step=global_steps)

        #train auc
        field_deepfm_output_sigmoid = tf.sigmoid(field_deepfm_output)
        auc = tf.metrics.auc(batch_labels, field_deepfm_output_sigmoid)
        tf.summary.scalar('auc1', auc[0])
        tf.summary.scalar('auc2', auc[1])
        tf.summary.scalar('loss', loss)
        return train_step, loss, global_steps, auc

    def fit(self, train_path):
        next_element = input_reader.get_input(train_path,
                                              self.num_parallel,
                                              self.batch_size,
                                              self.n_epoch)
        train_logit = self.train_op(next_element)

        tf.get_variable_scope().reuse_variables()
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        with tf.Session() as sess:
            print("Start of trainning")
            sess.run(init_op)
            #合并到Summary中
            merged = tf.summary.merge_all()
            #选定可视化存储目录
            writer = tf.summary.FileWriter('./summary/field_deepfm', graph=tf.get_default_graph())
            try:
                while True:
                    train_step, loss, global_step, auc = sess.run(train_logit)
                    if global_step % self.steps_to_logout == 0:
                        result = sess.run(merged)
                        writer.add_summary(result, global_step)
            except tf.errors.OutOfRangeError:
                print("End of dataset")
            writer.close()
