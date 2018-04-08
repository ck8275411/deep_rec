#!/usr/bin/env python
# -*- coding: utf-8 -*-
import tensorflow as tf
from os import path, listdir

__mtime__ = '2018/3/13'


def _default_parser(example_proto):
    features = {
        "label": tf.FixedLenFeature([], tf.int64),
        "indices": tf.VarLenFeature(tf.int64),
        "values": tf.VarLenFeature(tf.float32),
        "maxIndice": tf.FixedLenFeature([], tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return parsed_features


def get_files(my_path):
    return [path.join(my_path, f) for f in listdir(my_path) if path.isfile(path.join(my_path, f))]


def get_input(train_path,
              num_parallel=10,
              batch_size=1024,
              n_epoch=1,
              buffer_size=1024 * 1024,
              _parse_function=_default_parser):
    train_files = get_files(train_path)
    data = tf.data.TFRecordDataset(train_files, buffer_size=buffer_size) \
        .map(_parse_function, num_parallel) \
        .shuffle(buffer_size=10000) \
        .batch(batch_size) \
        .repeat(n_epoch)
    iterator = data.make_one_shot_iterator()
    next_element = iterator.get_next()
    return next_element
