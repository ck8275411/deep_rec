#!/usr/bin/env python
# -*- coding: utf-8 -*-

from model_class import FieldDeepFM
from model_class import DeepFM

__mtime__ = '2018/3/13'

if __name__ == '__main__':
    '''
    model = DeepFM.DeepFM(25,
                          1024,
                          8,
                          [128, 64, 8],
                          10560,
                          num_parallel=10,
                          activation='relu',
                          learning_rate=0.001,
                          optimizer='adam',
                          steps_to_logout=10)
    model.fit('./data')
    '''
    model = FieldDeepFM.FieldDeepFM(25,
                                    1024,
                                    8,
                                    [128, 64, 8],
                                    10560,
                                    './feature_field_data/20171115',
                                    num_parallel=10,
                                    activation='relu',
                                    learning_rate=0.001,
                                    optimizer='adam',
                                    steps_to_logout=10)

    model.fit('./data')


