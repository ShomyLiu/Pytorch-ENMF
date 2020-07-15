# -*- coding: utf-8 -*-

import numpy as np


class DefaultConfig:

    model = 'ENMF'
    dataset = './data/ml-1m'

    # -------------base config-----------------------#
    use_gpu = True
    gpu_id = 1
    multi_gpu = False
    gpu_ids = []

    seed = 2019
    epochs = 200
    num_workers = 0

    weight_decay = 1e-3  # optimizer rameteri
    lr = 5e-2
    drop_out = 0.3

    id_emb_size = 64
    neg_weight = 0.1
    top_k = [50, 100, 200]

    user_num = 6040
    item_num = 3706

    batch_size = 128

    def parse(self, kwargs):
        '''
        user can update the default hyperparamter
        '''
        self.train_matrix = np.load(f"{self.dataset}/train_matrix.npy", allow_pickle=True).tolist()
        self.test_matrix = np.load(f"{self.dataset}/test_matrix.npy", allow_pickle=True).tolist()
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception('opt has No key: {}'.format(k))
            setattr(self, k, v)

        # print('*************************************************')
        # print('user config:')
        # for k, v in self.__class__.__dict__.items():
            # if not k.startswith('__') and k != 'user_list' and k != 'item_list':
        # print("{} => {}".format(k, getattr(self, k)))

        # print('*************************************************')


opt = DefaultConfig()
