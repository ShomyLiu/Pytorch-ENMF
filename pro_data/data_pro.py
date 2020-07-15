# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.sparse
import os
from collections import defaultdict


def load_data(csv_file):
    tp = pd.read_csv(csv_file, sep='\t')
    return tp


def get_count(tp):
    user_groupbyid = tp[['uid']].groupby('uid', as_index=False)
    user_num = user_groupbyid.size().shape[0]
    item_groupbyid = tp[['sid']].groupby('sid', as_index=False)
    item_num = item_groupbyid.size().shape[0]
    return user_num, item_num


if __name__ == '__main__':
    np.random.seed(2019)
    data_root = './raw_data/ml-1m'

    tp_train = load_data(os.path.join(data_root, 'ml.train.txt'))
    tp_test = load_data(os.path.join(data_root, 'ml.test.txt'))
    save_folder = '../data/ml-1m'

    tp_all = tp_train.append(tp_test)
    user_num, item_num = get_count(tp_all)
    print(f"user number: {user_num}, item_number: {item_num}")

    u_train = np.array(tp_train['uid'], dtype=np.int32)
    i_train = np.array(tp_train['sid'], dtype=np.int32)
    u_test = np.array(tp_test['uid'], dtype=np.int32)
    i_test = np.array(tp_test['sid'], dtype=np.int32)

    count = np.ones(len(u_train))
    train_matrix = scipy.sparse.csr_matrix((count, (u_train, i_train)), dtype=np.int16, shape=(user_num, item_num))
    count = np.ones(len(u_test))
    test_matrix = scipy.sparse.csr_matrix((count, (u_test, i_test)), dtype=np.int16, shape=(user_num, item_num))

    u_iid_dict = defaultdict(list)
    for i in range(len(u_train)):
        u_iid_dict[u_train[i]].append(i_train[i])

    u_max_i = max([len(i) for i in u_iid_dict.values()])
    for i, id_list in u_iid_dict.items():
        if len(id_list) < u_max_i:
            u_iid_dict[i] += [item_num]*(u_max_i - len(id_list))

    print(f"the max number of user: {u_max_i}")

    u_id_train = np.array(list(set(u_train)), dtype=np.int32)
    u_id_test = np.array(list(set(u_test)), dtype=np.int32)
    u_iid_list = []
    for i in range(len(u_id_train)):
        u_iid_list.append(u_iid_dict[i])

    print(f'saving into {save_folder}...')
    np.save(f"{save_folder}/train_matrix.npy", train_matrix)
    np.save(f"{save_folder}/test_matrix.npy", test_matrix)
    np.save(f"{save_folder}/u_train.npy", u_id_train)
    np.save(f"{save_folder}/u_test.npy", u_id_test)
    np.save(f"{save_folder}/u_iid_list.npy", u_iid_list)
    print("finised.")
