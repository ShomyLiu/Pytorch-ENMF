# -*- coding: utf-8 -*-

import numpy as np


def evaluation(pre, test_batch, topK):
    '''
    predict: BS * ItemNum
    label: BS * ItemNum
    '''

    batch_users = pre.shape[0]
    recall = []
    true_bin = np.zeros_like(pre, dtype=bool)
    true_bin[test_batch.nonzero()] = True

    for kj in topK:
        idx_topk_part = np.argpartition(-pre, kj, 1)

        pre_bin = np.zeros_like(pre, dtype=bool)
        pre_bin[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]] = True

        tmp = (np.logical_and(true_bin, pre_bin).sum(axis=1)).astype(np.float32)
        recall.append(tmp / np.minimum(kj, true_bin.sum(axis=1)))

    ndcg = []

    for kj in topK:
        idx_topk_part = np.argpartition(-pre, kj, 1)

        topk_part = pre[np.arange(batch_users)[:, np.newaxis], idx_topk_part[:, :kj]]
        idx_part = np.argsort(-topk_part, axis=1)
        idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

        tp = np.log(2) / np.log(np.arange(2, kj + 2))

        DCG = (test_batch[np.arange(batch_users)[:, np.newaxis],
                          idx_topk].toarray() * tp).sum(axis=1)

        IDCG = np.array([(tp[:min(n, kj)]).sum()
                         for n in test_batch.getnnz(axis=1)])
        ndcg.append(DCG / IDCG)

    return recall, ndcg
