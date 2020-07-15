# -*- coding: utf-8 -*-

import time
import random
import fire

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from data import RecData
from config import opt
from model import ENMF
from utils import evaluation


def now():
    return str(time.strftime('%Y-%m-%d %H:%M:%S'))


def collate_fn(batch):
    uids, u_iid_list = zip(*batch)
    return torch.LongTensor(uids).cuda(), torch.LongTensor(u_iid_list).cuda()


def train(**kwargs):

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    opt.parse(kwargs)
    torch.cuda.set_device(opt.gpu_id)

    model = ENMF(opt).cuda()

    train_data = RecData("Train")
    test_data = RecData("Test")

    train_dataloader = DataLoader(train_data, opt.batch_size, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_data, opt.batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = optim.Adagrad(model.parameters(), opt.lr)

    print("start traning..")

    for epoch in range(opt.epochs):
        total_loss = 0.0
        print(f"{now()} Epoch {epoch}:")
        model.train()
        for _, (uids, u_iid_list) in enumerate(train_dataloader):
            loss = model(uids, u_iid_list)
            optimizer.zero_grad()
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        mean_loss = total_loss / len(train_data)
        print(f"\tloss: {mean_loss}")
        test(model, test_dataloader)


def test(model, test_dataloader):
    model.eval()

    recall_list = []
    ndcg_list = []
    for _ in range(len(opt.top_k)):
        recall_list.append([])
        ndcg_list.append([])

    with torch.no_grad():
        for _, (uids, uiid_list) in enumerate(test_dataloader):
            scores = model.rank(uids).cpu().numpy()
            scores = np.delete(scores, -1, axis=1)
            uids = uids.cpu().numpy()
            idx = np.zeros_like(scores, dtype=bool)
            idx[opt.train_matrix[uids].nonzero()] = True
            scores[idx] = -np.inf

            recall, ndcg = evaluation(scores, opt.test_matrix[uids], opt.top_k)
            for i in range(len(opt.top_k)):
                recall_list[i].append(recall[i])
                ndcg_list[i].append(ndcg[i])

        recall_list = [np.mean(np.hstack(r)) for r in recall_list]
        ndcg_list = [np.mean(np.hstack(r)) for r in ndcg_list]

    for i, k in enumerate(opt.top_k):
        print(f"\tRecall@{k}: {recall_list[i]}, ndcg@{k}: {ndcg_list[i]}")

    model.train()


if __name__ == "__main__":
    fire.Fire()
