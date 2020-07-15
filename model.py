# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


class ENMF(nn.Module):
    '''
    TOIS2020: Efficient Neural Matrix Factorization without Sampling for Recommendation
    original: https://github.com/chenchongthu/ENMF/
    '''
    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.user_embs = nn.Embedding(opt.user_num, opt.id_emb_size)
        self.item_embs = nn.Embedding(opt.item_num+1, opt.id_emb_size)
        self.h = nn.Parameter(torch.randn(opt.id_emb_size, 1))

        self.dropout = nn.Dropout(opt.drop_out)
        self.reset_para()

    def reset_para(self):
        nn.init.xavier_normal_(self.user_embs.weight)
        nn.init.xavier_normal_(self.item_embs.weight)
        nn.init.constant_(self.h, 0.01)

    def forward(self, uids, pos_iids):
        '''
        uids: B
        u_iids: B * L
        '''
        u_emb = self.dropout(self.user_embs(uids))
        pos_embs = self.item_embs(pos_iids)

        # torch.einsum("ab,abc->abc")
        # B * L * D
        mask = (~(pos_iids.eq(self.opt.item_num))).float()
        pos_embs = pos_embs * mask.unsqueeze(2)

        # torch.einsum("ac,abc->abc")
        # B * L * D
        pq = u_emb.unsqueeze(1) * pos_embs
        # torch.einsum("ajk,kl->ajl")
        # B * L
        hpq = pq.matmul(self.h).squeeze(2)

        # loss
        pos_data_loss = torch.sum((1 - self.opt.neg_weight) * hpq.square() - 2.0 * hpq)

        # torch.einsum("ab,ac->abc")
        part_1 = self.item_embs.weight.unsqueeze(2).bmm(self.item_embs.weight.unsqueeze(1))
        part_2 = u_emb.unsqueeze(2).bmm(u_emb.unsqueeze(1))

        # D * D
        part_1 = part_1.sum(0)
        part_2 = part_2.sum(0)
        part_3 = self.h.mm(self.h.t())
        all_data_loss = torch.sum(part_1 * part_2 * part_3)

        loss = self.opt.neg_weight * all_data_loss + pos_data_loss
        return loss

    def rank(self, uid):
        '''
        uid: Batch_size
        '''
        uid_embs = self.user_embs(uid)
        user_all_items = uid_embs.unsqueeze(1) * self.item_embs.weight
        items_score = user_all_items.matmul(self.h).squeeze(2)
        return items_score
