import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset


class SAINTDataset(Dataset):
    def __init__(self, group, n_question, max_seq):
        super(SAINTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_question = n_question
        self.samples = group

        self.user_ids = [x for x in group.index]

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, s_, qtype_, qd_, qms_, qattempt_, qa_ = self.samples[user_id]
        seq_len = len(q_)

        # q = np.array([self.n_question] * self.max_seq, dtype=int)
        q = np.zeros(self.max_seq, dtype=int)
        qa, qtype = q.copy(), q.copy()

        if seq_len >= self.max_seq:
            q[:] = q_[-self.max_seq:]
            qtype[:] = qtype_[-self.max_seq:]
            qa[:] = qa_[-self.max_seq:]
        else:
            q[-seq_len:] = q_
            qtype[-seq_len:] = qtype_
            qa[-seq_len:] = qa_

        target_qid = q[1:]
        target_qtype = qtype[1:]
        label = qa[1:]

        # x = np.zeros(self.max_seq-1, dtype=int)
        # x = q[:-1].copy()
        # x += (qa[:-1] == 1) * self.n_question
        response = qa[:-1].copy()
        x = response

        # return x, cat_f_list, con_f_list, label
        return target_qid, target_qtype, x, label
