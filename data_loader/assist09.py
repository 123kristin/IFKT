import numpy as np
import pandas as pd
import os

from torch.utils.data import Dataset

class Assist09Dataset(Dataset):
    def __init__(self, group, n_question, max_seq):
        super(Assist09Dataset, self).__init__()
        self.max_seq = max_seq
        self.n_question = n_question
        self.samples = group

        self.user_ids = [x for x in group.index] # [14, 21825, 51950, 52613, 53167, ...]

    def __len__(self):
        return len(self.user_ids) # 4151
    
    # 一个user的记录，包含这个user的q_idx, s_idx, q_type, q_diff, ms_first_response, attempt_count, correct
    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, s_, qtype_, qd_, qms_, qattempt_, qa_ = self.samples[user_id]
        seq_len = len(q_)

        q = np.zeros(self.max_seq, dtype=int)
        s, qa, qtype = q.copy(), q.copy(), q.copy() # qa指的是correct字段

        qd = np.zeros(self.max_seq, dtype=float)
        qms, qattempt = qd.copy(), qd.copy()


        # 将每个user的记录规整为max_seq长的序列
        ## 如果当前user的答题个数超过max_seq，取距离当前时刻最近的max_seq条记录
        if seq_len >= self.max_seq:
            q[:] = q_[-self.max_seq:]
            s[:] = s_[-self.max_seq:]
            qtype[:] = qtype_[-self.max_seq:]
            qd[:] = qd_[-self.max_seq:]
            qms[:] = qms_[-self.max_seq:]
            qattempt[:] = qattempt[-self.max_seq:]
            qa[:] = qa_[-self.max_seq:]
        # 如果当前user的答题个数小于max_seq，max_seq里前面是0座padding,后面是实际的答题记录
        else:
            q[-seq_len:] = q_
            s[-seq_len:] = s_
            qtype[-seq_len:] = qtype_
            qd[-seq_len:] = qd_
            qa[-seq_len:] = qa_
            qms[-seq_len:] = qms_
            qattempt[-seq_len:] = qattempt_
        
        target_qid = q[1:] # q_idx
        target_sid = s[1:] # s_idx
        target_qtype = qtype[1:] # q_type
        target_qd = qd[1:] # q_diff
        target_qms = qms[1:] # ms_first_response
        target_qattempt = qattempt[1:] # attempt_count
        label = qa[1:] # correct

        # interaction：yt = qt + rt × E
        x = np.zeros(self.max_seq-1, dtype=int)
        x = q[:-1].copy()
        x += (qa[:-1] == 1) * self.n_question

        return x, target_qid, target_sid, target_qtype, target_qd, target_qms, target_qattempt, label

        




        

    

    