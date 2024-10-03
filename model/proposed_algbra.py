import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pytorch_lightning as pl

from model.lightgcn import LightGCNStack
from sklearn.metrics import roc_auc_score

from model.rpe_att import MultiHeadAttentionLayer


class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)


def future_mask(seq_length):
    # 生成一个bool类型的上三角mask矩阵。对nn.MultiheadAttention, mask矩阵为true的将不参与注意力计算
    future_mask = np.triu(
        np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class MYMODEL(nn.Module):
    def __init__(self, n_question, n_qtype, max_seq, embed_dim, n_query_features, n_gcn_layers, gcn_data, dropout, new_att):
        super(MYMODEL, self).__init__()
        self.n_question = n_question
        self.embed_dim = embed_dim
        self.seq_len = max_seq
        self.n_query_features = n_query_features   # 有多少个要拼接的特征向量

        self.embedding = nn.Embedding(2*n_question+1, embed_dim)
        # self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)
        # self.pos_embedding = LearnablePositionalEmbedding(embed_dim, max_seq)
        self.qt_embedding = nn.Embedding(n_qtype, embed_dim)

        # 对这种非离散特征，使用线性层升维
        self.diff_embedding = nn.Linear(1, embed_dim)
        self.ms_first_response_embedding = nn.Linear(1, embed_dim)
        self.attempt_count_embedding = nn.Linear(1, embed_dim)

        # 特征拼接层
        self.combine_q_embedding = nn.Linear(
            embed_dim * n_query_features, embed_dim)
        self.q_feature_layer_norm = nn.LayerNorm([max_seq-1, embed_dim])

        self.gcn = LightGCNStack(
            embed_dim=embed_dim, num_layers=n_gcn_layers, dataset=gcn_data)

        self.multi_att = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=8, dropout=dropout) if not new_att else MultiHeadAttentionLayer(embed_dim, 8, dropout, 'cuda')

        self.dropout = nn.Dropout(dropout)
        self.layer_normal = nn.LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)

    # def forward(self, x, question_ids, skill_ids, qt_ids, diff, ms_response_norm, attempt_count_norm):
    def forward(self, x, question_ids, skill_ids, qt_ids, diff, ms_response_norm):
        # interact embedding
        # x: [bs, s_len, embed] => [s_len, bs, embed]
        x = self.embedding(x).permute(1, 0, 2)
        # pos_id = torch.arange(x.size(1)).unsqueeze(0).cuda()
        # pos_x = self.pos_embedding(pos_id)
        # x = x + pos_x

        qt = self.qt_embedding(qt_ids).permute(1, 0, 2)
        # get question and skill embedding matrix by lightgcn
        w_q, w_s = self.gcn.get_embeddings()
        # F.embedding函数：传入要嵌入的数据和嵌入矩阵，得到对应的嵌入（和nn.Embedding的作用一样）
        e = F.embedding(question_ids, w_q).permute(1, 0, 2)
        s = F.embedding(skill_ids, w_s).permute(1, 0, 2)

        d_list, ms_list, attempt_list = [], [], []

        # 对难度、响应时间、尝试次数这种连续性特征用一个线性层进行嵌入（1->128）
        for i in range(diff.shape[1]):  # diff is [batch_size, seq_len]
            # tmp is [batch, embed_dim, 1]
            tmp = self.diff_embedding(
                diff[:, i].unsqueeze(1).float()).unsqueeze(2)
            d_list.append(tmp)
        # d is [batch_size, embed_dim, seq_len] -> [seq_len, batch_size, embed_dim]
        d = torch.cat(d_list, dim=2).long().permute(2, 0, 1)
        del d_list  # 为了节约一点内存

        for i in range(ms_response_norm.shape[1]):
            tmp = self.ms_first_response_embedding(
                ms_response_norm[:, i].unsqueeze(1).float()).unsqueeze(2)
            ms_list.append(tmp)
        ms = torch.cat(ms_list, dim=2).long().permute(2, 0, 1)
        del ms_list

        # for i in range(attempt_count_norm.shape[1]):
        #     tmp = self.attempt_count_embedding(
        #         attempt_count_norm[:, i].unsqueeze(1).float()).unsqueeze(2)
        #     attempt_list.append(tmp)
        # attempt = torch.cat(attempt_list, dim=2).long().permute(2, 0, 1)
        # del attempt_list

        # 这里用concat-linear的方式进行特征融合
        combine_list = []
        # bs, seq_len, embed_dim * n_query_features
        # combine_embed = torch.cat((e, s, qt, attempt, d, ms), dim=2)
        combine_embed = torch.cat((e, s, qt, d, ms), dim=2)
        for i in range(combine_embed.shape[0]):
            tmp = self.combine_q_embedding(
                combine_embed[i, :, :].float()).unsqueeze(0)
            combine_list.append(tmp)
        q_in = torch.cat(combine_list, dim=0)
        del combine_list
        # Also, in the final version of the model, I added a
        # layer normalization layer (Ba, Kiros, and Hinton 2016) after the content embedding layer. It helps regularize the layer
        # weights and give a small improvement in performance.
        q_in = self.q_feature_layer_norm(
            q_in.permute(1, 0, 2)).permute(1, 0, 2)

        # AKT:In our experiments, we have found that using question embeddings for mapping both queries and keys is much more effective.
        # interact embedding is the  v
        k_in, v_in = q_in, x

        att_mask = future_mask(x.size(0)).cuda()
        att_output, att_weight = self.multi_att(
            q_in, k_in, v_in, attn_mask=att_mask)
        # changed
        att_output = self.layer_normal(att_output + q_in)
        # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        att_output = att_output.permute(1, 0, 2)

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1), att_weight


class MYModule(pl.LightningModule):
    """
    使用torch_lightning的方式实现模型，可以规范代码，并且自带早停，保存loss、最好的模型等功能 
    """

    def __init__(self, n_question, n_skill, n_qtype, max_seq, embed_dim, n_query_features, n_gcn_layers, gcn_data, dropout, new_att):
        super(MYModule, self).__init__()
        print(n_question, n_skill, n_qtype, max_seq,
              embed_dim, n_query_features, gcn_data)
        self.loss = nn.BCEWithLogitsLoss()
        self.model = MYMODEL(n_question, n_qtype, max_seq,
                             embed_dim, n_query_features, n_gcn_layers, gcn_data, dropout, new_att)
        self.n_question = n_question

    # def forward(self, x, question_ids, skill_ids, qt_ids, diff, ms_response_norm, attempt_count_norm):
    def forward(self, x, question_ids, skill_ids, qt_ids, diff, ms_response_norm):

        # return self.model(x, question_ids, skill_ids, qt_ids, diff, ms_response_norm, attempt_count_norm)
        return self.model(x, question_ids, skill_ids, qt_ids, diff, ms_response_norm)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    """ 
    这个函数执行一个epoch上对所有batch的训练，并记录每个batch的预测和loss
    """

    def training_step(self, batch, batch_idx):
        # x, target_qid, target_sid, target_qtype, target_qd, target_qms, target_qattempt, label = batch
        x, target_qid, target_sid, target_qtype, target_qd, target_qms, label = batch
        label = label.float()
        # mask，因为计算loss和其他指标的时候要把padding的数据去掉
        target_mask = (target_qid != 0)

        # output, _ = self(x, target_qid, target_sid, target_qtype,
                        #  target_qd, target_qms, target_qattempt)
        output, _ = self(x, target_qid, target_sid, target_qtype,
                         target_qd, target_qms)
        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)

        loss = self.loss(output, label)
        self.log("t_loss", loss, prog_bar=True)

        return {'loss': loss, 'output': output, 'label': label}

    """ 
    执行完一个epoch，对每个batch的预测和label进行拼接，得到整个训练（或者测试集）上的预测和真实值，然后计算acc、auc。
    """

    def training_epoch_end(self, training_ouput):
        out = torch.cat([i["output"] for i in training_ouput])
        labels = torch.cat([i["label"] for i in training_ouput])

        pred = (torch.sigmoid(out) >= 0.5).long()
        acc = (pred == labels).sum() / len(pred)

        out, labels = out.cpu().detach().numpy(), labels.cpu().detach().numpy()
        auc = roc_auc_score(labels, out)

        self.log_dict({'train_acc': acc, 'train_auc': auc}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        # x, target_qid, target_sid, target_qtype, target_qd, target_qms, target_qattempt, label = batch
        x, target_qid, target_sid, target_qtype, target_qd, target_qms, label = batch

        label = label.float()
        target_mask = (target_qid != 0)

        # output, _ = self(x, target_qid, target_sid, target_qtype,
                        #  target_qd, target_qms, target_qattempt)
        output, _ = self(x, target_qid, target_sid, target_qtype,
                         target_qd, target_qms)
        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)

        loss = self.loss(output, label)
        self.log("v_loss", loss, prog_bar=True)

        return {'val_loss': loss, 'output': output, 'label': label}

    def validation_epoch_end(self, validation_ouput):
        out = torch.cat([i["output"] for i in validation_ouput])
        labels = torch.cat([i["label"] for i in validation_ouput])

        pred = (torch.sigmoid(out) >= 0.5).long()
        acc = (pred == labels).sum() / len(pred)

        out, labels = out.cpu().detach().numpy(), labels.cpu().detach().numpy()
        auc = roc_auc_score(labels, out)

        self.log_dict({'v_auc': auc, 'v_acc': acc}, prog_bar=True)
