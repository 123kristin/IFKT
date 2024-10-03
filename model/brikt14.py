import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score
from model.lightgcn import LightGCNStack
from model.rpe_att import MultiHeadAttentionLayer


# drop_out = 0.9

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
    future_mask = np.triu(
        np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)


class MYMODEL(nn.Module):
    def __init__(self, n_question, n_qtype, max_seq, embed_dim, n_query_features, n_gcn_layers, gcn_data, dropout, new_att=False):
        super(MYMODEL, self).__init__()
        self.n_question = n_question
        self.seq_len = max_seq
        self.embed_dim = embed_dim
        self.n_query_features = n_query_features

        self.embedding = nn.Embedding(2*n_question+1, embed_dim) # interaction embedding
        self.qt_embedding = nn.Embedding(n_qtype, embed_dim) # problem type embedding
        # difficulty, mini second time, count of attempt
        self.diff_embedding = nn.Linear(1, embed_dim) # difficulty
        self.ms_first_response_embedding = nn.Linear(1, embed_dim) # mini second time 
        self.attempt_count_embedding = nn.Linear(1, embed_dim) # count of attempt
        # 拼接6个特征 e, s, qt, d, attempt, ms
        self.combine_q_embedding = nn.Linear(embed_dim * n_query_features, embed_dim)
        self.q_feature_layer_norm = nn.LayerNorm([max_seq-1, embed_dim]) # 对每一个batch进行归一化

        # LightGCN
        self.gcn = LightGCNStack(embed_dim=embed_dim, num_layers=n_gcn_layers, dataset=gcn_data) 
        # MultiheadAttention
        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=dropout) if not new_att else MultiHeadAttentionLayer(embed_dim, 8, dropout, 'cuda')
        self.dropout = nn.Dropout(dropout)
        self.layer_normal = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
        
    def forward(self, x, question_ids, skill_ids, qt_ids, diff, ms_response_norm, attempt_count_norm):
        # 初步interaction embedding
        x = self.embedding(x)
        # 初步problem type embedding
        qt = self.qt_embedding(qt_ids)

        d_list, ms_list, attempt_list = [], [], []
        # difficulty
        for i in range(diff.shape[1]):
            tmp = self.diff_embedding(
                diff[:, i].unsqueeze(1).float()).unsqueeze(2)
            d_list.append(tmp)
        d = torch.cat(d_list, dim=2).long()
        d = d.permute(2, 0, 1)
        del d_list
        # mini second time
        for i in range(ms_response_norm.shape[1]):
            tmp = self.ms_first_response_embedding(
                ms_response_norm[:, i].unsqueeze(1).float()).unsqueeze(2)
            ms_list.append(tmp)
        ms = torch.cat(ms_list, dim=2).long()
        ms = ms.permute(2, 0, 1)
        del ms_list
        # count of attempt
        for i in range(attempt_count_norm.shape[1]):
            tmp = self.attempt_count_embedding(
                attempt_count_norm[:, i].unsqueeze(1).float()).unsqueeze(2)
            attempt_list.append(tmp)
        attempt = torch.cat(attempt_list, dim=2).long()
        attempt = attempt.permute(2, 0, 1)
        del attempt_list

        # get question and skill embeddings by lightgcn
        W_Q, W_S = self.gcn.get_embeddings()
        gcn_e = F.embedding(question_ids, W_Q) # 采用T-sne对question embedding和skill embedding做可视化分析
        gcn_s = F.embedding(skill_ids, W_S)

        x = x.permute(1, 0, 2)  # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = gcn_e.permute(1, 0, 2)
        s = gcn_s.permute(1, 0, 2)
        qt = qt.permute(1, 0, 2)
        # 拼接6个特征：e, s, qt, d, attempt, ms
        combine_list = []
        # bs, seq_len, embed_dim
        combine_embed = torch.cat((e, s, qt, d, attempt, ms), dim=2)
        for i in range(combine_embed.shape[0]):
            tmp = self.combine_q_embedding(combine_embed[i, :, :].float()).unsqueeze(0)
            combine_list.append(tmp)
        q_in = torch.cat(combine_list, dim=0)
        del combine_list
        q_in = self.q_feature_layer_norm(q_in.permute(1, 0, 2)).permute(1, 0, 2)
        # k_in, v_in = q_in, x # 
        k_in, v_in = x, x
        # mask Multi-head attention
        att_mask = future_mask(x.size(0)).cuda()
        att_output, att_weight = self.multi_att(q_in, k_in, v_in, attn_mask=att_mask)
        # changed
        att_output = self.layer_normal(att_output + q_in)
        # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        att_output = att_output.permute(1, 0, 2)

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1), att_weight, gcn_e, gcn_s  


class MYModule(pl.LightningModule):
    def __init__(self, n_question, n_skill, n_qtype, max_seq, embed_dim, n_query_features, n_gcn_layers, gcn_data, dropout, new_att):
        super(MYModule, self).__init__()
        print(n_question, n_skill, n_qtype, max_seq, embed_dim, n_query_features, gcn_data)
        self.loss = nn.BCEWithLogitsLoss()
        self.model = MYMODEL(n_question, n_qtype, max_seq, embed_dim, n_query_features, n_gcn_layers, gcn_data, dropout, new_att)
        self.n_question = n_question

    def forward(self, x, question_ids, skill_ids, qt_ids, diff, ms_response_norm, attempt_count_norm):
        return self.model(x, question_ids, skill_ids, qt_ids, diff, ms_response_norm, attempt_count_norm)
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    # 返回要反向传播的loss即可
    # 其中batch 即为从 train_dataloader 采样的一个batch的数据，batch_idx即为目前batch的索引
    def training_step(self, batch, batch_idx):
        x, target_qid, target_sid, target_qtype, target_qd, target_qms, target_qattempt, label = batch

        label = label.float()
        target_mask = (target_qid != 0) 

        output, _, _, _ = self(x, target_qid, target_sid, target_qtype, target_qd, target_qms, target_qattempt)
        output = torch.masked_select(output, target_mask) # mask当前时刻后的信息
        label = torch.masked_select(label, target_mask)

        loss = self.loss(output, label)
        # 标记该loss，用于保存模型时监控该量，Tensorboard 损失/指标日志保存和查看
        self.log("train_loss", loss, prog_bar=True)  

        return {'loss': loss, 'output': output, 'label': label}


    # 在每一个 * 的epoch 完成之后会自动调用
    def training_epoch_end(self, training_ouput):
        out = torch.cat([i["output"] for i in training_ouput])
        labels = torch.cat([i["label"] for i in training_ouput])

        pred = (torch.sigmoid(out) >= 0.5).long()
        # 计算acc，pred>=0.5的则设为预测正确的
        acc = (pred == labels).sum() / len(pred)
        # 计算auc
        out, labels = out.cpu().detach().numpy(), labels.cpu().detach().numpy()
        auc = roc_auc_score(labels, out)
        # 加载多个信息：acc和auc
        self.log_dict({'train_acc': acc, 'train_auc': auc}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, target_qid, target_sid, target_qtype, target_qd, target_qms, target_qattempt, label = batch
        label = label.float()
        target_mask = (target_qid != 0)

        output, _, _, _ = self(x, target_qid, target_sid, target_qtype,
                         target_qd, target_qms, target_qattempt)
        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)

        loss = self.loss(output, label)
        self.log("val_loss", loss, prog_bar=True)

        return {'val_loss': loss, 'output': output, 'label': label}
        
    def validation_epoch_end(self, validation_ouput):
        out = torch.cat([i["output"] for i in validation_ouput])
        labels = torch.cat([i["label"] for i in validation_ouput])

        pred = (torch.sigmoid(out) >= 0.5).long()
        acc = (pred == labels).sum() / len(pred)

        out, labels = out.cpu().detach().numpy(), labels.cpu().detach().numpy()
        auc = roc_auc_score(labels, out)

        self.log_dict({'v_auc': auc, 'v_acc': acc}, prog_bar=True)
