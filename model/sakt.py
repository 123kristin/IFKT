import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pytorch_lightning as pl

from sklearn.metrics import roc_auc_score

drop_out = 0.5


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


class SAKT(nn.Module):
    def __init__(self, n_question, max_seq, embed_dim):
        super(SAKT, self).__init__()
        self.n_question = n_question
        self.embed_dim = embed_dim
        self.seq_len = max_seq

        self.embedding = nn.Embedding(2*n_question+1, embed_dim)
        self.e_embedding  = nn.Embedding(n_question+1, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=drop_out)

        self.dropout = nn.Dropout(drop_out)
        self.layer_normal = nn.LayerNorm(embed_dim)

        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)

    def forward(self, x, question_ids):
        # interact embedding
        x = self.embedding(x)
        pos_id = torch.arange(x.size(1)).unsqueeze(0).cuda()
        pos_x = self.pos_embedding(pos_id)
        x = x + pos_x

        e = self.e_embedding(question_ids)

        x = x.permute(1, 0, 2)  # x: [bs, s_len, embed] => [s_len, bs, embed]
        e = e.permute(1, 0, 2)

        # bs, seq_len, embed_dim
        k_in, v_in = x, x   # interact embedding is the k and v

        att_mask = future_mask(x.size(0)).cuda()
        att_output, att_weight = self.multi_att(
            e, k_in, v_in, attn_mask=att_mask)
        # changed
        att_output = self.layer_normal(att_output + e)
        # att_output: [s_len, bs, embed] => [bs, s_len, embed]
        att_output = att_output.permute(1, 0, 2)

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1), att_weight


class SAKTModule(pl.LightningModule):
    def __init__(self, n_question, max_seq, embed_dim):
        super(SAKTModule, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.sakt = SAKT(n_question, max_seq, embed_dim)

    def forward(self, x, question_ids):
        return self.sakt(x, question_ids)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, target_qid, label = batch

        label = label.float()
        target_mask = (target_qid != 0)

        output, _ = self(x, target_qid)
        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)

        loss = self.loss(output, label)
        self.log("t_loss", loss, prog_bar=True)

        return {'loss': loss, 'output': output, 'label': label}

    def training_epoch_end(self, training_ouput):
        out = torch.cat([i["output"] for i in training_ouput])
        labels = torch.cat([i["label"] for i in training_ouput])

        pred = (torch.sigmoid(out) >= 0.5).long()
        acc = (pred == labels).sum() / len(pred)

        out, labels = out.cpu().detach().numpy(), labels.cpu().detach().numpy()
        auc = roc_auc_score(labels, out)

        self.log_dict({'train_acc': acc, 'train_auc': auc}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        x, target_qid, label = batch

        label = label.float()
        target_mask = (target_qid != 0)

        output, _ = self(x, target_qid)
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
