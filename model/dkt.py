from turtle import forward
from matplotlib.pyplot import flag
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pytorch_lightning as pl
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score


class DKT(nn.Module):
    def __init__(self, n_question, hidden_size=256, emb_dim=128, n_layers=2):

        super(DKT, self).__init__()
        self.n_question = n_question
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(2*n_question+1, emb_dim)

        self.lstm = nn.LSTM(emb_dim, hidden_size, n_layers,
                            batch_first=True, dropout=0.2)

        self.pred = nn.Linear(hidden_size, n_question)

    def forward(self, x):
        x = self.embedding(x)
        # lstm output:[bs, seq_len, hidden] hidden [bs, hidden]
        x, _ = self.lstm(x)
        return self.pred(x)


class DKTModule(pl.LightningModule):
    def __init__(self, n_question):
        super(DKTModule, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.dkt = DKT(n_question)
        self.n_question = n_question

    def forward(self, x):
        return self.dkt(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, qid, label = batch
        label = label.float()
        qid = qid.long()
        target_mask = (x != 0)
        output = self(x)
        output = torch.gather(output, -1, qid.unsqueeze(2))
        output = torch.masked_select(output.squeeze(2), target_mask)
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
        x, qid, label = batch
        label = label.float()
        qid = qid.long()
        target_mask = (x != 0)

        output = self(x)
        #         The output yt is a vector of length equal to the number of problems, where each entry represents
        # the predicted probability that the student would answer that particular problem correctly. Thus the
        # prediction of at+1 can then be read from the entry in yt corresponding to qt+1.

        output = torch.gather(output, 2, qid.unsqueeze(2))
        output = torch.masked_select(output.squeeze(2), target_mask)
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
