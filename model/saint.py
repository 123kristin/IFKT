import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

dropout_rate = 0.2


class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """

    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff, out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff, out_features=dim_ff)

    def forward(self, ffn_in):
        return self.layer2(F.relu(self.layer1(ffn_in)))


class Encoder_block(nn.Module):
    """
    M = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    O = SkipConct(FFN(LayerNorm(M)))
    """

    def __init__(self, dim_model, heads_en, total_ex, total_cat, seq_len, dropout=0.5):
        super().__init__()
        self.dim_model = dim_model
        self.seq_len = seq_len
        # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
        self.embd_ex = nn.Embedding(total_ex, embedding_dim=dim_model)
        self.embd_cat = nn.Embedding(total_cat, embedding_dim=dim_model)

        self.multi_en = nn.MultiheadAttention(
            embed_dim=dim_model, num_heads=heads_en,)     # multihead attention
        # feedforward block
        self.ffn_en = Feed_Forward_block(dim_model)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)

    def forward(self, in_ex, in_cat, first_block=True):

        if first_block:
            in_ex = self.embd_ex(in_ex)
            in_cat = self.embd_cat(in_cat)
            in_pos = position_embedding(
                in_ex.shape[0], self.seq_len-1, self.dim_model)
            # combining the embedings
            out = in_ex + in_cat + in_pos                      # (b,n,d)
        else:
            out = in_ex

        out = self.dropout(out)
        # (n,b,d)  # print('pre multi', out.shape )
        out = out.permute(1, 0, 2)

        # Multihead attention
        n, _, _ = out.shape
        skip_out = out
        out, attn_wt = self.multi_en(out, out, out,
                                     attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        out = self.dropout(out)
        out = out + skip_out                                    # skip connection
        out = self.layer_norm1(out)                           # Layer norm

        # feed forward
        out = out.permute(1, 0, 2)                                # (b,n,d)
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout(out)
        out = out + skip_out                                    # skip connection
        out = self.layer_norm2(out)                           # Layer norm

        return out


class Decoder_block(nn.Module):
    """
    M1 = SkipConct(Multihead(LayerNorm(Qin;Kin;Vin)))
    M2 = SkipConct(Multihead(LayerNorm(M1;O;O)))
    L = SkipConct(FFN(LayerNorm(M2)))
    """

    def __init__(self, dim_model, total_in, heads_de, seq_len, dropout=0.5):
        super().__init__()
        self.dim_model = dim_model
        self.seq_len = seq_len
        self.embd_in = nn.Embedding(
            total_in, embedding_dim=dim_model)  # interaction embedding
        # M1 multihead for interaction embedding as q k v
        self.multi_de1 = nn.MultiheadAttention(
            embed_dim=dim_model, num_heads=heads_de)
        # M2 multihead for M1 out, encoder out, encoder out as q k v
        self.multi_de2 = nn.MultiheadAttention(
            embed_dim=dim_model, num_heads=heads_de)
        # feed forward layer
        self.ffn_en = Feed_Forward_block(dim_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

    def forward(self, in_in, en_out, first_block=True):

        if first_block:
            in_in = self.embd_in(in_in)
            in_pos = position_embedding(
                in_in.shape[0], self.seq_len-1, self.dim_model)
            # combining the embedings
            out = in_in + in_pos                                    # (b,n,d)
        else:
            out = in_in

        out = self.dropout(out)
        # (n,b,d)# print('pre multi', out.shape )
        out = out.permute(1, 0, 2)

        # Multihead attention M1
        n, _, _ = out.shape
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out,
                                      attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        out = self.dropout(out)
        out = skip_out + out                                        # skip connection
        out = self.layer_norm1(out)

        # Multihead attention M2
        # (b,n,d)-->(n,b,d)
        en_out = en_out.permute(1, 0, 2)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out,
                                      attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        out = out + skip_out
        en_out = self.layer_norm2(en_out)

        # feed forward
        out = out.permute(1, 0, 2)                                    # (b,n,d)
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout(out)
        out = out + skip_out                                        # skip connection
        out = self.layer_norm3(out)                               # Layer norm

        return out


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_mask(seq_len):
    return torch.from_numpy(np.triu(np.ones((seq_len, seq_len)), k=1).astype('bool')).cuda()


def position_encoding(pos, dim_model):
    # Encode one position with sin and cos
    # Attention Is All You Need uses positinal sines, SAINT paper does not specify
    pos_enc = np.zeros(dim_model)
    for i in range(0, dim_model, 2):
        pos_enc[i] = np.sin(pos / (10000 ** (2 * i / dim_model)))
        pos_enc[i + 1] = np.cos(pos / (10000 ** (2 * i / dim_model)))
    return pos_enc


def position_embedding(bs, seq_len, dim_model):
    # Return the position embedding for the whole sequence
    pe_array = np.array([[position_encoding(pos, dim_model)
                        for pos in range(seq_len)]] * bs)
    return torch.from_numpy(pe_array).float().cuda()


class saint(nn.Module):
    def __init__(self, dim_model, num_en, num_de, heads_en, total_ex, total_cat, total_in, heads_de, seq_len, dropout=0.2):
        super().__init__()

        self.num_en = num_en
        self.num_de = num_de

        self.encoder = get_clones(Encoder_block(
            dim_model, heads_en, total_ex, total_cat, seq_len, dropout), num_en)
        self.decoder = get_clones(Decoder_block(
            dim_model, total_in, heads_de, seq_len, dropout), num_de)

        self.out = nn.Linear(in_features=dim_model, out_features=1)

    def forward(self, in_ex, in_cat,  in_in):

        # pass through each of the encoder blocks in sequence
        first_block = True
        for x in range(self.num_en):
            if x >= 1:
                first_block = False
            in_ex = self.encoder[x](in_ex, in_cat, first_block=first_block)
            # passing same output as q,k,v to next encoder block
            in_cat = in_ex

        # pass through each decoder blocks in sequence
        first_block = True
        for x in range(self.num_de):
            if x >= 1:
                first_block = False
            in_in = self.decoder[x](
                in_in, en_out=in_ex, first_block=first_block)

        # Output layer
        in_in = self.out(in_in).squeeze(-1)
        return in_in


class SAINTModule(pl.LightningModule):
    def __init__(self, dim_model, num_en, num_de, heads_en, total_ex, total_cat, total_in, heads_de, seq_len):
        super(SAINTModule, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.model = saint(dim_model, num_en, num_de, heads_en,
                           total_ex, total_cat, total_in, heads_de, seq_len)
        self.n_question = total_ex

    def forward(self, target_qid, target_qtype, response):
        return self.model(target_qid, target_qtype, response)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        target_qid, target_qtype, response, label = batch

        label = label.float()
        target_mask = (target_qid != 0)

        output = self(target_qid, target_qtype, response)
        # print(output.size(), target_mask.size())

        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)

        loss = self.loss(output, label)
        self.log("t_loss", loss, prog_bar=True)

        return {'loss': loss, 'output': output, 'label': label}

    def training_epoch_end(self, training_ouput):
        out = torch.cat([i["output"] for i in training_ouput])
        labels = torch.cat([i["label"] for i in training_ouput])

        pred = (torch.sigmoid(out) >= 0.5).long()
        # pred = (out >= 0.5).long()
        acc = (pred == labels).sum() / len(pred)

        out, labels = out.cpu().detach().numpy(), labels.cpu().detach().numpy()
        auc = roc_auc_score(labels, out)

        self.log_dict({'train_acc': acc, 'train_auc': auc}, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        target_qid, target_qtype, response, label = batch

        label = label.float()
        target_mask = (target_qid != 0)

        output = self(target_qid, target_qtype, response)
        # print(output.size(), target_mask.size())
        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)

        loss = self.loss(output, label)
        self.log("v_loss", loss, prog_bar=True)

        return {'val_loss': loss, 'output': output, 'label': label}

    def validation_epoch_end(self, validation_ouput):
        out = torch.cat([i["output"] for i in validation_ouput])
        labels = torch.cat([i["label"] for i in validation_ouput])

        pred = (torch.sigmoid(out) >= 0.5).long()
        # pred = (out >= 0.5).long()
        acc = (pred == labels).sum() / len(pred)

        out, labels = out.cpu().detach().numpy(), labels.cpu().detach().numpy()
        auc = roc_auc_score(labels, out)

        self.log_dict({'v_auc': auc, 'v_acc': acc}, prog_bar=True)


# # forward prop on dummy data

# seq_len = 100
# total_ex = 1200
# total_cat = 234
# total_in = 2


# def random_data(bs, seq_len, total_ex, total_cat, total_in=2):
#     ex = torch.randint(0, total_ex, (bs, seq_len))
#     cat = torch.randint(0, total_cat, (bs, seq_len))
#     de = torch.randint(0, total_in, (bs, seq_len))
#     return ex, cat, de


# in_ex, in_cat, in_de = random_data(64, seq_len, total_ex, total_cat, total_in)


# model = saint(dim_model=128,
#               num_en=6,
#               num_de=6,
#               heads_en=8,
#               heads_de=8,
#               total_ex=total_ex,
#               total_cat=total_cat,
#               total_in=total_in,
#               seq_len=seq_len
#               )

# outs = model(in_ex, in_cat, in_de)

# print(outs.shape)
