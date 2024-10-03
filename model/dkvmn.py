import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score


class DKVMNHeadGroup(nn.Module):
    def __init__(self, memory_size, memory_state_dim, is_write):
        super(DKVMNHeadGroup, self).__init__()
        """"
        Parameters
            memory_size:        scalar
            memory_state_dim:   scalar
            is_write:           boolean
        """
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim
        self.is_write = is_write
        if self.is_write:
            self.erase = torch.nn.Linear(
                self.memory_state_dim, self.memory_state_dim, bias=True)
            self.add = torch.nn.Linear(
                self.memory_state_dim, self.memory_state_dim, bias=True)
            nn.init.kaiming_normal_(self.erase.weight)
            nn.init.kaiming_normal_(self.add.weight)
            nn.init.constant_(self.erase.bias, 0)
            nn.init.constant_(self.add.bias, 0)

    def addressing(self, control_input, memory):
        """
        Parameters
            control_input:          Shape (batch_size, control_state_dim)
            memory:                 Shape (memory_size, memory_state_dim)
        Returns
            correlation_weight:     Shape (batch_size, memory_size)
        """
        similarity_score = torch.matmul(control_input, torch.t(memory))
        correlation_weight = torch.nn.functional.softmax(
            similarity_score, dim=1)  # Shape: (batch_size, memory_size)
        return correlation_weight

    def read(self, memory, control_input=None, read_weight=None):
        """
        Parameters
            control_input:  Shape (batch_size, control_state_dim)
            memory:         Shape (batch_size, memory_size, memory_state_dim)
            read_weight:    Shape (batch_size, memory_size)
        Returns
            read_content:   Shape (batch_size,  memory_state_dim)
        """
        if read_weight is None:
            read_weight = self.addressing(
                control_input=control_input, memory=memory)
        read_weight = read_weight.view(-1, 1)
        memory = memory.view(-1, self.memory_state_dim)
        rc = torch.mul(read_weight, memory)
        read_content = rc.view(-1, self.memory_size, self.memory_state_dim)
        read_content = torch.sum(read_content, dim=1)
        return read_content

    def write(self, control_input, memory, write_weight):
        """
        Parameters
            control_input:      Shape (batch_size, control_state_dim)
            write_weight:       Shape (batch_size, memory_size)
            memory:             Shape (batch_size, memory_size, memory_state_dim)
        Returns
            new_memory:         Shape (batch_size, memory_size, memory_state_dim)
        """
        assert self.is_write
        erase_signal = torch.sigmoid(self.erase(control_input))
        add_signal = torch.tanh(self.add(control_input))
        erase_reshape = erase_signal.view(-1, 1, self.memory_state_dim)
        add_reshape = add_signal.view(-1, 1, self.memory_state_dim)
        write_weight_reshape = write_weight.view(-1, self.memory_size, 1)
        erase_mult = torch.mul(erase_reshape, write_weight_reshape)
        add_mul = torch.mul(add_reshape, write_weight_reshape)
        new_memory = memory * (1 - erase_mult) + add_mul
        return new_memory


class DKVMN(nn.Module):
    def __init__(self, memory_size, memory_key_state_dim, memory_value_state_dim, init_memory_key):
        super(DKVMN, self).__init__()
        """
        :param memory_size:             scalar
        :param memory_key_state_dim:    scalar
        :param memory_value_state_dim:  scalar
        :param init_memory_key:         Shape (memory_size, memory_value_state_dim)
        :param init_memory_value:       Shape (batch_size, memory_size, memory_value_state_dim)
        """
        self.memory_size = memory_size
        self.memory_key_state_dim = memory_key_state_dim
        self.memory_value_state_dim = memory_value_state_dim

        self.key_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                       memory_state_dim=self.memory_key_state_dim,
                                       is_write=False)

        self.value_head = DKVMNHeadGroup(memory_size=self.memory_size,
                                         memory_state_dim=self.memory_value_state_dim,
                                         is_write=True)

        self.memory_key = init_memory_key

        self.memory_value = None

    def init_value_memory(self, memory_value):
        self.memory_value = memory_value

    def attention(self, control_input):
        correlation_weight = self.key_head.addressing(
            control_input=control_input, memory=self.memory_key)
        return correlation_weight

    def read(self, read_weight):
        read_content = self.value_head.read(
            memory=self.memory_value, read_weight=read_weight)

        return read_content

    def write(self, write_weight, control_input):
        memory_value = self.value_head.write(control_input=control_input,
                                             memory=self.memory_value,
                                             write_weight=write_weight)

        self.memory_value = nn.Parameter(memory_value.data)

        return self.memory_value


class DKVMNMODEL(nn.Module):

    def __init__(self, n_question, q_embed_dim=50, qa_embed_dim=100, memory_size=20, final_fc_dim=10):
        super(DKVMNMODEL, self).__init__()
        self.n_question = n_question
        self.q_embed_dim = q_embed_dim
        self.qa_embed_dim = qa_embed_dim
        self.memory_size = memory_size
        self.memory_key_state_dim = q_embed_dim
        self.memory_value_state_dim = qa_embed_dim
        self.final_fc_dim = final_fc_dim

        self.read_embed_linear = nn.Linear(
            self.memory_value_state_dim + self.memory_key_state_dim, self.final_fc_dim, bias=True)
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)
        self.init_memory_key = nn.Parameter(torch.randn(
            self.memory_size, self.memory_key_state_dim))
        nn.init.kaiming_normal_(self.init_memory_key)
        self.init_memory_value = nn.Parameter(torch.randn(
            self.memory_size, self.memory_value_state_dim))
        nn.init.kaiming_normal_(self.init_memory_value)

        self.mem = DKVMN(memory_size=self.memory_size,
                         memory_key_state_dim=self.memory_key_state_dim,
                         memory_value_state_dim=self.memory_value_state_dim, init_memory_key=self.init_memory_key)

        self.q_embed = nn.Embedding(
            self.n_question + 1, self.q_embed_dim, padding_idx=0)
        self.qa_embed = nn.Embedding(
            2 * self.n_question + 1, self.qa_embed_dim, padding_idx=0)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)

    def forward(self, qa_data, q_data):
        batch_size = q_data.shape[0]
        seqlen = q_data.shape[1]
        q_embed_data = self.q_embed(q_data)
        qa_embed_data = self.qa_embed(qa_data)

        memory_value = nn.Parameter(torch.cat(
            [self.init_memory_value.unsqueeze(0) for _ in range(batch_size)], 0).data)
        self.mem.init_value_memory(memory_value)

        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)

        value_read_content_l = []
        input_embed_l = []
        for i in range(seqlen):
            # Attention
            q = slice_q_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(q)

            # Read Process
            read_content = self.mem.read(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(q)

            # Write Process
            qa = slice_qa_embed_data[i].squeeze(1)
            self.mem.write(correlation_weight, qa)

        all_read_value_content = torch.cat(
            [value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1)
        input_embed_content = torch.cat(
            [input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1)

        predict_input = torch.cat(
            [all_read_value_content, input_embed_content], 2)
        read_content_embed = torch.tanh(self.read_embed_linear(
            predict_input.view(batch_size * seqlen, -1)))

        # [batch_size * seq_len, 1]
        pred = self.predict_linear(read_content_embed).view(-1, 1)
        pred = pred.view(batch_size, -1)
        # target_1d = target.view(-1, 1)  # [batch_size * seq_len, 1]
        # mask = target_1d.ge(1)  # [batch_size * seq_len, 1]

        # filtered_pred = torch.masked_select(pred_1d, mask)
        # filtered_target = torch.masked_select(target_1d, mask) - 1
        # loss = torch.nn.functional.binary_cross_entropy_with_logits(
        #     filtered_pred, filtered_target.float())

        # return loss, torch.sigmoid(filtered_pred), filtered_target.float()
        return pred


class DKVMNModule(pl.LightningModule):
    def __init__(self, n_question):
        super(DKVMNModule, self).__init__()
        self.loss = nn.BCEWithLogitsLoss()
        self.model = DKVMNMODEL(n_question)

    def forward(self, x, question_ids):
        return self.model(x, question_ids)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        x, target_qid, label = batch

        label = label.float()
        target_mask = (target_qid != 0)

        output = self(x, target_qid)
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

        output = self(x, target_qid)
        # print(output.size())
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
