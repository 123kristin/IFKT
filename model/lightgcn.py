# 参考https://github.com/tm1897/mlg_cs224w_project/tree/main
import torch
import torch.nn as nn
import torch_scatter

from torch_geometric.nn.conv import MessagePassing

class LightGCNStack(nn.Module):
    """
    ### params: \n
    embed_dim: embedding dimension \n
    num_layers: num of lightgcn conv \n
    dataset: birepartite dataset, see birepartited_data.py \n
    """

    def __init__(self, embed_dim, num_layers, dataset):
        super(LightGCNStack, self).__init__()
        self.dataset = dataset # 多余
        assert (num_layers >= 0), 'Number of layers is not >=0'
        self.convs = nn.ModuleList([LightGCN(embed_dim)
                                   for _ in range(num_layers)])
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.dataset = dataset
        self.embed_q = nn.Embedding(dataset.num_q, self.embed_dim).cuda()
        self.embed_s = nn.Embedding(dataset.num_s, self.embed_dim).cuda()

    def reset_parameters(self):
        self.embeddings.reset_parameters()

    def forward(self):
        w_q, w_s = self.embed_q.weight, self.embed_s.weight
        final_embed_q = torch.zeros(size=w_q.size(), device='cuda')
        final_embed_s = torch.zeros(size=w_s.size(), device='cuda')
        final_embed_q = final_embed_q + w_q / (self.num_layers + 1)
        final_embed_s = final_embed_s + w_s / (self.num_layers + 1)

        # lightgcn convolution
        for i in range(self.num_layers):
            w_q = self.convs[i]((w_s, w_q), self.dataset.edge_index_s2q, size=(
                self.dataset.num_s, self.dataset.num_q))
            w_s = self.convs[i]((w_q, w_s), self.dataset.edge_index_q2s, size=(
                self.dataset.num_q, self.dataset.num_s))
            #
            final_embed_q = final_embed_q + w_q / (self.num_layers + 1)
            final_embed_s = final_embed_s + w_s / (self.num_layers + 1)

        return final_embed_q, final_embed_s

    # batch_size, seq_len
    def get_embeddings(self):
        # if self.num_layers == 0:
        #     return self.embed_q(question_ids), self.embed_s(skill_ids)

        w_q, w_s = self.forward()
        return w_q, w_s

        # wq_embed, ws_embed = nn.Embedding.from_pretrained(
        #     w_q), nn.Embedding.from_pretrained(w_s)
        # q_embed, s_embed = wq_embed(question_ids), ws_embed(skill_ids)
        # del wq_embed
        # del ws_embed
        

class LightGCN(MessagePassing):
    def __init__(self, embed_dim, **kwargs):
        super(LightGCN, self).__init__(node_dim=0, **kwargs)
        self.embed_dim = embed_dim

    def forward(self, x, edge_index, size=None):
        return self.propagate(edge_index=edge_index, x=(x[0], x[1]), size=size)

    def message(self, x_j):
        return x_j

    def aggregate(self, inputs, index, dim_size=None):
        return torch_scatter.scatter(src=inputs, index=index, dim=0, dim_size=dim_size, reduce='mean')