import pandas as pd
from torch_geometric.data import Data
import torch

class BipartiteData(Data):
    def __init__(self, edge_index_q2s=None, edge_index_s2q=None, num_s=None, num_q=None):
        super().__init__()
        self.edge_index_q2s = edge_index_q2s
        self.edge_index_s2q = edge_index_s2q
        self.num_q = num_q
        self.num_s = num_s

    def __inc__(self, key, value, *args, **kwargs):
        """ 
            Returns the incremental count to cumulatively increase the value of the next attribute of :obj:`key` when creating batches.
        """
        if key == 'edge_index_q2s':
            return torch.tensor([[self.num_q], [self.num_s]])
        elif key == 'edge_index_s2q':
            return torch.tensor([[self.num_s], [self.num_q]])
        else:
            return super(BipartiteData, self).__inc__(key, value)

def load_bipartitedata(df_data):
    """ 
        ### args: \n
        df_data: pandas dataframe with columns 'qid', 'sid', 'weight'
    """
    q_node_ids = 0
    q_node_id_dict = dict()
    s_node_ids = 0
    s_node_id_dict = dict()

    q_reverse_dict = dict()
    s_reverse_dict = dict()
    edge_data_q2s = []
    edge_data_s2q = []

    for index, row in df_data.iterrows():
        if str(row['qid']) not in q_node_id_dict.keys():
            q_node_id_dict[str(row['qid'])] = q_node_ids
            q_reverse_dict[q_node_ids] = row['qid']
            q_node_ids = q_node_ids + 1
        if str(row['sid']) not in s_node_id_dict.keys():
            s_node_id_dict[str(row['sid'])] = s_node_ids
            s_reverse_dict[s_node_ids] = row['sid']
            s_node_ids = s_node_ids + 1
        edge_data_q2s.append(
            [q_node_id_dict[str(row['qid'])], s_node_id_dict[str(row['sid'])]])
        edge_data_s2q.append(
            [s_node_id_dict[str(row['sid'])], q_node_id_dict[str(row['qid'])]])

    gcn_data = BipartiteData(edge_index_q2s=torch.LongTensor(edge_data_q2s).t().contiguous(),
                         edge_index_s2q=torch.LongTensor(edge_data_s2q).t().contiguous(),
                         num_q=q_node_ids, num_s=s_node_ids).to('cuda')
    return gcn_data, q_reverse_dict, s_reverse_dict, q_node_id_dict, s_node_id_dict

