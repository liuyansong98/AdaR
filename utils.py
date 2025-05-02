import numpy as np
import torch
import os
import random
import argparse
import sys
from torch import nn as nn
import settings as settings
import pandas as pd
from collections import OrderedDict


def get_args():
    parser = argparse.ArgumentParser('MyModel 2025')

    # General
    parser.add_argument('-d', '--data', type=str,
                        default='social_TKG_cate_level1_filter',
                        help='dataset to use')
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed')
    parser.add_argument('--gpu', type=int, default=0,
                        help='the GPU to be used')

    # Training-related
    parser.add_argument('--n_epoch', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--bs', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--drop_out', type=float, default=0.15,
                        help='dropout probability for all dropout layers')
    parser.add_argument('--tolerance', type=float, default=0,
                        help='tolerated marginal improvement for early stopper')

    # Model-related
    parser.add_argument('--n_layer', type=int, default=2,
                        help='number of layers to be sampled (only valid when n_degree has a single element)')
    parser.add_argument('--n_head', type=int, default=3,
                        help='number of heads for attention pooling')
    parser.add_argument('--pos_dim', type=int, default=60,
                        help='dimension of the positional encoding')
    parser.add_argument('--embed_dim', type=int, default=600,
                        help='dimension of the node and relation embedding')
    parser.add_argument('--temporal_bias', default=0.01, type=float,
                        help='temporal_bias')
    parser.add_argument('--solver', type=str, default='euler', choices=['euler', 'rk4', 'dopri5'],
                        help='the ODE solver to be used')
    parser.add_argument('--step_size', type=float, default=0.125,
                        help='step size to be used in fixed-step solvers (e.g., euler and rk4)')
    parser.add_argument('--limit_ngh_span', action='store_true',
                        help="whether to limit the maximum number of spanned temporal neighbors")
    parser.add_argument('--ngh_span', nargs='*', default=['2048', '16'],
                        help='a list of maximum number of spanned temporal neighbors for different hops')
    parser.add_argument('--find_score_weight', action='store_true', default=False,
                        help='use auto find score weight')
    parser.add_argument('--default_weight', type=float, default=0.1,
                        help='default score weight to be used in final score calculate')
    parser.add_argument('--case_study', action='store_true', default=False,
                        help='is conduct case study')
    parser.add_argument('--flexible_capacity', type=float, default=0.6,
                        help='the flexible capacity of history memory')
    parser.add_argument('--base_capacity', type=int, default=2,
                        help='the base capacity of history memory')
    parser.add_argument('--path_encode', type=str, default='LSTM',
                        choices=['ODE', 'LSTM', 'mamba', 'GRU', 'Transformer', 'LSTM_time', 'mamba_time', 'GRU_time',
                                 'Transformer_time'], help='the path encoder to be used')

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)

    return args, sys.argv

class EarlyStopMonitor(object):
    def __init__(self, max_round=4, higher_better=True, tolerance=1e-3):
        self.max_round = max_round
        self.num_round = 0

        self.epoch_count = 0
        self.best_epoch = 0

        self.last_best = None
        self.higher_better = higher_better
        self.tolerance = tolerance

    def early_stop_check(self, curr_val):
        if not self.higher_better:
            curr_val *= -1
        if self.last_best is None:
            self.last_best = curr_val
        elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
            self.last_best = curr_val
            self.num_round = 0
            self.best_epoch = self.epoch_count
        else:
            self.num_round += 1
        self.epoch_count += 1
        return self.num_round >= self.max_round


class RandEdgeSampler(object):
    def __init__(self, src_list, dst_list):
        src_list = np.concatenate(src_list)
        dst_list = np.concatenate(dst_list)
        self.src_list = np.unique(src_list)
        self.dst_list = np.unique(dst_list)

    def sample(self, size):
        src_index = np.random.randint(0, len(self.src_list), size)
        dst_index = np.random.randint(0, len(self.dst_list), size)

        return self.src_list[src_index], self.dst_list[dst_index]


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def put(self, key: int, value: int) -> None:
        if key in self.cache:
            # 如果 key 已经存在，更新其值并将其移到末尾
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.capacity:
            # 如果缓存已满，删除最不常使用的元素（OrderedDict 会自动按插入顺序排列）
            self.cache.popitem(last=False)
        # 将新的键值对插入到字典的末尾
        self.cache[key] = value


def process_sampling_numbers(num_neighbors, num_layers):
    num_neighbors = [int(n) for n in num_neighbors]
    if len(num_neighbors) == 1:
        num_neighbors = num_neighbors * num_layers
    else:
        num_layers = len(num_neighbors)
    return num_neighbors, num_layers


def get_embedding(num_embeddings, embedding_dims, zero_init=False, device=None):
    if type(embedding_dims) == int:
        embedding_dims = [embedding_dims]
    if zero_init:
        embed = nn.Parameter(torch.zeros(num_embeddings, *embedding_dims))
        # embed = torch.Tensor(num_embeddings, *embedding_dims)
    else:
        embed = nn.Parameter(torch.Tensor(num_embeddings, *embedding_dims))
        # embed = torch.Tensor(num_embeddings, *embedding_dims)
        nn.init.xavier_uniform_(embed, gain=nn.init.calculate_gain('relu'))
    if device is not None:
        embed = embed.to(device)

    return embed


def load_feature(dataname, device):
    node_feat_fpath = os.path.join(settings.DATA_ROOT, dataname, 'entity_embedding_pro.pt')
    relation_feat_fpath = os.path.join(settings.DATA_ROOT, dataname, 'relation_embedding_pro.pt')
    node_feature = torch.load(node_feat_fpath)
    node_feature = nn.Parameter(node_feature).to(device)
    relation_feat = torch.load(relation_feat_fpath)
    relation_feat = nn.Parameter(relation_feat).to(device)

    return node_feature, relation_feat


def load_temporal_knowledge_graph(dataset_name):
    # if dataset_name in settings.ALL_GRAPHS:
    train_file, val_file, test_file = "train.txt", "valid.txt", "test.txt"
    # else:
    #     raise ValueError(f"Invalid graph name: {dataset_name}")

    column_names = ['head', 'rel', 'tail', 'time', '_']
    train_data_table = load_data_table(dataset_name, train_file, column_names)
    val_data_table = load_data_table(dataset_name, val_file, column_names)
    test_data_table = load_data_table(dataset_name, test_file, column_names)
    all_data_table = pd.concat([train_data_table, val_data_table, test_data_table], ignore_index=True)
    eidx = np.arange(len(all_data_table), dtype=int)

    stat_table = load_data_table(dataset_name, "stat.txt", column_names=['num_entities', 'num_relations', '_'])
    num_entities, num_relations = stat_table['num_entities'].item(), stat_table['num_relations'].item()

    all_heads = all_data_table['head'].to_numpy()
    all_tails = all_data_table['tail'].to_numpy()
    all_rels = all_data_table['rel'].to_numpy()
    all_timestamps = all_data_table['time'].to_numpy()
    # all_timestamps = eidx
    edge_idxs = eidx

    return all_heads, all_tails, all_rels, all_timestamps, edge_idxs, num_entities, num_relations

def load_data_table(graph_name, file_name, column_names=None):
    data_fpath = os.path.join(settings.DATA_ROOT, graph_name, file_name)
    return pd.read_table(data_fpath, sep='\t', names=column_names)
