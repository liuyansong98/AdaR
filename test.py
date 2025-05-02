import numpy as np
import os
import pandas as pd
import torch
from module import myModel
from graph import NeighborFinder
from log import *
from utils import *
from train import *
import heapq
from collections import OrderedDict
from typing import Dict

np.set_printoptions(threshold=1e6)

settings_path = os.path.dirname(__file__)

DATA_ROOT = os.path.join(settings_path, 'data')
CACHE_ROOT = os.path.join(settings_path, 'cache')
RESULT_ROOT = os.path.join(settings_path, 'result')

GRAPH_ICEWS18 = "ICEWS18"
GRAPH_ICEWS14 = "ICEWS14"
GRAPH_ICEWS05_15 = "ICEWS05_15"
GRAPH_GDELT = "GDELT"

ALL_GRAPHS = [GRAPH_ICEWS18, GRAPH_ICEWS14, GRAPH_ICEWS05_15, GRAPH_GDELT]


def load_temporal_knowledge_graph(dataset_name):
    # if dataset_name in ALL_GRAPHS:
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
    _ = all_data_table['_'].to_numpy()
    # all_timestamps = eidx
    edge_idxs = eidx

    return all_heads, all_tails, all_rels, all_timestamps, _, edge_idxs, num_entities, num_relations


def load_data_table(graph_name, file_name, column_names=None):
    data_fpath = os.path.join(DATA_ROOT, graph_name, file_name)
    return pd.read_table(data_fpath, sep='\t', names=column_names)


def statistics():
    heads, tails, rels, timestamps, _, edge_idxs, num_entities, num_relations = \
        load_temporal_knowledge_graph("ICEWS_500")

    full_adj_list = [[] for _ in range(num_entities)]
    for src, dst, eidx, ts in zip(heads, tails, rels, timestamps):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx, ts))
        # full_adj_list[dst].append((src, eidx+num_relations, ts))

    print(max(heads))
    print(min(heads))
    neighbor_count = 0
    neighbor_list = []
    rel_num_list = np.zeros([num_entities, num_relations])
    for i in range(len(full_adj_list)):
        neighbor_list.append(len(full_adj_list[i]))
        rel = [x for _1, x, _2 in full_adj_list[i]]
        for j in range(len(rel)):
            rel_num_list[i][rel[j]] += 1

        if len(full_adj_list[i]) < 8:
            neighbor_count += 1

    neighbor_list = torch.tensor(neighbor_list)
    rel_num_list = torch.tensor(rel_num_list)
    neighbor_list_sorted, neighbor_list_indices = torch.sort(neighbor_list, descending=True)
    rel_num_list_sorted, rel_num_list_indices = torch.sort(rel_num_list, dim=1, descending=True)
    print("邻居数量最多的前20个节点id及邻居数量")
    print(neighbor_list_indices[0:20])
    print(neighbor_list_sorted[0:20])
    print("邻居数量最多的节点的邻接关系统计（数量、节点ID）")
    print(rel_num_list_sorted[neighbor_list_indices[0:20], 0:20])
    print(rel_num_list_indices[neighbor_list_indices[0:20], 0:20])

    print("邻居小于x的节点数量、比例")
    print(neighbor_count)
    print(neighbor_count / num_entities)

    heads_num = np.zeros([num_entities])
    rels_num = np.zeros([num_relations])
    tails_num = np.zeros([num_entities])
    assert len(heads) == len(tails), (len(tails), len(tails))
    for i in range(len(heads)):
        # if heads[i] < 100:
        heads_num[heads[i]] += 1
        # if tails[i] < 100:
        tails_num[tails[i]] += 1

    # print("头部节点数量列表")
    # print(list(heads_num))
    # print("尾部节点数量列表")
    # print(list(tails_num))

    # print(max((heads_num + 10) / (tails_num+10)))
    # print(min((heads_num + 10) / (tails_num+10)))
    # x = (heads_num + 10) / (tails_num+10)
    # x.tolist()
    # x.sort()
    # print("x")
    # print(list(x))

    tails_num = list(tails_num)
    low_num = 0

    for i in range(len(tails_num)):
        if tails_num[i] < 11:
            low_num = low_num + 1

    print("出现次数小于x的节点数量、比例")
    print(low_num)
    print(low_num / num_entities)

    tails_num.sort()
    tails_num.reverse()
    print("尾节点出现频次列表（由大到小）")
    # print(tails_num)
    node_num = 0
    for i in range(len(tails_num) // 2):
        node_num = node_num + tails_num[i]

    print(node_num)
    print(node_num / len(heads))

    grater_x_num = 0
    for i in range(len(tails_num)):
        if tails_num[i] > 1000:
            grater_x_num += tails_num[i]

    print("出现次数大于5的节点出现次数占比")
    print(grater_x_num / len(heads))

    for i in range(len(rels)):
        rels_num[rels[i]] += 1

    rels_num = list(rels_num)
    rels_num.sort()
    rels_num.reverse()
    print(rels_num)


def bisect_left_adapt(a, x):
    lo = 0
    hi = len(a)
    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] <= x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def compute_rep_trip(data_name):
    heads, tails, rels, timestamps, _, edge_idxs, num_entities, num_relations = \
        load_temporal_knowledge_graph(data_name)
    triplets_set = set()
    count = 0
    for i in range(len(heads)):
        tmp_triplet = (heads[i], rels[i], tails[i])
        if tmp_triplet not in triplets_set:
            triplets_set.add(tmp_triplet)
        else:
            count += 1
    print('{}数据集重复发生的事件数：{}\t比例：{}'.format(data_name, count, count / len(heads)))


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def dataset_divide(data_name, old_time_interval, new_time_interval):
    heads, tails, rels, timestamps, _, edge_idxs, num_entities, num_relations = \
        load_temporal_knowledge_graph(data_name)

    for i in range(len(timestamps)):
        timestamps[i] = timestamps[i] // new_time_interval * new_time_interval

    val_time, test_time = list(np.quantile(timestamps, [0.80, 0.90]))
    train_mask = timestamps <= val_time
    test_mask = timestamps > test_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)

    _ = np.zeros(len(heads))
    train_data = np.vstack(
        (heads[train_mask], rels[train_mask], tails[train_mask], timestamps[train_mask], _[train_mask]))

    valid_data = np.vstack((heads[val_mask], rels[val_mask], tails[val_mask], timestamps[val_mask], _[val_mask]))

    test_data = np.vstack((heads[test_mask], rels[test_mask], tails[test_mask], timestamps[test_mask], _[test_mask]))

    data_name = data_name + str(new_time_interval // old_time_interval)
    data_dir_fpath = os.path.join(DATA_ROOT, data_name)
    mkdirs(data_dir_fpath)

    data_fpath = os.path.join(DATA_ROOT, data_name, "train.txt")
    np.savetxt(data_fpath, train_data.T, fmt='%d', delimiter="\t")

    data_fpath = os.path.join(DATA_ROOT, data_name, "valid.txt")
    np.savetxt(data_fpath, valid_data.T, fmt='%d', delimiter="\t")

    data_fpath = os.path.join(DATA_ROOT, data_name, "test.txt")
    np.savetxt(data_fpath, test_data.T, fmt='%d', delimiter="\t")

    print('{} divided finished'.format(data_name))


def eval(model_path):
    args, sys_argv = get_args()
    print(args)
    set_random_seed(args.seed)

    device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')
    args.device = device

    heads, tails, rels, timestamps, _, edge_idxs, num_entities, num_relations = \
        load_temporal_knowledge_graph(args.data)

    all_nodes_l = np.arange(0, num_entities)

    val_time, test_time, end_time = list(np.quantile(timestamps, [0.8, 0.9, 1]))

    logger = logging.getLogger()

    # if args.mode == 't':
    logger.info('Transductive evaluating...')
    logger.info(model_path)
    valid_train_flag = (timestamps <= val_time)
    valid_val_flag = (timestamps <= test_time) * (timestamps > val_time)
    valid_test_flag = (timestamps > test_time) * (timestamps <= end_time)
    full_data_flag = timestamps <= end_time

    train_src_l, train_dst_l, train_ts_l, train_e_idx_l = \
        heads[valid_train_flag], tails[valid_train_flag], timestamps[valid_train_flag], \
            rels[valid_train_flag]

    val_src_l, val_dst_l, val_ts_l, val_e_idx_l = \
        heads[valid_val_flag], tails[valid_val_flag], timestamps[valid_val_flag], \
            rels[valid_val_flag]

    test_src_l, test_dst_l, test_ts_l, test_e_idx_l = \
        heads[valid_test_flag], tails[valid_test_flag], timestamps[valid_test_flag], \
            rels[valid_test_flag]

    heads_, tails_, timestamps_, rels_ = \
        heads[full_data_flag], tails[full_data_flag], timestamps[full_data_flag], \
            rels[full_data_flag]

    val_ts_l_discr = []
    val_bs_idx = [0]
    if "social" in args.data and args.data != "social_TKG_cate_level1_filter_discr1h":
        for i in range(len(val_ts_l)):
            val_ts_l_discr.append(val_ts_l[i] // 3600)
        for k in range(0, len(val_ts_l_discr) - 1):
            if val_ts_l_discr[k] != val_ts_l_discr[k + 1]:
                val_bs_idx.append(k + 1)
        val_bs_idx.append(len(val_ts_l_discr))
    else:
        for k in range(0, len(val_ts_l) - 1):
            if val_ts_l[k] != val_ts_l[k + 1]:
                val_bs_idx.append(k + 1)
        val_bs_idx.append(len(val_ts_l))

    test_ts_l_discr = []
    test_bs_idx = [0]
    if "social" in args.data and args.data != "social_TKG_cate_level1_filter_discr1h":
        for i in range(len(test_ts_l)):
            test_ts_l_discr.append(test_ts_l[i] // 3600)
        for k in range(0, len(test_ts_l_discr) - 1):
            if test_ts_l_discr[k] != test_ts_l_discr[k + 1]:
                test_bs_idx.append(k + 1)
        test_bs_idx.append(len(test_ts_l_discr))
    else:
        for k in range(0, len(test_ts_l) - 1):
            if test_ts_l[k] != test_ts_l[k + 1]:
                test_bs_idx.append(k + 1)
        test_bs_idx.append(len(test_ts_l))

    train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l
    val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l
    train_val_data = (train_data, val_data)

    history_entity_dic: Dict[(int, int), LRUCache] = {}  # 定义一个空字典(s,r)->LRUcache
    history_flex_cap = [0 for _ in range(num_entities * num_relations)]
    history_flex_cap_no_Rep = [0 for _ in range(num_entities * num_relations)]
    train_triple_list = []
    for i in range(len(train_src_l)):
        train_triple_list.append((train_src_l[i], train_e_idx_l[i], train_dst_l[i]))  # 三元组
    train_triple_set = set(train_triple_list)
    for i in range(len(train_triple_set)):  # 不计算重复出现的三元组
        history_flex_cap_no_Rep[train_src_l[i] * num_relations + train_e_idx_l[i]] += 1
    for i in range(len(train_src_l)):
        history_flex_cap[train_src_l[i] * num_relations + train_e_idx_l[i]] += 1

    # history_cap = np.floor(np.array(history_flex_cap_no_Rep) * args.flexible_capacity + args.base_capacity)
    history_cap = np.floor(np.array(history_flex_cap) * args.flexible_capacity + args.base_capacity)
    for i in range(num_entities):
        for j in range(num_relations):
            if history_cap[i * num_relations + j] == 0:
                continue
            else:
                history_entity_dic[(i, j)] = LRUCache(history_cap[i * num_relations + j])

    full_adj_list = [[] for _ in range(num_entities)]
    for src, dst, eidx, ts in zip(heads_, tails_, rels_, timestamps_):
        full_adj_list[src].append((dst, eidx, ts))
        # full_adj_list[dst].append((src, eidx, ts))
        full_adj_list[dst].append((src, eidx + num_relations, ts))

    full_ngh_finder = NeighborFinder(full_adj_list, temporal_bias=args.temporal_bias,
                                     limit_ngh_span=args.limit_ngh_span, ngh_span=args.ngh_span,
                                     num_entity=num_entities, data_name=args.data)

    partial_adj_list = [[] for _ in range(num_entities)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        partial_adj_list[src].append((dst, eidx, ts))
        # partial_adj_list[dst].append((src, eidx, ts))
        partial_adj_list[dst].append((src, eidx + num_relations, ts))
    for src, dst, eidx, ts in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l):
        partial_adj_list[src].append((dst, eidx, ts))
        # partial_adj_list[dst].append((src, eidx, ts))
        partial_adj_list[dst].append((src, eidx + num_relations, ts))

    partial_ngh_finder = NeighborFinder(partial_adj_list, temporal_bias=args.temporal_bias,
                                        limit_ngh_span=args.limit_ngh_span, ngh_span=args.ngh_span,
                                        num_entity=num_entities, data_name=args.data)

    logger.info('Sampling module - temporal bias: {}, score_weight:{} '.format(
        args.temporal_bias, args.default_weight))

    train_rand_sampler = RandEdgeSampler((train_src_l,), (train_dst_l,))
    val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_dst_l, val_dst_l))
    test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_dst_l, val_dst_l, test_dst_l))
    rand_samplers = train_rand_sampler, val_rand_sampler

    """init dynamic entity embeddings"""
    init_dynamic_entity_embeds = get_embedding(num_entities, args.embed_dim, zero_init=False)

    """init relation embeddings"""
    init_dynamic_relation_embeds = get_embedding(num_relations * 2, args.embed_dim, zero_init=False)

    ''' load node and relation feature '''
    # init_dynamic_entity_embeds, init_dynamic_relation_embeds = load_feature(args.data, args.device)

    model = myModel(n_feat=init_dynamic_entity_embeds, e_feat=init_dynamic_relation_embeds, device=args.device,
                    pos_dim=args.pos_dim, num_layers=args.n_layer,
                    solver=args.solver, step_size=args.step_size, drop_out=args.drop_out,
                    n_head=args.n_head, bs=args.bs, path_encode=args.path_encode)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.update_ngh_finder(full_ngh_finder)
    model.to(device)
    row = train_src_l * num_relations + train_e_idx_l
    col = train_dst_l
    d1 = np.ones(len(row))
    all_tail_seq = sp.csr_matrix((d1, (row, col)), shape=(num_entities * num_relations, num_entities))
    for i in range(len(train_src_l)):
        history_entity_dic[train_src_l[i], train_e_idx_l[i]].put(train_dst_l[i], train_ts_l[i])
    # 自动选择最优score_weight
    best_score_weight = -1
    if args.find_score_weight:
        mrr, best_score_weight = eval_one_epoch(model, all_nodes_l, test_rand_sampler, val_src_l, val_dst_l, val_ts_l,
                                                val_e_idx_l,
                                                val_bs_idx, args, logger, all_tail_seq, history_entity_dic,
                                                num_entities, num_relations,
                                                epoch=200, is_need_filter=False, find_score_weight=True,
                                                default_weight=args.default_weight,
                                                stage='val')

    row = val_src_l * num_relations + val_e_idx_l
    col = val_dst_l
    d1 = np.ones(len(row))
    val_tail_seq = sp.csr_matrix((d1, (row, col)), shape=(num_entities * num_relations, num_entities))
    all_tail_seq = all_tail_seq + val_tail_seq
    # 更新历史记录
    for i in range(len(val_src_l)):
        history_entity_dic[val_src_l[i], val_e_idx_l[i]].put(val_dst_l[i], val_ts_l[i])
    model.ngh_finder.init_node_degree()
    model.ngh_finder.update_node_degree(train_src_l, train_dst_l)
    model.ngh_finder.update_node_degree(val_src_l, val_dst_l)

    if best_score_weight != -1:
        args.default_weight = best_score_weight
    if "social" in args.data:
        args.bs = 1
    logger.info('Start testing...')
    mrr, best_score_weight = eval_one_epoch(model, all_nodes_l, test_rand_sampler, test_src_l, test_dst_l, test_ts_l,
                                            test_e_idx_l,
                                            test_bs_idx, args, logger, all_tail_seq, history_entity_dic, num_entities,
                                            num_relations,
                                            epoch=200, is_need_filter=True, default_weight=args.default_weight,
                                            case_study=args.case_study, model_path=model_path, stage='test')


# 计算整个数据集的邻居数量和邻居时间间隔统计信息
def calculate_ngh(data_name):
    print(f"\ndata_name: {data_name}, {datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')}")
    heads, tails, rels, timestamps, _, edge_idxs, num_entities, num_relations = \
        load_temporal_knowledge_graph(data_name)

    full_adj_list = [[] for _ in range(num_entities)]
    for src, dst, eidx, ts in zip(heads, tails, rels, timestamps):
        full_adj_list[src].append((dst, eidx, ts))
        # full_adj_list[dst].append((src, eidx, ts))
        full_adj_list[dst].append((src, eidx + num_relations, ts))
    full_ngh_finder = NeighborFinder(full_adj_list, temporal_bias=1,
                                     num_entity=num_entities)
    ngh_delta_time_list = []
    ngh_30_delta_time_list = []
    ngh_num_list = []
    for i in range(len(heads)):
        ngh_idx, ngh_eidx, ngh_ts, ngh_binomial_prob = full_ngh_finder.find_before(heads[i], timestamps[i] - 1,
                                                                                   e_idx=None)

        if len(ngh_idx) == 0:
            continue
        if len(ngh_idx) >= 30:
            delta_t_30 = timestamps[i] - ngh_ts
            ngh_30_delta_time_list.append(np.mean(delta_t_30[np.argsort(delta_t_30)[:30]]))
        else:
            delta_t_30 = timestamps[i] - ngh_ts
            ngh_30_delta_time_list.append(np.mean(delta_t_30))
        delta_t_ = timestamps[i] - ngh_ts
        ngh_delta_time_list.append(np.mean(delta_t_))
        ngh_num_list.append(len(ngh_eidx))

    print("avg ngh num: ", np.mean(np.array(ngh_num_list)))
    print("avg ngh delta time: ", np.mean(np.array(ngh_delta_time_list)))
    print("avg ngh 30 delta time: ", np.mean(np.array(ngh_30_delta_time_list)))


# 计算数据集中各节点的邻居数量和邻居时间间隔统计信息
def calculate_ngh_by_node(data_name):
    print(f"\n数据集: {data_name}, {datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')}")
    heads, tails, rels, timestamps, _, edge_idxs, num_entities, num_relations = \
        load_temporal_knowledge_graph(data_name)

    ngh_time_delta_avg = [[] for _ in range(num_entities)]
    ngh30_time_delta_avg = [[] for _ in range(num_entities)]

    full_adj_list = [[] for _ in range(num_entities)]
    for src, dst, eidx, ts in zip(heads, tails, rels, timestamps):
        full_adj_list[src].append((dst, eidx, ts))
        # full_adj_list[dst].append((src, eidx, ts))
        full_adj_list[dst].append((src, eidx + num_relations, ts))
    full_ngh_finder = NeighborFinder(full_adj_list, temporal_bias=1,
                                     num_entity=num_entities)
    # ngh_delta_time_list = []
    # ngh_30_delta_time_list = []
    ngh_num_list = []
    for i in range(len(heads)):
        ngh_idx, ngh_eidx, ngh_ts, ngh_binomial_prob = full_ngh_finder.find_before(heads[i], timestamps[i] - 1,
                                                                                   e_idx=None)

        ngh_idx_tails, ngh_eidx_tails, ngh_ts_tails, ngh_binomial_prob_tails = full_ngh_finder.find_before(tails[i], timestamps[i] - 1,
                                                                                   e_idx=None)

        if len(ngh_idx) == 0 and len(ngh_idx_tails) == 0:
            continue
        else:
            if len(ngh_idx) != 0:
                if len(ngh_idx) >= 30:
                    delta_t_30 = timestamps[i] - ngh_ts
                    ngh30_time_delta_avg[heads[i]].append(np.mean(delta_t_30[np.argsort(delta_t_30)[:30]]))
                else:
                    delta_t_30 = timestamps[i] - ngh_ts
                    ngh30_time_delta_avg[heads[i]].append(np.mean(delta_t_30))
                delta_t_ = timestamps[i] - ngh_ts
                ngh_time_delta_avg[heads[i]].append(np.mean(delta_t_))

            if len(ngh_idx_tails) != 0:
                if len(ngh_idx_tails) >= 30:
                    delta_t_30_ = timestamps[i] - ngh_ts_tails
                    ngh30_time_delta_avg[tails[i]].append(np.mean(delta_t_30_[np.argsort(delta_t_30_)[:30]]))
                else:
                    delta_t_30_ = timestamps[i] - ngh_ts_tails
                    ngh30_time_delta_avg[tails[i]].append(np.mean(delta_t_30_))
                delta_t = timestamps[i] - ngh_ts_tails
                ngh_time_delta_avg[tails[i]].append(np.mean(delta_t))
            ngh_num_list.append(len(ngh_eidx) * 2)

    ngh_time_delta_avg_list = []
    ngh30_time_delta_avg_list = []
    for i in range(num_entities):
        x = len(ngh_time_delta_avg[i])
        if x != 0:
            ngh_time_delta_avg_list.append(np.mean(np.array(ngh_time_delta_avg[i])))
            ngh30_time_delta_avg_list.append(np.mean(np.array(ngh30_time_delta_avg[i])))
        # else:
        #     ngh_time_delta_avg_list.append(float("inf"))
        #     ngh30_time_delta_avg_list.append(float("inf"))

    print("avg ngh num: ", np.mean(np.array(ngh_num_list))/2)
    # 获取排序后的索引
    sorted_indices_avg = np.argsort(-np.array(ngh_time_delta_avg_list))
    sorted_avg = np.array(ngh_time_delta_avg_list)[sorted_indices_avg]
    sorted_indices_avg30 = np.argsort(-np.array(ngh30_time_delta_avg_list))
    sorted_avg30 = np.array(ngh30_time_delta_avg_list)[sorted_indices_avg30]
    print("avg ngh delta time")
    for i in range(len(sorted_avg)):
        # print(sorted_indices_avg[i], sorted_avg[i])
        print("{:.2f}".format(sorted_avg[i]))

    # print("\navg ngh 30 delta time")
    # for i in range(num_entities):
    #     print(sorted_avg30[i])

# 计算测试集时间间隔分类上下界（邻居节点时间间隔最密集的一部分和最稀疏的一部分）
def calculate_up_low_boundary(data_name):

    train_file, val_file, test_file = "train.txt", "valid.txt", "test.txt"
    column_names = ['head', 'rel', 'tail', 'time', '_']
    print(f"\ndata_name: {data_name}, {datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')}")
    heads, tails, rels, timestamps, _, edge_idxs, num_entities, num_relations = \
        load_temporal_knowledge_graph(data_name)
    test_data_table = load_data_table(data_name, test_file, column_names)
    test_heads = test_data_table['head'].to_numpy()
    test_tails = test_data_table['tail'].to_numpy()
    test_rels = test_data_table['rel'].to_numpy()
    test_timestamps = test_data_table['time'].to_numpy()

    ngh_time_delta_avg = [[] for _ in range(num_entities)]  # 保存每次的邻居平均间隔
    ngh_delta_time = [0 for _ in range(num_entities)]   # 保存每个实体的类型的平均时间间隔
    ngh30_time_delta_avg = [[] for _ in range(num_entities)]    # 保存每次的近30邻居的平均间隔
    ngh30_delta_time = [0 for _ in range(num_entities)]     # 保存每个实体的类型的近20邻居平均时间间隔

    full_adj_list = [[] for _ in range(num_entities)]
    for src, dst, eidx, ts in zip(heads, tails, rels, timestamps):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx + num_relations, ts))
    full_ngh_finder = NeighborFinder(full_adj_list, temporal_bias=1,
                                     num_entity=num_entities)

    ngh_num_list = []
    for i in range(len(heads)):
        ngh_idx, ngh_eidx, ngh_ts, ngh_binomial_prob = full_ngh_finder.find_before(heads[i], timestamps[i] - 1,
                                                                                   e_idx=None)
        ngh_idx_tails, ngh_eidx_tails, ngh_ts_tails, ngh_binomial_prob_tails = full_ngh_finder.find_before(tails[i], timestamps[i] - 1,
                                                                                   e_idx=None)

        if len(ngh_idx) == 0 and len(ngh_idx_tails) == 0:
            continue
        else:
            if len(ngh_idx) != 0:
                if len(ngh_idx) >= 30:
                    delta_t_30 = timestamps[i] - ngh_ts
                    ngh30_time_delta_avg[heads[i]].append(np.mean(delta_t_30[np.argsort(delta_t_30)[:30]]))
                else:
                    delta_t_30 = timestamps[i] - ngh_ts
                    ngh30_time_delta_avg[heads[i]].append(np.mean(delta_t_30))
                delta_t_ = timestamps[i] - ngh_ts
                ngh_time_delta_avg[heads[i]].append(np.mean(delta_t_))

            if len(ngh_idx_tails) != 0:
                if len(ngh_idx_tails) >= 30:
                    delta_t_30_ = timestamps[i] - ngh_ts_tails
                    ngh30_time_delta_avg[tails[i]].append(np.mean(delta_t_30_[np.argsort(delta_t_30_)[:30]]))
                else:
                    delta_t_30_ = timestamps[i] - ngh_ts_tails
                    ngh30_time_delta_avg[tails[i]].append(np.mean(delta_t_30_))
                delta_t = timestamps[i] - ngh_ts_tails
                ngh_time_delta_avg[tails[i]].append(np.mean(delta_t))
            ngh_num_list.append(len(ngh_eidx) * 2)

    for i in range(len(ngh_time_delta_avg)):
        if len(ngh_time_delta_avg[i]) != 0:
            ngh_delta_time[i] = np.mean(np.array(ngh_time_delta_avg[i]))
            ngh30_delta_time[i] = np.mean(np.array(ngh30_time_delta_avg[i]))

    up_bound_facts_30 = 0
    low_bound_facts_30 = 0
    test_data_entity_num_list = [0 for _ in range(num_entities)]
    for i in range(len(test_heads)):
        test_data_entity_num_list[test_heads[i]] += 1

    ngh_time_delta_avg_list = []
    ngh30_time_delta_avg_list = []
    for i in range(num_entities):
        x = len(ngh_time_delta_avg[i])
        if x != 0:
            ngh_time_delta_avg_list.append(np.mean(np.array(ngh_time_delta_avg[i])))
            ngh30_time_delta_avg_list.append(np.mean(np.array(ngh30_time_delta_avg[i])))
        else:
            ngh_time_delta_avg_list.append(-1.0)
            ngh30_time_delta_avg_list.append(-1.0)
    # 获取排序后的索引
    sorted_indices_avg = np.argsort(-np.array(ngh_time_delta_avg_list))
    sorted_avg = np.array(ngh_time_delta_avg_list)[sorted_indices_avg]
    sorted_indices_avg30 = np.argsort(-np.array(ngh30_time_delta_avg_list))
    sorted_avg30 = np.array(ngh30_time_delta_avg_list)[sorted_indices_avg30]

    accumulated_up_num = 0
    accumulated_low_num = 0
    up_threshold = 0
    low_threshold = 0
    for i in range(len(sorted_avg30)):
        accumulated_up_num += test_data_entity_num_list[sorted_indices_avg30[i]]
        if accumulated_up_num > len(test_heads) * 0.3:
            up_threshold = sorted_avg30[i]
            break

    for i in range(len(sorted_avg30)-1, -1, -1):
        if sorted_avg30[i] > 0:
            accumulated_low_num += test_data_entity_num_list[sorted_indices_avg30[i]]
            if accumulated_low_num > len(test_heads) * 0.3:
                low_threshold = sorted_avg30[i]
                break

    print(f"数据集：{data_name}。上界阈值：{up_threshold}， 下界阈值：{low_threshold}")

def save_test_dense_sparse_split(data_name: str):
    data_name_2_up_bound_time = {"ICEWS14s_divide": 318, "ICEWS05_15_divide": 161, "GDELTsmall_divide": 1094,
                                 "ICEWS18_divide": 1754, "socialTKG": 122}
    data_name_2_low_bound_time = {"ICEWS14s_divide": 248, "ICEWS05_15_divide": 127, "GDELTsmall_divide": 876,
                                  "ICEWS18_divide": 1589, "socialTKG": 81, }
    data_name_2_up_bound_ngh = {"ICEWS14s_divide": 349, "ICEWS05_15_divide": 2866, "GDELTsmall_divide": 4058,
                                "ICEWS18_divide": 1019, "socialTKG": 53}
    data_name_2_low_bound_ngh = {"ICEWS14s_divide": 38, "ICEWS05_15_divide": 198, "GDELTsmall_divide": 621,
                                 "ICEWS18_divide": 87, "socialTKG": 14}
    data_name_2_up_bound_time_interval = {"ICEWS14s_divide": 54, "ICEWS05_15_divide": 2730, "GDELTsmall_divide": 196,
                                          "ICEWS18_divide": 593, "social_TKG_cate_level1_filter": 746468}
    data_name_2_low_bound_time_interval = {"ICEWS14s_divide": 9, "ICEWS05_15_divide": 272, "GDELTsmall_divide": 38,
                                           "ICEWS18_divide": 68, "socialTKG": 310210}

    train_file, val_file, test_file = "train.txt", "valid.txt", "test.txt"
    column_names = ['head', 'rel', 'tail', 'time', '_']
    print(f"\n数据集: {data_name}, {datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')}")
    train_data_table = load_data_table(data_name, train_file, column_names)
    val_data_table = load_data_table(data_name, val_file, column_names)
    test_data_table = load_data_table(data_name, test_file, column_names)
    print(f"训练集、验证集、测试集长度：{len(train_data_table)}, {len(val_data_table)}, {len(test_data_table)}")
    all_data_table = pd.concat([train_data_table, val_data_table, test_data_table], ignore_index=True)
    eidx = np.arange(len(all_data_table), dtype=int)

    stat_table = load_data_table(data_name, "stat.txt", column_names=['num_entities', 'num_relations', '_'])
    num_entities, num_relations = stat_table['num_entities'].item(), stat_table['num_relations'].item()

    heads = all_data_table['head'].to_numpy()
    tails = all_data_table['tail'].to_numpy()
    rels = all_data_table['rel'].to_numpy()
    timestamps = all_data_table['time'].to_numpy()
    test_heads = test_data_table['head'].to_numpy()
    test_tails = test_data_table['tail'].to_numpy()
    test_rels = test_data_table['rel'].to_numpy()
    test_timestamps = test_data_table['time'].to_numpy()

    # 计算节点邻居数量分布
    entity_2_nghNum = np.zeros(num_entities, int)
    for i in range(len(heads)):
        entity_2_nghNum[heads[i]] += 1
    print(f"平均邻居数量：{np.mean(entity_2_nghNum)}")
    timestamps_discr = []
    if "social" in data_name:
        for i in range(len(test_timestamps)):
            timestamps_discr.append(test_timestamps[i] // 3600)
        test_timestamps = np.array(timestamps_discr)
    _ = test_data_table['_'].to_numpy()
    timestamps_2_num = {}
    for i in range(len(test_timestamps)):
        timestamps_2_num[test_timestamps[i]] = timestamps_2_num.get(test_timestamps[i], 0) + 1

    ngh_time_delta_avg = [[] for _ in range(num_entities)]
    entity_2_ngh_time_interval = [0 for _ in range(num_entities)]
    ngh30_time_delta_avg = [[] for _ in range(num_entities)]
    entity_2_ngh30_time_interval = [0 for _ in range(num_entities)]

    full_adj_list = [[] for _ in range(num_entities)]
    for src, dst, eidx, ts in zip(heads, tails, rels, timestamps):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx + num_relations, ts))
    full_ngh_finder = NeighborFinder(full_adj_list, temporal_bias=1,
                                     num_entity=num_entities)

    ngh_num_list = []
    for i in range(len(heads)):
        ngh_idx, ngh_eidx, ngh_ts, ngh_binomial_prob = full_ngh_finder.find_before(heads[i], timestamps[i] - 1,
                                                                                   e_idx=None)
        ngh_idx_tails, ngh_eidx_tails, ngh_ts_tails, ngh_binomial_prob_tails = full_ngh_finder.find_before(tails[i], timestamps[i] - 1,
                                                                                   e_idx=None)

        if len(ngh_idx) == 0 and len(ngh_idx_tails) == 0:
            continue
        else:
            if len(ngh_idx) != 0:
                if len(ngh_idx) >= 30:
                    delta_t_30 = timestamps[i] - ngh_ts
                    ngh30_time_delta_avg[heads[i]].append(np.mean(delta_t_30[np.argsort(delta_t_30)[:30]]))
                else:
                    delta_t_30 = timestamps[i] - ngh_ts
                    ngh30_time_delta_avg[heads[i]].append(np.mean(delta_t_30))
                delta_t_ = timestamps[i] - ngh_ts
                ngh_time_delta_avg[heads[i]].append(np.mean(delta_t_))

            if len(ngh_idx_tails) != 0:
                if len(ngh_idx_tails) >= 30:
                    delta_t_30_ = timestamps[i] - ngh_ts_tails
                    ngh30_time_delta_avg[tails[i]].append(np.mean(delta_t_30_[np.argsort(delta_t_30_)[:30]]))
                else:
                    delta_t_30_ = timestamps[i] - ngh_ts_tails
                    ngh30_time_delta_avg[tails[i]].append(np.mean(delta_t_30_))
                delta_t = timestamps[i] - ngh_ts_tails
                ngh_time_delta_avg[tails[i]].append(np.mean(delta_t))
            ngh_num_list.append(len(ngh_eidx) * 2)

    for i in range(len(ngh_time_delta_avg)):
        if len(ngh_time_delta_avg[i]) != 0:
            entity_2_ngh_time_interval[i] = np.mean(np.array(ngh_time_delta_avg[i]))
            entity_2_ngh30_time_interval[i] = np.mean(np.array(ngh30_time_delta_avg[i]))

    time_dense_sparse_index = []
    time_interval_dense_sparse_index = []
    ngh_dense_sparse_index = []
    for i in range(len(test_timestamps)):
        if timestamps_2_num[test_timestamps[i]] > data_name_2_up_bound_time[data_name]:
            time_dense_sparse_index.append(1)
        elif timestamps_2_num[test_timestamps[i]] < data_name_2_low_bound_time[data_name]:
            time_dense_sparse_index.append(-1)
        else:
            time_dense_sparse_index.append(0)
        if entity_2_nghNum[test_heads[i]] > data_name_2_up_bound_ngh[data_name]:
            ngh_dense_sparse_index.append(1)
        elif entity_2_nghNum[test_heads[i]] < data_name_2_low_bound_ngh[data_name]:
            ngh_dense_sparse_index.append(-1)
        else:
            ngh_dense_sparse_index.append(0)
        if entity_2_ngh30_time_interval[test_heads[i]] > data_name_2_up_bound_time_interval[data_name]:
            time_interval_dense_sparse_index.append(1)
        elif entity_2_ngh30_time_interval[test_heads[i]] < data_name_2_low_bound_time_interval[data_name]:
            time_interval_dense_sparse_index.append(-1)
        else:
            time_interval_dense_sparse_index.append(0)

    # 验证是否计算正确。两个值应该都是0.3
    print(f"验证snapshot稀疏。{time_dense_sparse_index.count(1)/len(time_dense_sparse_index)}, {time_dense_sparse_index.count(-1)/len(time_dense_sparse_index)}")
    print(
        f"验证邻居稀疏。{ngh_dense_sparse_index.count(1) / len(ngh_dense_sparse_index)}, {ngh_dense_sparse_index.count(-1) / len(ngh_dense_sparse_index)}")
    print(
        f"验证时间间隔稀疏。{time_interval_dense_sparse_index.count(1) / len(time_interval_dense_sparse_index)}, {time_interval_dense_sparse_index.count(-1) / len(time_interval_dense_sparse_index)}")

    data_fpath_time = os.path.join(DATA_ROOT, data_name, "dense_sparse_index_time.npy")
    data_fpath_ngh = os.path.join(DATA_ROOT, data_name, "dense_sparse_index_ngh.npy")
    data_fpath_time_interval = os.path.join(DATA_ROOT, data_name, "dense_sparse_index_time_interval.npy")
    np.save(data_fpath_time, np.array(time_dense_sparse_index))
    np.save(data_fpath_ngh, np.array(ngh_dense_sparse_index))
    np.save(data_fpath_time_interval, np.array(time_interval_dense_sparse_index))
    print(f"time dense_sparse_index length: {len(time_dense_sparse_index)}")
    print(f"ngh dense_sparse_index length: {len(ngh_dense_sparse_index)}")
    print(f"ngh dense_sparse_index length: {len(time_interval_dense_sparse_index)}")
    print("success saved dense_sparse_index!")

if __name__ == "__main__":
    # eval("./saved_checkpoints/1743689121.421062-social_TKG_cate_level1_filter-t-3-64k16k4-60/best_checkpoint.pth")
    eval("./best_models/1744761972.0329447-social_TKG_cate_level1_filter-t-3-64k16k4-60/best-model.pth")
