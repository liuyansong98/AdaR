import numpy as np
import pandas as pd
from log import *
from utils import *
from train import *
from module import myModel
from graph import NeighborFinder
import resource
from typing import Dict


def main():
    args, sys_argv = get_args()

    set_random_seed(args.seed)

    device = torch.device('cuda:{}'.format(args.gpu)) if torch.cuda.is_available() else torch.device('cpu')
    args.device = device

    heads, tails, rels, timestamps, edge_idxs, num_entities, num_relations = \
        load_temporal_knowledge_graph(args.data)

    all_nodes_l = np.arange(0, num_entities)

    logger, get_checkpoint_path, best_model_path = set_up_logger(args, sys_argv)

    timestamps_discr = []
    if "social" in args.data and args.data != "social_TKG_cate_level1_filter_discr1h":
        for i in range(len(timestamps)):
            timestamps_discr.append(timestamps[i] // 3600)
        val_time, test_time, end_time = list(np.quantile(timestamps_discr, [0.8, 0.9, 1]))
        valid_train_flag = (timestamps_discr <= val_time)
        valid_val_flag = (timestamps_discr <= test_time) * (timestamps_discr > val_time)
        valid_test_flag = (timestamps_discr > test_time) * (timestamps_discr <= end_time)
        full_data_flag = timestamps_discr <= end_time
    else:
        val_time, test_time, end_time = list(np.quantile(timestamps, [0.8, 0.9, 1]))
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

    full_adj_list = [[] for _ in range(num_entities)]
    for src, dst, eidx, ts in zip(heads_, tails_, rels_, timestamps_):
        full_adj_list[src].append((dst, eidx, ts))
        full_adj_list[dst].append((src, eidx + num_relations, ts))

    full_ngh_finder = NeighborFinder(full_adj_list, temporal_bias=args.temporal_bias,
                                     limit_ngh_span=args.limit_ngh_span, ngh_span=args.ngh_span,
                                     num_entity=num_entities, data_name=args.data)

    partial_adj_list = [[] for _ in range(num_entities)]
    for src, dst, eidx, ts in zip(train_src_l, train_dst_l, train_e_idx_l, train_ts_l):
        partial_adj_list[src].append((dst, eidx, ts))
        partial_adj_list[dst].append((src, eidx + num_relations, ts))
    for src, dst, eidx, ts in zip(val_src_l, val_dst_l, val_e_idx_l, val_ts_l):
        partial_adj_list[src].append((dst, eidx, ts))
        partial_adj_list[dst].append((src, eidx + num_relations, ts))

    partial_ngh_finder = NeighborFinder(partial_adj_list, temporal_bias=args.temporal_bias,
                                        limit_ngh_span=args.limit_ngh_span, ngh_span=args.ngh_span,
                                        num_entity=num_entities, data_name=args.data)

    ngh_finders = partial_ngh_finder, full_ngh_finder
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

    model = myModel(n_feat=init_dynamic_entity_embeds, e_feat=init_dynamic_relation_embeds, device=args.device,
                    pos_dim=args.pos_dim, num_layers=args.n_layer,
                    solver=args.solver, step_size=args.step_size, drop_out=args.drop_out,
                    get_checkpoint_path=get_checkpoint_path,
                    n_head=args.n_head, bs=args.bs, path_encode=args.path_encode).to(device)

    # 加载,继续训练
    # model.load_state_dict(torch.load("./saved_checkpoints/1744786925.109515-ICEWS18_divide-t-3-64k16k4-60/best_checkpoint.pth", map_location=device))
    # model.update_ngh_finder(full_ngh_finder)
    # model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stopper = EarlyStopMonitor(tolerance=args.tolerance)

    history_entity_dic: Dict[(int, int), LRUCache] = {}  # (s,r)->LRUcache
    history_flex_cap = [0 for _ in range(num_entities * num_relations)]
    history_flex_cap_no_Rep = [0 for _ in range(num_entities * num_relations)]
    train_triple_list = []
    for i in range(len(train_src_l)):
        train_triple_list.append((train_src_l[i], train_e_idx_l[i], train_dst_l[i]))  #
    train_triple_set = set(train_triple_list)
    for i in range(len(train_triple_set)):
        history_flex_cap_no_Rep[train_src_l[i] * num_relations + train_e_idx_l[i]] += 1
    for i in range(len(train_src_l)):
        history_flex_cap[train_src_l[i] * num_relations + train_e_idx_l[i]] += 1

    history_cap = np.floor(np.array(history_flex_cap_no_Rep) * args.flexible_capacity + args.base_capacity)
    # history_cap = np.floor(np.array(history_flex_cap) * args.flexible_capacity + args.base_capacity).astype(int)

    for i in range(num_entities):
        for j in range(num_relations):
            if history_cap[i * num_relations + j] == 0:
                continue
            else:
                history_entity_dic[(i, j)] = LRUCache(history_cap[i * num_relations + j])

    train_val(train_val_data, all_nodes_l, model, history_entity_dic, args, optimizer,
              early_stopper, ngh_finders, rand_samplers, logger, num_entities, num_relations)

    row = train_src_l * num_relations + train_e_idx_l
    col = train_dst_l
    d1 = np.ones(len(row))
    all_tail_seq = sp.csr_matrix((d1, (row, col)), shape=(num_entities * num_relations, num_entities))
    model.ngh_finder.init_node_degree()
    model.ngh_finder.update_node_degree(train_src_l, train_dst_l)

    # 更新历史记录
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

    # 更新历史记录
    for i in range(len(val_src_l)):
        history_entity_dic[val_src_l[i], val_e_idx_l[i]].put(val_dst_l[i], val_ts_l[i])

    row = val_src_l * num_relations + val_e_idx_l
    col = val_dst_l
    d1 = np.ones(len(row))
    val_tail_seq = sp.csr_matrix((d1, (row, col)), shape=(num_entities * num_relations, num_entities))
    all_tail_seq = all_tail_seq + val_tail_seq
    model.ngh_finder.init_node_degree()
    model.ngh_finder.update_node_degree(train_src_l, train_dst_l)
    model.ngh_finder.update_node_degree(val_src_l, val_dst_l)
    model.update_ngh_finder(full_ngh_finder)
    if best_score_weight != -1:
        args.default_weight = best_score_weight
    if "social" in args.data:
        args.bs = 1
    logger.info('Start testing...')
    mrr, best_score_weight = eval_one_epoch(model, all_nodes_l, test_rand_sampler, test_src_l, test_dst_l, test_ts_l,
                                            test_e_idx_l,
                                            test_bs_idx, args, logger, all_tail_seq, history_entity_dic, num_entities,
                                            num_relations,
                                            epoch=200, is_need_filter=True, find_score_weight=False,
                                            default_weight=args.default_weight,
                                            stage='test')

    logger.info('Saving model...')
    torch.save(model.state_dict(), best_model_path)
    logger.info('Saved model to {}'.format(best_model_path))
    logger.info('model saved')

if __name__ == "__main__":
    main()
