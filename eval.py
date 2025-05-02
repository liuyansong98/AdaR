import copy

import math
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import scipy.sparse as sp
import os
import heapq
import settings as settings

TEST_BATCH_SIZE = 32


def eval_one_epoch(model, all_nodes_l, sampler, src, dst, ts, e_idx_l, val_bs_idx, args, logger,
                   all_tail_seq_obj, history_entity_dic, num_e, num_r, epoch, is_need_filter=False, find_score_weight=False,
                   default_weight=0.5, case_study=False, model_path="", stage='test'):

    if case_study:
        settings_path = os.path.dirname(__file__)
        DATA_ROOT = os.path.join(settings_path, 'data')
        data_fpath = os.path.join(DATA_ROOT, args.data, "entity2id.txt")
        column_names = ['entity', 'id']
        entity_id = pd.read_table(data_fpath, sep='\t', names=column_names)
        entity_id_pairs = zip(list(entity_id['entity']), list(entity_id['id']))
        # entity2id = {k:v for k,v in entity_id_pairs}
        id2entity = {v:k for k,v in entity_id_pairs}

        data_fpath = os.path.join(DATA_ROOT, args.data, "relation2id.txt")
        column_names = ['rel', 'id']
        relation_id = pd.read_table(data_fpath, sep='\t', names=column_names)
        relation_name = list(relation_id['rel'])
        relation_name_inv = []
        for rel in relation_name:
            relation_name_inv.append(rel+"_inv")
        rel_id = list(relation_id['id'])
        rel_id_inv = []
        for r_id in rel_id:
            rel_id_inv.append(r_id+num_r)
        rel_id = rel_id + rel_id_inv
        relation_name = relation_name + relation_name_inv
        relation_id_pairs = zip(relation_name, rel_id)
        # rel2id = {k: v for k, v in relation_id_pairs}
        id2rel = {v: k for k, v in relation_id_pairs}

        assert model_path != "", "case study输出文件地址错误！：" + model_path
        print(os.path.dirname(model_path)+"/case_study_id.txt")
        with open(os.path.dirname(model_path)+"/case_study_id.txt", "w") as file:
            file.write("start case study")

    with ((torch.no_grad())):
        model = model.eval()

        srt2o = defaultdict(list)
        sr2o = defaultdict(list)

        # 计算filter的过滤字典
        ts_discr = []
        if is_need_filter:
            if "social" in args.data and args.data != "social_TKG_cate_level1_filter_discr1h":
                for i in range(len(ts)):
                    ts_discr.append(ts[i] // 3600)
                for i in range(len(src)):
                    sr2o[(src[i], e_idx_l[i])].append(dst[i])
                    srt2o[(src[i], e_idx_l[i], ts_discr[i])].append(dst[i])
            else:
                for i in range(len(src)):
                    sr2o[(src[i], e_idx_l[i])].append(dst[i])
                    srt2o[(src[i], e_idx_l[i], ts[i])].append(dst[i])

        num_test_instance = len(src)
        # num_test_batch = math.ceil(num_test_instance / args.bs)
        num_test_batch = len(val_bs_idx) - 1


        if find_score_weight:
            score_weight_list = np.linspace(0.1, 0.9, num=9).tolist()
        else:
            score_weight_list = [default_weight]

        best_score_weight_mrr = 0
        best_score_weight = -1
        train_node_degree = copy.deepcopy(model.ngh_finder.node_degree)
        for score_weight in score_weight_list:
            t_results = {}
            t_results_time = {}
            t_results_static = {}
            hit_n_count_raw = np.zeros(15)
            hit_n_count_time = np.zeros(15)
            testSet_raw_rank = []
            testSet_time_rank = []
            loss = 0
            sample_start = 0
            sample_end = 0
            num_batch = 0
            all_tail_seq_obj_c = all_tail_seq_obj.copy()
            history_entity_dic_copy = copy.deepcopy(history_entity_dic)
            model.ngh_finder.node_degree = copy.deepcopy(train_node_degree)
            # history_entity_dic_copy = history_entity_dic
            for k in tqdm(range(num_test_batch)):
            # for k in tqdm(range(num_test_batch)):
            #     s_idx = k * args.bs
            #     e_idx = min(num_test_instance - 1, s_idx + args.bs)

                sample_start = val_bs_idx[k]
                sample_end = val_bs_idx[k + 1]
                # sample_start = k * args.bs
                # sample_end = min(num_test_instance - 1, sample_start + args.bs)

                if sample_start == sample_end:
                    continue
                # src_l_cut = src[s_idx:e_idx]
                # dst_l_cut = dst[s_idx:e_idx]
                # ts_l_cut = ts[s_idx:e_idx]
                # e_l_cut = e_idx_l[s_idx:e_idx]

                # size = len(src_l_cut)
                # _, dst_l_fake = sampler.sample(size)

                n_batch = math.ceil((sample_end - sample_start) / args.bs)
                for x in range(n_batch):
                    num_batch += 1
                    s_idx = x * args.bs + sample_start
                    e_idx = min(sample_end, s_idx + args.bs)

                    if s_idx == e_idx:
                        continue
                    src_l_cut, dst_l_cut = src[s_idx:e_idx], dst[s_idx:e_idx]
                    ts_l_cut = ts[s_idx:e_idx]
                    e_l_cut = e_idx_l[s_idx:e_idx]

                    seq_idx = src_l_cut * num_r + e_l_cut
                    tail_seq = torch.Tensor(all_tail_seq_obj_c[seq_idx].todense())
                    one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
                    one_hot_tail_seq = one_hot_tail_seq.to(args.device)

                    batch_loss, score, attn_output_weights, subgraph_src = model.inference(src_l_cut, dst_l_cut, all_nodes_l, ts_l_cut, e_l_cut,
                                                        one_hot_tail_seq, tail_seq, history_entity_dic_copy, epoch, num_r, score_weight, stage=stage)


                    b_range = torch.arange(score.shape[0], device=args.device)
                    loss += batch_loss.item()
                    ranks = []
                    for i in range(len(src_l_cut)):
                        tmp_score = score[i]
                        pred_ground = tmp_score[dst_l_cut[i]]
                        ob_pred_comp1 = (tmp_score > pred_ground).data.cpu().numpy()
                        ob_pred_comp2 = (tmp_score == pred_ground).data.cpu().numpy()
                        target_rank_i = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1
                        ranks.append(target_rank_i)
                        if case_study:
                            subgraph_src1, subgraph_src2, subgraph_src3 = subgraph_src
                            node_records1, eidx_records1, t_records1 = subgraph_src1
                            node_records2, eidx_records2, t_records2 = subgraph_src2
                            node_records3, eidx_records3, t_records3 = subgraph_src3
                            path_num1 = eidx_records1.shape[1]
                            path_num2 = eidx_records2.shape[1]
                            path_num3 = eidx_records3.shape[1]
                            if target_rank_i==1:
                                # 写入case study的文件
                                # 保存格式： H@{};s,r,?,t;path1(edge);path1(t);path1(node);path2(edge);path2(t);path2(node);path3(edge);path3(t);path3(node)
                                tmp_str = "\nH@{1};\t" + "(" + id2entity[src_l_cut[i]] + "; " + id2rel[e_l_cut[i]] + ";  " + id2entity[dst_l_cut[i]] + ";  " + str(ts_l_cut[i]) + ");\n"
                                assert model_path != "", "case study输出文件地址错误！："+model_path
                                attn_output_weights_i = attn_output_weights[i].tolist()
                                max_val_lis = heapq.nlargest(10, attn_output_weights_i)
                                max_idx_lis = []
                                for item in max_val_lis:
                                    idx = attn_output_weights_i.index(item)
                                    max_idx_lis.append(idx)
                                    path_score = str(float(attn_output_weights_i[idx]))
                                    attn_output_weights_i[idx] = float('-inf')
                                    if idx < path_num1:
                                        tmp_str += "path(rel(time)->tail):" + id2rel.get(int(eidx_records1[i,idx,1]),str(int(eidx_records1[i,idx,1]))) + "(" + str(int(t_records1[i,idx,1])) + ")" + "->" + id2entity.get(int(node_records1[i,idx,1]), str(int(node_records1[i,idx,1]))) +"  path score: " + path_score + "\n"
                                    elif path_num1 <= idx < path_num1+path_num2:
                                        idx = idx-path_num1
                                        x = eidx_records2[i,idx,1]
                                        y = eidx_records2[i,idx,2]
                                        z = node_records2[i,idx,1]
                                        tmp_str += "rel_path:" + id2rel.get(int(eidx_records2[i,idx,1]),str(int(eidx_records2[i,idx,1]))) + "(" + str(int(t_records2[i,idx,1])) + ")" + "->" +  id2entity.get(int(node_records2[i,idx,1]), str(int(node_records2[i,idx,1]))) + \
                                                   "->" + id2rel.get(int(eidx_records2[i,idx,2]),str(int(eidx_records2[i,idx,2]))) + "(" + str(int(t_records2[i,idx,2])) + ")" + "->" +  id2entity.get(int(node_records2[i,idx,2]), str(int(node_records2[i,idx,2]))) + \
                                                   "  path score: " + path_score + "\n"
                                    else:
                                        idx = idx - path_num1 - path_num2
                                        tmp_str += "rel_path:" + id2rel.get(int(eidx_records3[i,idx,1]),str(int(eidx_records3[i,idx,1]))) + "(" + str(int(t_records3[i,idx,1])) + ")" + "->" +  id2entity.get(int(node_records3[i,idx,1]), str(int(node_records3[i,idx,1]))) + \
                                                   "->" + id2rel.get(int(eidx_records3[i,idx,2]),str(int(eidx_records3[i,idx,2]))) + "(" + str(int(t_records3[i,idx,2])) + ")" + "->" +  id2entity.get(int(node_records3[i,idx,2]), str(int(node_records3[i,idx,2]))) + \
                                                   "->" + id2rel.get(int(eidx_records3[i, idx, 3]), str(int(eidx_records3[i, idx, 3]))) + "(" + str(int(t_records3[i, idx, 3])) + ")" + "->" + id2entity.get(int(node_records3[i, idx, 3]), str(int(node_records3[i, idx, 3]))) + \
                                                   "  path score: " + path_score + "\n"

                                with open(os.path.dirname(model_path) + "/case_study_id.txt", "a") as file:
                                    file.write(tmp_str)



                        # raw ranking
                        # ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                        #     b_range, dst_l_cut]
                    testSet_raw_rank.extend(ranks)
                    for j in range(len(ranks)):
                        if ranks[j] < 11:
                            hit_n_count_raw[int(ranks[j])] += 1
                        elif 10 < ranks[j] < 51:
                            hit_n_count_raw[11] += 1
                        elif 50 < ranks[j] < 101:
                            hit_n_count_raw[12] += 1
                        else:
                            hit_n_count_raw[13] += 1

                    ranks = torch.tensor(ranks).float()
                    t_results['count_raw'] = torch.numel(ranks) + t_results.get('count_raw', 0.0)
                    t_results['mar_raw'] = torch.sum(ranks).item() + t_results.get('mar_raw', 0.0)
                    t_results['mrr_raw'] = torch.sum(1.0 / ranks).item() + t_results.get('mrr_raw', 0.0)
                    for j in range(10):
                        t_results['hits@{}_raw'.format(j + 1)] = torch.numel(ranks[ranks <= (j + 1)]) + t_results.get(
                            'hits@{}_raw'.format(j + 1), 0.0)

                    # logger.info("HITS10 {}".format(torch.numel(ranks[ranks <= 10])/torch.numel(ranks)))
                    # logger.info("HITS3 {}".format(torch.numel(ranks[ranks <= 3])/torch.numel(ranks)))
                    # logger.info("HITS1 {}".format(torch.numel(ranks[ranks <= 1])/torch.numel(ranks)))
                    # logger.info("MRR {}".format(torch.sum(1.0 / ranks).item()/torch.numel(ranks)))
                    # logger.info("MAR {}".format(torch.sum(ranks).item()/torch.numel(ranks)))

                    # 计算time-aware filter 和 time-unaware filter
                    if is_need_filter:

                        # 计算time-aware filter
                        time_aware_score = score
                        target_score = score[b_range, dst_l_cut]
                        for j in range(len(src_l_cut)):
                            if "social" in args.data and args.data != "social_TKG_cate_level1_filter_discr1h":
                                time_aware_score[j][srt2o[(src_l_cut[j], e_l_cut[j], ts_l_cut[j] // 3600)]] = -10000000
                            else:
                                time_aware_score[j][srt2o[(src_l_cut[j], e_l_cut[j], ts_l_cut[j])]] = -10000000
                        time_aware_score[b_range, dst_l_cut] = target_score

                        time_aware_ranks = []
                        for j in range(len(src_l_cut)):
                            tmp_score = time_aware_score[j]
                            pred_ground = tmp_score[dst_l_cut[j]]
                            ob_pred_comp1 = (tmp_score > pred_ground).data.cpu().numpy()
                            ob_pred_comp2 = (tmp_score == pred_ground).data.cpu().numpy()
                            target_rank_i = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1
                            time_aware_ranks.append(target_rank_i)

                        testSet_time_rank.extend(time_aware_ranks)
                        for j in range(len(time_aware_ranks)):
                            if time_aware_ranks[j] < 11:
                                hit_n_count_time[int(time_aware_ranks[j])] += 1
                            elif 10 < time_aware_ranks[j] < 51:
                                hit_n_count_time[11] += 1
                            elif 50 < time_aware_ranks[j] < 101:
                                hit_n_count_time[12] += 1
                            else:
                                hit_n_count_time[13] += 1

                        time_aware_ranks = torch.tensor(time_aware_ranks).float()
                        # time-aware filter ranking
                        # time_aware_ranks = 1 + torch.argsort(torch.argsort(time_aware_score, dim=1, descending=True), dim=1,
                        #                           descending=False)[b_range, dst_l_cut]
                        # time_aware_ranks = time_aware_ranks.float()


                        t_results_time['count_time_f'] = torch.numel(time_aware_ranks) + t_results_time.get('count_time_f', 0.0)
                        t_results_time['mar_time_f'] = torch.sum(time_aware_ranks).item() + t_results_time.get('mar_time_f', 0.0)
                        t_results_time['mrr_time_f'] = torch.sum(1.0 / time_aware_ranks).item() + t_results_time.get('mrr_time_f', 0.0)
                        for j in range(10):
                            t_results_time['hits@{}_time_f'.format(j + 1)] = torch.numel(time_aware_ranks[time_aware_ranks <= (j + 1)]) + t_results_time.get(
                                'hits@{}_time_f'.format(j + 1), 0.0)

                        # 计算time-unaware filter
                        time_unaware_score = score
                        for j in range(len(src_l_cut)):
                            time_unaware_score[j][sr2o[(src_l_cut[j], e_l_cut[j])]] = -10000000
                        time_unaware_score[b_range, dst_l_cut] = target_score

                        time_unaware_ranks = []
                        for j in range(len(src_l_cut)):
                            tmp_score = time_unaware_score[j]
                            pred_ground = tmp_score[dst_l_cut[j]]
                            ob_pred_comp1 = (tmp_score > pred_ground).data.cpu().numpy()
                            ob_pred_comp2 = (tmp_score == pred_ground).data.cpu().numpy()
                            target_rank_i = np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1
                            time_unaware_ranks.append(target_rank_i)
                        time_unaware_ranks = torch.tensor(time_unaware_ranks).float()

                        # time-unaware filter ranking
                        # time_unaware_ranks = 1 + torch.argsort(torch.argsort(time_unaware_score, dim=1, descending=True), dim=1,
                        #                           descending=False)[b_range, dst_l_cut]
                        # time_unaware_ranks = time_unaware_ranks.float()

                        t_results_static['count_time_uf'] = torch.numel(time_unaware_ranks) + t_results_static.get('count_time_uf', 0.0)
                        t_results_static['mar_time_uf'] = torch.sum(time_unaware_ranks).item() + t_results_static.get('mar_time_uf', 0.0)
                        t_results_static['mrr_time_uf'] = torch.sum(1.0 / time_unaware_ranks).item() + t_results_static.get('mrr_time_uf', 0.0)
                        for j in range(10):
                            t_results_static['hits@{}_time_uf'.format(j + 1)] = torch.numel(time_unaware_ranks[time_unaware_ranks <= (j + 1)]) + t_results_static.get(
                                'hits@{}_time_uf'.format(j + 1), 0.0)

                    row = src_l_cut * num_r + e_l_cut
                    col = dst_l_cut
                    d1 = np.ones(len(row))
                    tim_tail_seq = sp.csr_matrix((d1, (row, col)), shape=(num_e * num_r, num_e))
                    all_tail_seq_obj_c = all_tail_seq_obj_c + tim_tail_seq
                    model.ngh_finder.update_node_degree(src_l_cut, dst_l_cut)
                    # 更新历史记录
                    for i in range(len(src_l_cut)):
                        history_entity_dic_copy[(src_l_cut[i], e_l_cut[i])].put(dst_l_cut[i], ts_l_cut[i])

            loss = loss / num_batch
            t_results['mar_raw'] = round(t_results['mar_raw'] / t_results['count_raw'], 5)
            t_results['mrr_raw'] = round(t_results['mrr_raw'] / t_results['count_raw'], 5)
            for j in range(10):
                t_results['hits@{}_raw'.format(j + 1)] = round(
                    t_results['hits@{}_raw'.format(j + 1)] / t_results['count_raw'], 5)

            if is_need_filter:
                t_results_time['mar_time_f'] = round(t_results_time['mar_time_f'] / t_results_time['count_time_f'], 5)
                t_results_time['mrr_time_f'] = round(t_results_time['mrr_time_f'] / t_results_time['count_time_f'], 5)
                for j in range(10):
                    t_results_time['hits@{}_time_f'.format(j + 1)] = round(
                        t_results_time['hits@{}_time_f'.format(j + 1)] / t_results_time['count_time_f'], 5)

                t_results_static['mar_time_uf'] = round(t_results_static['mar_time_uf'] / t_results_static['count_time_uf'], 5)
                t_results_static['mrr_time_uf'] = round(t_results_static['mrr_time_uf'] / t_results_static['count_time_uf'], 5)
                for j in range(10):
                    t_results_static['hits@{}_time_uf'.format(j + 1)] = round(
                        t_results_static['hits@{}_time_uf'.format(j + 1)] / t_results_static['count_time_uf'], 5)

            logger.info("\n===========score weight: {}===========".format(score_weight))
            logger.info("===========evaluating or testing RAW===========")
            logger.info("HITS10 {}".format(t_results['hits@10_raw']))
            logger.info("HITS3 {}".format(t_results['hits@3_raw']))
            logger.info("HITS1 {}".format(t_results['hits@1_raw']))
            logger.info("MRR {}".format(t_results['mrr_raw']))
            logger.info("MAR {}".format(t_results['mar_raw']))

            if best_score_weight_mrr < t_results['mrr_raw']:
                best_score_weight_mrr = t_results['mrr_raw']
                best_score_weight = score_weight

            if is_need_filter:
                logger.info("===========evaluating or testing time-aware filter===========")
                logger.info("HITS10 {}".format(t_results_time['hits@10_time_f']))
                logger.info("HITS3 {}".format(t_results_time['hits@3_time_f']))
                logger.info("HITS1 {}".format(t_results_time['hits@1_time_f']))
                logger.info("MRR {}".format(t_results_time['mrr_time_f']))
                logger.info("MAR {}".format(t_results_time['mar_time_f']))

                logger.info("===========evaluating or testing time-unaware filter===========")
                logger.info("HITS10 {}".format(t_results_static['hits@10_time_uf']))
                logger.info("HITS3 {}".format(t_results_static['hits@3_time_uf']))
                logger.info("HITS1 {}".format(t_results_static['hits@1_time_uf']))
                logger.info("MRR {}".format(t_results_static['mrr_time_uf']))
                logger.info("MAR {}".format(t_results_static['mar_time_uf']))

            logger.info("===========raw ranks distribution===========")
            for i in range(len(hit_n_count_raw)):
                logger.info(hit_n_count_raw[i])
            logger.info("===========time ranks distribution===========")
            for i in range(len(hit_n_count_time)):
                logger.info(hit_n_count_time[i])
            if stage == "test" and "WIKI" not in args.data and "YAGO" not in args.data:
                testSet_time_rank = np.array(testSet_time_rank)
                testSet_raw_rank = np.array(testSet_raw_rank)
                time_data_fpath = os.path.join(settings.DATA_ROOT, args.data, "dense_sparse_index_time.npy")
                ngh_data_fpath = os.path.join(settings.DATA_ROOT, args.data, "dense_sparse_index_ngh.npy")
                time_interval_data_fpath = os.path.join(settings.DATA_ROOT, args.data, "dense_sparse_index_time_interval.npy")
                time_dense_sparse_index = np.load(time_data_fpath)
                ngh_dense_sparse_indesx = np.load(ngh_data_fpath)
                time_interval_dense_sparse_indesx = np.load(time_interval_data_fpath)
                time_dense_index = np.where(time_dense_sparse_index == 1)
                time_sparse_index = np.where(time_dense_sparse_index == -1)
                ngh_dense_index = np.where(ngh_dense_sparse_indesx == 1)
                ngh_sparse_index = np.where(ngh_dense_sparse_indesx == -1)
                time_interval_dense_index = np.where(time_interval_dense_sparse_indesx == 1)
                time_interval_sparse_index = np.where(time_interval_dense_sparse_indesx == -1)

                time_dense_raw_rank = testSet_raw_rank[time_dense_index]
                time_sparse_raw_rank = testSet_raw_rank[time_sparse_index]
                time_dense_time_rank = testSet_time_rank[time_dense_index]
                time_sparse_time_rank = testSet_time_rank[time_sparse_index]
                ngh_dense_raw_rank = testSet_raw_rank[ngh_dense_index]
                ngh_sparse_raw_rank = testSet_raw_rank[ngh_sparse_index]
                ngh_dense_time_rank = testSet_time_rank[ngh_dense_index]
                ngh_sparse_time_rank = testSet_time_rank[ngh_sparse_index]
                time_interval_dense_raw_rank = testSet_raw_rank[time_interval_dense_index]
                time_interval_sparse_raw_rank = testSet_raw_rank[time_interval_sparse_index]
                time_interval_dense_time_rank = testSet_time_rank[time_interval_dense_index]
                time_interval_sparse_time_rank = testSet_time_rank[time_interval_sparse_index]

                print(f"time dense average mrr: {np.mean(1 / time_dense_raw_rank)}, "
                      f"hit@1: {np.count_nonzero(time_dense_raw_rank <= 1) / len(time_dense_raw_rank)}, "
                      f"hit@3: {np.count_nonzero(time_dense_raw_rank <= 3) / len(time_dense_raw_rank)}, "
                      f"hit@10: {np.count_nonzero(time_dense_raw_rank <= 10) / len(time_dense_raw_rank)}")
                print(f"time sparse average mrr: {np.mean(1 / time_sparse_raw_rank)}, "
                      f"hit@1: {np.count_nonzero(time_sparse_raw_rank <= 1) / len(time_sparse_raw_rank)}, "
                      f"hit@3: {np.count_nonzero(time_sparse_raw_rank <= 3) / len(time_sparse_raw_rank)}, "
                      f"hit@10: {np.count_nonzero(time_sparse_raw_rank <= 10) / len(time_sparse_raw_rank)}")
                print(f"ngh dense average mrr: {np.mean(1 / ngh_dense_raw_rank)}, "
                      f"hit@1: {np.count_nonzero(ngh_dense_raw_rank <= 1) / len(ngh_dense_raw_rank)}, "
                      f"hit@3: {np.count_nonzero(ngh_dense_raw_rank <= 3) / len(ngh_dense_raw_rank)}, "
                      f"hit@10: {np.count_nonzero(ngh_dense_raw_rank <= 10) / len(ngh_dense_raw_rank)}")
                print(f"ngh sparse average mrr: {np.mean(1 / ngh_sparse_raw_rank)}, "
                      f"hit@1: {np.count_nonzero(ngh_sparse_raw_rank <= 1) / len(ngh_sparse_raw_rank)}, "
                      f"hit@3: {np.count_nonzero(ngh_sparse_raw_rank <= 3) / len(ngh_sparse_raw_rank)}, "
                      f"hit@10: {np.count_nonzero(ngh_sparse_raw_rank <= 10) / len(ngh_sparse_raw_rank)}")
                print(f"time interval dense average mrr: {np.mean(1 / time_interval_dense_raw_rank)}, "
                      f"hit@1: {np.count_nonzero(time_interval_dense_raw_rank <= 1) / len(time_interval_dense_raw_rank)}, "
                      f"hit@3: {np.count_nonzero(time_interval_dense_raw_rank <= 3) / len(time_interval_dense_raw_rank)}, "
                      f"hit@10: {np.count_nonzero(time_interval_dense_raw_rank <= 10) / len(time_interval_dense_raw_rank)}")
                print(f"time interval sparse average mrr: {np.mean(1 / time_interval_sparse_raw_rank)}, "
                      f"hit@1: {np.count_nonzero(time_interval_sparse_raw_rank <= 1) / len(time_interval_sparse_raw_rank)}, "
                      f"hit@3: {np.count_nonzero(time_interval_sparse_raw_rank <= 3) / len(time_interval_sparse_raw_rank)}, "
                      f"hit@10: {np.count_nonzero(time_interval_sparse_raw_rank <= 10) / len(time_interval_sparse_raw_rank)}")


                print(f"time dense average mrr: {np.mean(1 / time_dense_time_rank)}, "
                      f"hit@1: {np.count_nonzero(time_dense_time_rank <= 1) / len(time_dense_time_rank)}, "
                      f"hit@3: {np.count_nonzero(time_dense_time_rank <= 3) / len(time_dense_time_rank)}, "
                      f"hit@10: {np.count_nonzero(time_dense_time_rank <= 10) / len(time_dense_time_rank)}")
                print(f"time sparse average mrr: {np.mean(1 / time_sparse_time_rank)}, "
                      f"hit@1: {np.count_nonzero(time_sparse_time_rank <= 1) / len(time_sparse_time_rank)}, "
                      f"hit@3: {np.count_nonzero(time_sparse_time_rank <= 3) / len(time_sparse_time_rank)}, "
                      f"hit@10: {np.count_nonzero(time_sparse_time_rank <= 10) / len(time_sparse_time_rank)}")
                print(f"ngh dense average mrr: {np.mean(1 / ngh_dense_time_rank)}, "
                      f"hit@1: {np.count_nonzero(ngh_dense_time_rank <= 1) / len(ngh_dense_time_rank)}, "
                      f"hit@3: {np.count_nonzero(ngh_dense_time_rank <= 3) / len(ngh_dense_time_rank)}, "
                      f"hit@10: {np.count_nonzero(ngh_dense_time_rank <= 10) / len(ngh_dense_time_rank)}")
                print(f"ngh sparse average mrr: {np.mean(1 / ngh_sparse_time_rank)}, "
                      f"hit@1: {np.count_nonzero(ngh_sparse_time_rank <= 1) / len(ngh_sparse_time_rank)}, "
                      f"hit@3: {np.count_nonzero(ngh_sparse_time_rank <= 3) / len(ngh_sparse_time_rank)}, "
                      f"hit@10: {np.count_nonzero(ngh_sparse_time_rank <= 10) / len(ngh_sparse_time_rank)}")
                print(f"time interval dense average mrr: {np.mean(1 / time_interval_dense_time_rank)}, "
                      f"hit@1: {np.count_nonzero(time_interval_dense_time_rank <= 1) / len(time_interval_dense_time_rank)}, "
                      f"hit@3: {np.count_nonzero(time_interval_dense_time_rank <= 3) / len(time_interval_dense_time_rank)}, "
                      f"hit@10: {np.count_nonzero(time_interval_dense_time_rank <= 10) / len(time_interval_dense_time_rank)}")
                print(f"time interval sparse average mrr: {np.mean(1 / time_interval_sparse_time_rank)}, "
                      f"hit@1: {np.count_nonzero(time_interval_sparse_time_rank <= 1) / len(time_interval_sparse_time_rank)}, "
                      f"hit@3: {np.count_nonzero(time_interval_sparse_time_rank <= 3) / len(time_interval_sparse_time_rank)}, "
                      f"hit@10: {np.count_nonzero(time_interval_sparse_time_rank <= 10) / len(time_interval_sparse_time_rank)}")

        if find_score_weight:
            logger.info("\n===========find best score weight done===========")
            logger.info("best MRR: {}, score weight {}\n".format(best_score_weight_mrr, best_score_weight))
    return t_results['mrr_raw'], best_score_weight
