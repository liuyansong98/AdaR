import gc

import torch
import numpy as np
from tqdm import tqdm
import math
from eval import *
import logging
from datetime import datetime
import scipy.sparse as sp
from torch.cuda.amp import GradScaler, autocast
import copy

logging.basicConfig(level=logging.DEBUG)


def train_val(train_val_data, all_nodes_l, model, history_entity_dic, args, optimizer, early_stopper, ngh_finders,
              rand_samplers, logger, num_e, num_r):
    train_data, val_data = train_val_data
    train_src_l, train_dst_l, train_ts_l, train_e_idx_l = train_data
    val_src_l, val_dst_l, val_ts_l, val_e_idx_l = val_data
    train_rand_sampler, val_rand_sampler = rand_samplers
    partial_ngh_finder, full_ngh_finder = ngh_finders

    model.update_ngh_finder(full_ngh_finder)

    src_num_ngh_list = []
    for i in range(len(train_src_l)):
        ngh_idx, ngh_eidx, ngh_ts, ngh_binomial_prob = model.ngh_finder.find_before(train_src_l[i], train_ts_l[i],
                                                                                    e_idx=None)
        src_num_ngh_list.append(len(ngh_idx))
    src_num_ngh_list = np.array(src_num_ngh_list)
    src_num_ngh_id_list = np.argsort(src_num_ngh_list)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2,
                                                           threshold=0.0001, threshold_mode='rel', cooldown=0,
                                                           min_lr=1e-08,
                                                           eps=1e-08)

    num_instance = len(train_src_l)

    train_bs_idx = [0]
    train_ts_l_discr = []
    if "social" in args.data and args.data != "social_TKG_cate_level1_filter_discr1h":
        for i in range(len(train_ts_l)):
            train_ts_l_discr.append(train_ts_l[i] // 3600)
        for k in range(0, len(train_ts_l_discr) - 1):
            if train_ts_l_discr[k] != train_ts_l_discr[k + 1]:
                train_bs_idx.append(k + 1)
        train_bs_idx.append(len(train_ts_l_discr))
    else:
        for k in range(0, len(train_ts_l) - 1):
            if train_ts_l[k] != train_ts_l[k + 1]:
                train_bs_idx.append(k + 1)
        train_bs_idx.append(len(train_ts_l))

    val_bs_idx = [0]
    val_ts_l_discr = []
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

    num_batch = math.ceil(num_instance / args.bs)
    timestamp_num = len(train_bs_idx) - 1
    train_losses = []

    # num_batch = 0
    max_mrr = 0.0
    for epoch in range(args.n_epoch):
        # np.random.shuffle(idx_list)
        logger.info("\n" + "=" * 80 + "\n"
                                      f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')} [Epoch {epoch:03d}] start")
        logger.info('start {} epoch'.format(epoch))
        t_results = {}
        loss = 0
        # sample_start = 0
        # sample_end = 0
        sample_end = num_instance
        all_tail_seq = sp.csr_matrix(([], ([], [])), shape=(num_e * num_r, num_e))
        history_entity_dic_copy = copy.deepcopy(history_entity_dic)
        # history_entity_dic_copy = history_entity_dic
        model.ngh_finder.init_node_degree()

        # for k in tqdm(range(timestamp_num)):
        for k in tqdm(range(num_batch)):
            model.train()
            # sample_start = train_bs_idx[k]
            # sample_end = train_bs_idx[k + 1]

            # n_batch = math.ceil((sample_end-sample_start) / args.bs)
            # for x in range(n_batch):
            # if k < 30:
            #     continue
            # else:
            # num_batch += 1
            s_idx = k * args.bs
            e_idx = min(sample_end, s_idx + args.bs)

            if s_idx == e_idx:
                continue
            bach_id = src_num_ngh_id_list[s_idx:e_idx]
            avg_ngh = np.mean(src_num_ngh_list[bach_id])
            ngh_sample_pram = 0
            if avg_ngh > 64:
                ngh_sample_pram = 64
            elif 4 < avg_ngh <= 64:
                ngh_sample_pram = int(avg_ngh * 0.8)
            else:
                ngh_sample_pram = 4
            # src_l_cut, dst_l_cut = train_src_l[s_idx:e_idx], train_dst_l[s_idx:e_idx]
            # ts_l_cut = train_ts_l[s_idx:e_idx]
            # e_l_cut = train_e_idx_l[s_idx:e_idx]
            src_l_cut, dst_l_cut = train_src_l[bach_id], train_dst_l[bach_id]
            ts_l_cut = train_ts_l[bach_id]
            e_l_cut = train_e_idx_l[bach_id]

            seq_idx = src_l_cut * num_r + e_l_cut
            tail_seq = torch.Tensor(all_tail_seq[seq_idx].todense())
            one_hot_tail_seq = tail_seq.masked_fill(tail_seq != 0, 1)
            one_hot_tail_seq = one_hot_tail_seq.to(args.device)

            # size = len(src_l_cut)
            # _, dst_l_fake = train_rand_sampler.sample(negatives * size)
            optimizer.zero_grad()
            # with autocast():
            model.train()
            batch_loss, score, _, _ = model.inference(src_l_cut, dst_l_cut, all_nodes_l, ts_l_cut, e_l_cut,
                                                      one_hot_tail_seq, tail_seq,
                                                      history_entity_dic_copy,
                                                      epoch=epoch, num_r=num_r, stage='train',
                                                      ngh_sample_pram=ngh_sample_pram)

            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.item()
            # torch.cuda.empty_cache()
            b_range = torch.arange(score.shape[0], device=args.device)

            # raw ranking
            ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                b_range, dst_l_cut]

            ranks = ranks.float()
            t_results['count_raw'] = torch.numel(ranks) + t_results.get('count_raw', 0.0)
            t_results['mar_raw'] = torch.sum(ranks).item() + t_results.get('mar_raw', 0.0)
            t_results['mrr_raw'] = torch.sum(1.0 / ranks).item() + t_results.get('mrr_raw', 0.0)
            for i in range(10):
                t_results['hits@{}_raw'.format(i + 1)] = torch.numel(ranks[ranks <= (i + 1)]) + t_results.get(
                    'hits@{}_raw'.format(i + 1), 0.0)

        t_results['mar_raw'] = round(t_results['mar_raw'] / t_results['count_raw'], 5)
        t_results['mrr_raw'] = round(t_results['mrr_raw'] / t_results['count_raw'], 5)
        for j in range(10):
            t_results['hits@{}_raw'.format(j + 1)] = round(
                t_results['hits@{}_raw'.format(j + 1)] / t_results['count_raw'], 5)

        train_losses.append(loss / num_batch)
        logger.info(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')} [Epoch {epoch:03d}] end"
                    f"batch train loss total={train_losses[epoch]:.4f} | "
                    + "\n")

        logger.info("===========training RAW===========")
        logger.info("Epoch {}, HITS10 {}".format(epoch, t_results['hits@10_raw']))
        logger.info("Epoch {}, HITS3 {}".format(epoch, t_results['hits@3_raw']))
        logger.info("Epoch {}, HITS1 {}".format(epoch, t_results['hits@1_raw']))
        logger.info("Epoch {}, MRR {}".format(epoch, t_results['mrr_raw']))
        logger.info("Epoch {}, MAR {}".format(epoch, t_results['mar_raw']))
        # 更新历史记录
        for i in range(len(train_src_l)):
            history_entity_dic_copy[train_src_l[i], train_e_idx_l[i]].put(train_dst_l[i], train_ts_l[i])
        row = train_src_l * num_r + train_e_idx_l
        col = train_dst_l
        d1 = np.ones(len(row))
        all_tail_seq = sp.csr_matrix((d1, (row, col)), shape=(num_e * num_r, num_e))
        model.ngh_finder.update_node_degree(train_src_l, train_dst_l)
        mrr, best_score_weight = eval_one_epoch(model, all_nodes_l, val_rand_sampler, val_src_l, val_dst_l, val_ts_l,
                                                val_e_idx_l, val_bs_idx, args, logger, all_tail_seq,
                                                history_entity_dic_copy, num_e, num_r, epoch,
                                                is_need_filter=False, default_weight=args.default_weight, stage='val')
        if epoch > 12:
            scheduler.step(mrr)
        del history_entity_dic_copy
        gc.collect()

        print(f"Epoch {epoch + 1}: Learning rate: {optimizer.param_groups[0]['lr']}")

        if epoch == 0:
            checkpoint_dir = '/'.join(model.checkpoint_path.split('/')[:-1])
            # model.ngh_finder.save_ngh_stats(checkpoint_dir)

        if early_stopper.early_stop_check(mrr) and epoch > 36:
            # if early_stopper.early_stop_check(mrr):
            logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
            logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
            best_checkpoint_path = model.checkpoint_path
            model.load_state_dict(torch.load(best_checkpoint_path, map_location=args.device))
            logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
            model.eval()
            break
        elif max_mrr < mrr:
            torch.save(model.state_dict(), model.checkpoint_path)
            max_mrr = mrr
            logger.info("saved best Epoch {}, MRR {}".format(epoch, mrr))
