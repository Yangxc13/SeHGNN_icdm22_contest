import os
import gc
import json
import time
import uuid
import argparse
import datetime
import numpy as np

import torch
import torch.nn.functional as F

from model import *
from utils import *


def main(args):
    dataset, g, init_labels, num_nodes, n_classes, raw_train_nid, raw_val_nid, raw_test_nid, evaluator, \
        g_test, final_test_nid = load_icdm22(args)
    torch.save((raw_train_nid, raw_val_nid, raw_test_nid, num_nodes, init_labels), 'data_splits.pt')

    tgt_type = 'item'
    train_nid, val_nid, test_nid = raw_train_nid.clone(), raw_val_nid.clone(), raw_test_nid.clone()

    labels = init_labels.clone()
    labels[labels < 0] = -1

    resplit_graph = False

    if resplit_graph:
        # 训练集 验证集 划分时，属于同一 f 或 b 的要么同属训练集，要么同属验证集
        etype2adj = {}
        adj2etype = {}
        for i, etype in enumerate(g.etypes):
            stype, _, dtype = g.to_canonical_etype(etype)
            num_s = g.num_nodes(stype)
            num_d = g.num_nodes(dtype)
            if stype == 'item': stype = 'i'
            if dtype == 'item': dtype = 'i'
            name = f'{dtype}{stype}'

            etype2adj[etype] = name
            adj2etype[name] = etype

        adjs = {}
        for i, etype in enumerate(g.etypes):
            name = etype2adj[etype]
            if name not in ['fi', 'bi']: continue

            src, dst, eid = g._graph.edges(i)
            adj = SparseTensor(row=dst, col=src, sparse_sizes=(num_d, num_s))
            adjs[name] = adj

        set_random_seed(args.seed)
        print('seed =', args.seed)

        fi = adjs['fi'].clone()
        mask = labels[fi.storage.col()] >= 0
        fi_tv = fi.masked_select_nnz(mask, 'coo')

        all_f = torch.where(fi_tv.storage.rowcount() > 0)[0] # 41410
        perm = torch.randperm(len(all_f))

        train_f = all_f[perm[:int(0.8*len(all_f))]]
        val_f = all_f[perm[int(0.8*len(all_f)):]]

        train_nid = torch.sort(fi_tv[train_f].storage.col())[0]
        val_nid = torch.sort(fi_tv[val_f].storage.col())[0]

        # 去掉多余的 ib bi 边，重新构建图
        bi = adjs['bi'].clone()

        label_train = torch.zeros(len(labels))
        label_train[train_nid] = 1
        label_val = torch.zeros(len(labels))
        label_val[val_nid] = 1

        bi.storage._value = label_train[bi.storage.col()]
        a = bi.sum(1) # 6854 values, 35690 sum, a[c].sum()=21144
        print(torch.unique(a[a>0].long(), return_counts=True))
        bi.storage._value = label_val[bi.storage.col()]
        b = bi.sum(1) # 2491 values, 8654 sum, b[c].sum()=6276
        print(torch.unique(b[b>0].long(), return_counts=True))
        c = torch.where((a > 0) & (b > 0))[0] # 1426

        a_mask = torch.randn(len(a)) > 0
        a0 = torch.where((a > 0) & (b > 0) & a_mask)[0] # 755 values, a[a0].sum()=19968, b[a0].sum()=4850
        a1 = torch.where((a > 0) & (b > 0) & (~a_mask))[0] # 671 values, a[a1].sum()=1176, b[a1].sum()=1426
        print(a0.shape, a[a0].sum(), b[a0].sum())
        print(a1.shape, a[a1].sum(), b[a1].sum())

        bi_train_mask = (a > 0) & ((b == 0) | a_mask)
        bi_val_mask = (b > 0) & ((a == 0) | (~a_mask))
        assert torch.all(bi_train_mask & bi_val_mask == 0)

        g_train = g.clone()
        for i, etype in enumerate(g_train.etypes):
            name = etype2adj[etype]
            if name == 'bi':
                src, dst, eid = g_train._graph.edges(i)
                drop_eid = eid[bi_val_mask[dst]]
                g_train.remove_edges(drop_eid, etype)
                break

        g_val = g.clone()
        for i, etype in enumerate(g_val.etypes):
            name = etype2adj[etype]
            if name == 'bi':
                src, dst, eid = g_val._graph.edges(i)
                drop_eid = eid[bi_train_mask[dst]]
                g_val.remove_edges(drop_eid, etype)
                break

        dtype, stype = 'b', 'item'
        dst_name, src_name = 'b', 'i'
        for graph in [g_train, g_val]:
            graph.nodes[stype].data['flag'] = (graph.nodes[stype].data[src_name].sum(1, keepdim=True) > 1e-6).float() # 注意，目前只可针对 icdm 使用，因为icdm的特征不含负值

            graph[etype].update_all(fn.copy_u('flag', 'm'), fn.mean('m', 'num'), etype=etype)

            graph[etype].update_all(fn.copy_u(src_name, 'm'), fn.mean('m', dst_name), etype=etype)
            
            graph.nodes[dtype].data[dst_name] = \
                graph.nodes[dtype].data[dst_name] * (graph.nodes[dtype].data['num'] > 0) / (graph.nodes[dtype].data['num'] + 1e-12)

            graph.nodes[stype].data.pop('flag')
            graph.nodes[dtype].data.pop('num')

        feat_i_train, feats_b_train, feats_f_train, adjs_train = generate_features(g_train)
        feat_i_val, feats_b_val, feats_f_val, adjs_val = generate_features(g_val)

        train_feats = [feat_i_train, feats_b_train, feats_f_train, adjs_train]
        val_feats = [feat_i_val, feats_b_val, feats_f_val, adjs_val]

    else:
        if args.enlarge_val_set:
            train_err_nodes, train_err_times, val_err_nodes, val_err_times = torch.load('err_times.pt')

            remove_val = val_err_nodes[val_err_times >= 3]
            remove_train = train_err_nodes[train_err_times >= 3]

            assert set(remove_val.tolist()).issubset(set(val_nid.tolist()))
            assert set(remove_train.tolist()).issubset(set(train_nid.tolist()))

            print('remove_train', torch.unique(init_labels[remove_train], return_counts=True))
            print('remove_val', torch.unique(init_labels[remove_val], return_counts=True))

            val_nid = set(val_nid.tolist()) - set(remove_val.tolist())
            val_nid = torch.LongTensor(list(val_nid))
            train_nid = (set(train_nid.tolist()) | set(remove_val.tolist())) - set(remove_train.tolist())
            # train_nid = set(train_nid.tolist()) | set(remove_val.tolist())
            train_nid = torch.LongTensor(list(train_nid))

            print(f'Train nums {len(raw_train_nid)}->{len(train_nid)}', torch.unique(init_labels[train_nid], return_counts=True))
            print(f'Valid nums {len(raw_val_nid)}->{len(val_nid)}', torch.unique(init_labels[val_nid], return_counts=True))

        feat_i, feats_b, feats_f, adjs = generate_features(g)
        train_feats = val_feats = test_feats = [feat_i, feats_b, feats_f, adjs]

    train_node_nums = len(train_nid)
    valid_node_nums = len(val_nid)
    test_node_nums = len(test_nid)
    trainval_point = train_node_nums
    valtest_point = trainval_point + valid_node_nums
    total_num_nodes = len(train_nid) + len(val_nid) + len(test_nid)

    feat_i_test, feats_b_test, feats_f_test, adjs_test = generate_features(g_test)
    final_test_feats = [feat_i_test, feats_b_test, feats_f_test, adjs_test]


    checkpt_folder = f'./output/{args.dataset}/'
    if not os.path.exists(checkpt_folder):
        os.makedirs(checkpt_folder)

    if args.amp:
        scalar = torch.cuda.amp.GradScaler()
    else:
        scalar = None

    device = "cuda:{}".format(args.gpu)
    labels_cuda = labels.long().to(device)
    eval_loader = []

    stage = 0
    epochs = 200

    for run_time in range(args.run_times):
        checkpt_file = checkpt_folder + uuid.uuid4().hex
        print(checkpt_file)

        if len(eval_loader):
            del eval_loader
        eval_loader = []

        def warp_loader(eval_nodes, feats):
            part_loader = []
            feats_i, feats_b, feats_f, adjs = feats

            for batch_start in range(0, len(eval_nodes), args.test_batch_size):
                batch_end = min(len(eval_nodes), batch_start + args.test_batch_size)
                batch = eval_nodes[batch_start:batch_end]

                batch_feats = {'i': feats_i[batch]}

                nodes = torch.unique(adjs['ib'][batch].storage.col())
                batch_adj_ib = adjs['ib'][batch, nodes]
                for k, v in feats_b.items():
                    batch_feats[k] = batch_adj_ib @ v[nodes]

                nodes = torch.unique(adjs['if'][batch].storage.col())
                batch_adj_if = adjs['if'][batch, nodes]
                for k, v in feats_f.items():
                    batch_feats[k] = batch_adj_if @ v[nodes]

                part_loader.append((batch_feats, labels[batch]))

            return part_loader

        eval_loader += warp_loader(train_nid, train_feats)
        eval_loader += warp_loader(val_nid, val_feats)

        feat_keys = list(eval_loader[0][0].keys())

        # =======
        # Construct network
        # # =======
        if False:
            model_class = SeHGNN_mix
        else:
            model_class = SeHGNN_mag

        model = model_class(args.dataset,
            args.embed_size, args.hidden, n_classes,
            feat_keys, 'i',
            dropout=args.dropout,
            input_drop=args.input_drop,
            att_drop=args.att_drop,
            n_layers_1=args.n_layers_1,
            n_layers_2=args.n_layers_2,
            act=args.act,
            residual=args.residual,
            bns=args.bns)
        model = model.to(device)
        print(model)
        print("# Params:", get_n_params(model))

        loss_fcn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                    weight_decay=args.weight_decay)

        best_epoch = 0
        best_val_loss = 1000000
        best_val_acc = 0
        best_val_ap = 0
        best_test_acc = 0
        best_test_ap = 0
        count = 0
        gc.collect()

        train_loader = torch.utils.data.DataLoader(
            train_nid, batch_size=args.batch_size, shuffle=True, drop_last=False)

        for epoch in range(epochs):
            gc.collect()
            start = time.time()
            loss, acc, ap = train(model, train_loader, loss_fcn, optimizer, evaluator, device, train_feats, labels_cuda, scalar=scalar)
            end = time.time()

            log = f'Epoch {epoch}, Time(s): {end-start:.6f}, estimated train loss {loss:.6f}, acc {acc*100:.4f}, ap {ap*100:.4f}\n'

            if epoch % args.eval_every == 0:
                torch.cuda.empty_cache()
                with torch.no_grad():
                    model.eval()
                    raw_preds = []

                    start = time.time()
                    for batch_feats, _ in eval_loader:
                        batch_feats = {k: v.to(device) for k,v in batch_feats.items()}
                        raw_preds.append(model(batch_feats).cpu())
                    raw_preds = torch.cat(raw_preds, dim=0)
                    preds = raw_preds.argmax(dim=-1)
                    probs = F.softmax(raw_preds, dim=1)[:, 1]

                    loss_val = loss_fcn(raw_preds[trainval_point:valtest_point], labels[val_nid]).item()
                    
                    train_acc = evaluator(preds[:trainval_point], labels[train_nid])
                    train_ap = average_precision_score(labels[train_nid].numpy(), probs[:trainval_point].numpy())

                    val_acc = evaluator(preds[trainval_point:valtest_point], labels[val_nid])
                    val_ap = average_precision_score(labels[val_nid].numpy(), probs[trainval_point:valtest_point].numpy())

                    end = time.time()
                    log += f'Time: {end-start:.6f}, Val loss: {loss_val:.6f}\n'
                    log += f'Train: {train_acc*100:.4f} {train_ap*100:.4f}, Val acc: {val_acc*100:.4f}, Val ap: {val_ap*100:.4f}'
                    log += '\n'

                if val_ap > best_val_ap: # val_acc > best_val_acc
                    best_epoch = epoch
                    best_val_acc = val_acc
                    best_val_ap = val_ap

                    torch.save(model.state_dict(), f'{checkpt_file}.pkl')
                    count = 0
                else:
                    count = count + args.eval_every
                    if count >= args.patience:
                        break

                log += f'Best Epoch {best_epoch}, Val acc {best_val_acc*100:.4f}, Val ap {best_val_ap*100:.4f}'
                model.train()
                torch.cuda.empty_cache()
                gc.collect()
            print(log, flush=True)

        print(f'Best Epoch {best_epoch}, Val {best_val_acc*100:.4f}, Test {best_val_ap*100:.4f}')

        model.load_state_dict(torch.load(f'{checkpt_file}.pkl'))
        torch.cuda.empty_cache()

        if True:
            all_loader = torch.utils.data.DataLoader(
                torch.cat((train_nid, val_nid, test_nid)), batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            # all_loader = torch.utils.data.DataLoader(
            #     torch.arange(len(feats_i)), batch_size=args.test_batch_size, shuffle=False, drop_last=False)

            raw_preds = gen_output_torch(model, test_feats, all_loader, device)
            torch.save(raw_preds, f'{checkpt_file}.pt')

            if args.dataset == 'icdm22':
                print(f'{checkpt_file}.pt', f'{checkpt_file}.json')
                rev_item_map = torch.load('rev_item_map.pt')
                with open(f'{checkpt_file}_test.json', 'w+') as f:
                    pred_score = F.softmax(raw_preds[-len(test_nid):], dim=1)[:, 1].data.cpu().numpy()
                    for idx, pred_score in zip(test_nid.numpy(), pred_score):
                        y_dict = {}
                        y_dict["item_id"] = int(rev_item_map[idx])
                        y_dict["score"] = float(pred_score)
                        json.dump(y_dict, f)
                        f.write('\n')
        if True:
            final_test_loader = torch.utils.data.DataLoader(
                final_test_nid, batch_size=args.test_batch_size, shuffle=False, drop_last=False)
            # final_test_loader = torch.utils.data.DataLoader(
            #     torch.arange(len(feat_i_test)), batch_size=args.test_batch_size, shuffle=False, drop_last=False)

            raw_preds = gen_output_torch(model, final_test_feats, final_test_loader, device)
            torch.save(raw_preds, f'{checkpt_file}_final_test.pt')

            print(f'{checkpt_file}_final_test.pt', f'{checkpt_file}_final_test.json')
            rev_item_map = torch.load('rev_item_map_test.pt')
            with open(f'{checkpt_file}_final_test.json', 'w+') as f:
                pred_score = F.softmax(raw_preds, dim=1)[:, 1].data.cpu().numpy()
                for idx, pred_score in zip(final_test_nid.numpy(), pred_score):
                    y_dict = {}
                    y_dict["item_id"] = int(rev_item_map[idx])
                    y_dict["score"] = float(pred_score)
                    json.dump(y_dict, f)
                    f.write('\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SeHGNN')
    ## For environment costruction
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed used in the training")
    parser.add_argument("--dataset", type=str, default="ogbn-mag")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--root", type=str, default='/ssd/yangxiaocheng/icdm_graph_competition/')
    parser.add_argument("--datadir", type=str, default='../data/')
    ## For pre-processing
    parser.add_argument("--extra-embedding", type=str, default='',
                        help="whether to use extra embeddings from RotatE")
    parser.add_argument("--embed-size", type=int, default=256,
                        help="inital embedding size of nodes with no attributes")
    parser.add_argument("--num-hops", type=int, default=2,
                        help="number of hops for propagation of raw labels")
    ## For network structure
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout on activation")
    parser.add_argument("--n-layers-1", type=int, default=2,
                        help="number of layers of feature projection")
    parser.add_argument("--n-layers-2", type=int, default=2,
                        help="number of layers of the downstream task")
    # parser.add_argument("--n-layers-3", type=int, default=4,
    #                     help="number of layers of residual label connection")
    parser.add_argument("--input-drop", type=float, default=0.1,
                        help="input dropout of input features")
    parser.add_argument("--att-drop", type=float, default=0.,
                        help="attention dropout of model")
    parser.add_argument("--residual", action='store_true', default=False,
                        help="whether to connect the input features")
    parser.add_argument("--act", type=str, default='relu',
                        help="the activation function of the model")
    parser.add_argument("--bns", action='store_true', default=False,
                        help="whether to process the input features")
    # parser.add_argument("--norm", action='store_true', default=False,
    #                     help="whether to norm the input features")
    ## for training
    parser.add_argument("--amp", action='store_true', default=False,
                        help="whether to amp to accelerate training with float16(half) calculation")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--test-batch-size", type=int, default=10000)
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop of times of the experiment")
    # parser.add_argument("--alpha", type=float, default=0.5,
    #                     help="initial residual parameter for the model")
    parser.add_argument("--enlarge-val-set", action='store_true', default=False)
    parser.add_argument("--run-times", type=int, default=5)

    # args = parser.parse_args(('--dataset icdm22'
    #     + ' --num-hops 2 --n-layers-1 2 --n-layers-2 2'
    #     + ' --residual --act leaky_relu --lr 0.001 --weight-decay 0'
    #     + ' --patience 50 --amp'
    #     + ' --test-batch-size 32768 --batch-size 1024').split(' '))
    args = parser.parse_args()
    print(args)

    main(args)
