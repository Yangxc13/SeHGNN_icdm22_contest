import os
import gc
import sys
import random
import datetime
from tqdm import tqdm

import dgl
import dgl.function as fn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_sparse import remove_diag, set_diag

import numpy as np
from sklearn.metrics import f1_score, average_precision_score


def set_random_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


def clear_hg(new_g, echo=False):
    if echo: print('Remove keys left after propagation')
    for ntype in new_g.ntypes:
        keys = list(new_g.nodes[ntype].data.keys())
        if len(keys):
            if echo: print(ntype, keys)
            for k in keys:
                new_g.nodes[ntype].data.pop(k)
    return new_g


def train(model, train_loader, loss_fcn, optimizer, evaluator, device, feats, labels_cuda, scalar=None):
    model.train()
    total_loss = 0
    iter_num = 0
    y_true, y_pred, y_score = [], [], []
    feats_i, feats_b, feats_f, adjs = feats

    for batch in tqdm(train_loader):
        batch_feats = {'i': feats_i[batch].to(device)}

        nodes = torch.unique(adjs['ib'][batch].storage.col())
        batch_adj_ib = adjs['ib'][batch, nodes]
        for k, v in feats_b.items():
            batch_feats[k] = (batch_adj_ib @ v[nodes]).to(device)

        nodes = torch.unique(adjs['if'][batch].storage.col())
        batch_adj_if = adjs['if'][batch, nodes]
        for k, v in feats_f.items():
            batch_feats[k] = (batch_adj_if @ v[nodes]).to(device)

        batch_y = labels_cuda[batch]

        optimizer.zero_grad()
        if scalar is not None:
            with torch.cuda.amp.autocast():
                output_att = model(batch_feats)
                loss_train = loss_fcn(output_att, batch_y)
            scalar.scale(loss_train).backward()
            scalar.step(optimizer)
            scalar.update()
        else:
            output_att = model(batch_feats)
            loss_train = loss_fcn(output_att, batch_y, batch_weights)
            loss_train.backward()
            optimizer.step()

        y_true.append(batch_y.cpu().to(torch.long))
        y_pred.append(output_att.argmax(dim=-1, keepdim=True).cpu())
        y_score.append(output_att.softmax(dim=1)[:,1].data.cpu())
        total_loss += loss_train.item()
        iter_num += 1
    loss = total_loss / iter_num
    acc = evaluator(torch.cat(y_true, dim=0), torch.cat(y_pred, dim=0))
    ap = average_precision_score(torch.cat(y_true, dim=0).numpy(), torch.cat(y_score, dim=0).numpy())
    return loss, acc, ap


@torch.no_grad()
def gen_output_torch(model, feats, test_loader, device):
    model.eval()
    preds = []
    feats_i, feats_b, feats_f, adjs = feats
    for batch in tqdm(test_loader):
        batch_feats = {'i': feats_i[batch].to(device)}

        nodes = torch.unique(adjs['ib'][batch].storage.col())
        batch_adj_ib = adjs['ib'][batch, nodes]
        for k, v in feats_b.items():
            batch_feats[k] = (batch_adj_ib @ v[nodes]).to(device)

        nodes = torch.unique(adjs['if'][batch].storage.col())
        batch_adj_if = adjs['if'][batch, nodes]
        for k, v in feats_f.items():
            batch_feats[k] = (batch_adj_if @ v[nodes]).to(device)

        preds.append(model(batch_feats).cpu())
    preds = torch.cat(preds, dim=0)
    return preds


def load_icdm22(args):
    if not os.path.exists('icdm_session1.pt'):
        sys.path.append(f'{args.root}/dgl_example/')
        from icdm2022_dataset import ICDM2022Dataset
        dataset = ICDM2022Dataset(session='session1', load_labels=True, raw_dir=f'{args.root}/data/dgl_data', verbose=True)
        g = dataset[0]
        n_classes = dataset.num_classes  # 2
        torch.save(g, 'icdm_session1.pt')
        torch.save(dataset.rev_item_map, 'rev_item_map.pt')
    else:
        g = torch.load('icdm_session1.pt')
        dataset = (g)
        n_classes = 2
    '''
    {'a': 204128, 'b': 45602, 'c': 178777, 'd': 1687, 'e': 379320, 'f': 1063739, 'i(tem)': 11933366} total 13806619
    {
        f -> G -> a: 59208411
        e -> H -> a: 1194499
        i -> A -> b: 3781140
        f -> D -> c: 1269499
        f -> C -> d: 521220
        f -> F -> e: 1248723
        f -> B -> i: 11683940
    } # total 78907432 (157814864 considering both directions)
    '''
    tgt_type = 'item'

    init_labels = g.nodes[tgt_type].data.pop('label')
    num_nodes = len(init_labels)
    print(torch.unique(init_labels, return_counts=True))
    # {nan: 11847804, 0: 77198, 1: 8364}

    for k, v in g.ndata['h'].items():
        print(k, v.shape)
    '''
        a torch.Size([204128, 256])
        b torch.Size([45602, 256])
        c torch.Size([178777, 256])
        d torch.Size([1687, 256])
        e torch.Size([379320, 256])
        f torch.Size([1063739, 256])
        item torch.Size([11933366, 256])
    '''
    for ntype in g.ntypes:
        if ntype == 'item':
            g.nodes['item'].data['i'] = g.nodes['item'].data.pop('h')
        else:
            g.nodes[ntype].data[ntype] = g.nodes[ntype].data.pop('h')

    train_mask = g.nodes[tgt_type].data.pop('train_mask')
    val_mask = g.nodes[tgt_type].data.pop('val_mask')
    test_mask = g.nodes[tgt_type].data.pop('test_mask')
    train_nid = torch.nonzero(train_mask, as_tuple=False).squeeze() # 68449
    val_nid = torch.nonzero(val_mask, as_tuple=False).squeeze()     # 17113
    test_nid = torch.nonzero(test_mask, as_tuple=False).squeeze()   # 36925

    evaluator = lambda preds, labels: (preds.flatten() == labels.flatten()).sum().item() / len(labels.flatten())

    if not os.path.exists('icdm_session2.pt'):
        sys.path.append(f'{args.root}/dgl_example/')
        from icdm2022_dataset import ICDM2022Dataset
        dataset = ICDM2022Dataset(session='session2', load_labels=False, raw_dir=f'{args.root}/data/dgl_data', verbose=True)
        g_test = dataset[0]
        torch.save(g_test, 'icdm_session2.pt')
        torch.save(dataset.rev_item_map, 'rev_item_map_test.pt')
    else:
        g_test = torch.load('icdm_session2.pt')
    for ntype in g_test.ntypes:
        if ntype == 'item':
            g_test.nodes['item'].data['i'] = g_test.nodes['item'].data.pop('h')
        else:
            g_test.nodes[ntype].data[ntype] = g_test.nodes[ntype].data.pop('h')
    final_test_mask = g_test.nodes[tgt_type].data.pop('test_mask')
    final_test_nid = torch.nonzero(final_test_mask, as_tuple=False).squeeze()   # 75086

    return dataset, g, init_labels, num_nodes, n_classes, train_nid, val_nid, test_nid, evaluator, g_test, final_test_nid


def hg_propagate_icdm22(new_g, tgt_type, num_hops, max_hops, extra_metapath, echo=False, remove_zero_degree_nodes=False, remove_zero_degree_features=False):
    etype2adj = {}
    adj2etype = {}
    for i, etype in enumerate(new_g.etypes):
        tic = datetime.datetime.now()
        stype, _, dtype = new_g.to_canonical_etype(etype)
        if stype == 'item': stype = 'i'
        if dtype == 'item': dtype = 'i'
        adj_name = f'{dtype}{stype}'
        etype2adj[etype] = adj_name
        adj2etype[adj_name] = etype

    for hop in range(1, max_hops):
        reserve_heads = [ele[-hop:] for ele in extra_metapath if len(ele) > hop]
        for etype in new_g.etypes:
            stype, _, dtype = new_g.to_canonical_etype(etype)
            # sname = stype if stype != 'item' else 'i'
            # dname = dtype if dtype != 'item' else 'i'

            for k in list(new_g.nodes[stype].data.keys()):
                if hop == 1 and k == 'item': k = 'i'
                if len(k) == hop:
                    if dtype == 'item':
                        current_dst_name = f'i{k}'
                    else:
                        current_dst_name = f'{dtype}{k}'

                    expected_relation = f'{tgt_type}{dtype}' if dtype != 'item' else f'{tgt_type}i'
                    if hop + 1 == max_hops and dtype != tgt_type:
                        continue
                    if hop + 2 == max_hops and expected_relation not in adj2etype and dtype != tgt_type:
                        continue
                    if (hop == num_hops and dtype != tgt_type and k not in reserve_heads) \
                      or (hop > num_hops and k not in reserve_heads):
                        continue
                    if echo: print(k, etype, current_dst_name)

                    if not remove_zero_degree_nodes and not remove_zero_degree_features:
                        new_g[etype].update_all(
                            fn.copy_u(k, 'm'),
                            fn.mean('m', current_dst_name), etype=etype)
                    else:
                        if remove_zero_degree_features: # 缺失值不仅针对每个节点（该节点的特征全部缺失），进一步的针对每个特征值（每个节点只有多个特征值是非缺失的，其余均为缺失）
                            new_g.nodes[stype].data['flag'] = (new_g.nodes[stype].data[k] > 1e-6).float()
                        else:
                            new_g.nodes[stype].data['flag'] = (new_g.nodes[stype].data[k].sum(1, keepdim=True) > 1e-6).float() # 注意，目前只可针对 icdm 使用，因为icdm的特征不含负值
                        new_g[etype].update_all(
                            fn.copy_u('flag', 'm'),
                            fn.mean('m', 'num'), etype=etype)

                        new_g[etype].update_all(
                            fn.copy_u(k, 'm'),
                            fn.mean('m', current_dst_name), etype=etype)
                        new_g.nodes[dtype].data[current_dst_name] = \
                            new_g.nodes[dtype].data[current_dst_name] * (new_g.nodes[dtype].data['num'] > 0) \
                            / (new_g.nodes[dtype].data['num'] + 1e-12)

                        new_g.nodes[stype].data.pop('flag')
                        new_g.nodes[dtype].data.pop('num')

        # remove no-use items
        for ntype in new_g.ntypes:
            if ntype == tgt_type: continue
            removes = []
            for k in new_g.nodes[ntype].data.keys():
                if len(k) <= hop:
                    removes.append(k)
            for k in removes:
                new_g.nodes[ntype].data.pop(k)
            if echo and len(removes): print('remove', removes)
        gc.collect()

        if echo: print(f'-- hop={hop} ---')
        for ntype in new_g.ntypes:
            for k, v in new_g.nodes[ntype].data.items():
                if echo: print(f'{ntype} {k} {v.shape}')
        if echo: print(f'------\n')

    return new_g


def generate_features(g, num_hops=1, echo=True, remove_zero_degree_nodes=False, remove_zero_degree_features=False,
    extra_metapath_list=[['bif'], ['fae', 'fea', 'fib', 'faf', 'fcf', 'fdf', 'fef']]):
    extra_metapath = [ele for ele in extra_metapath_list[0] if len(ele) > num_hops + 1]
    print(f'Current num hops = {num_hops}')
    if len(extra_metapath):
        max_hops = max(num_hops + 1, max([len(ele) for ele in extra_metapath]))
    else:
        max_hops = num_hops + 1

    graph = g.clone()
    graph = hg_propagate_icdm22(graph, 'b', num_hops, max_hops, extra_metapath,
        echo, remove_zero_degree_nodes, remove_zero_degree_features)

    feats_b = {}
    keys = list(graph.nodes['b'].data.keys())
    print(f'Involved feat keys {keys}')
    for k in keys:
        feats_b[k] = graph.nodes['b'].data[k].clone()
    del graph

    extra_metapath = [ele for ele in extra_metapath_list[1] if len(ele) > num_hops + 1]
    print(f'Current num hops = {num_hops}')
    if len(extra_metapath):
        max_hops = max(num_hops + 1, max([len(ele) for ele in extra_metapath]))
    else:
        max_hops = num_hops + 1

    graph = g.clone()
    graph = hg_propagate_icdm22(graph, 'f', num_hops, max_hops, extra_metapath, 
        echo, remove_zero_degree_nodes, remove_zero_degree_features)

    feats_f = {}
    keys = list(graph.nodes['f'].data.keys())
    print(f'Involved feat keys {keys}')
    for k in keys:
        feats_f[k] = graph.nodes['f'].data[k].clone()
    del graph
    gc.collect()

    adjs = {}
    for i, etype in enumerate(g.etypes):
        stype, _, dtype = g.to_canonical_etype(etype)
        num_s = g.num_nodes(stype)
        num_d = g.num_nodes(dtype)
        if stype == 'item': stype = 'i'
        if dtype == 'item': dtype = 'i'
        name = f'{dtype}{stype}'
        if name[0] not in ['i', 'b', 'f'] and name[1] not in ['i', 'b', 'f']: continue

        src, dst, eid = g._graph.edges(i)
        adj = SparseTensor(row=dst, col=src, sparse_sizes=(num_d, num_s))
        adjs[name] = adj

    feat_i = g.nodes['item'].data['i'].clone()
    feat_i[torch.LongTensor([13, 97, 103, 139, 183, 215, 248])] = 0
    # feat_i = torch.cat((feat_i, feat_i>0), dim=1)

    # feats_b.pop('bi')
    # feats_f.pop('fi')

    return feat_i, feats_b, feats_f, adjs
