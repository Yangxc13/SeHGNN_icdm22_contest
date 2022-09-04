import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    "Self attention layer for `n_channels`."
    def __init__(self, n_channels, att_drop=0., act='none'):
        super(Transformer, self).__init__()
        self.query ,self.key, self.value = [self._conv(n_channels, c) for c in (n_channels//8, n_channels//8, n_channels)]
        self.gamma = nn.Parameter(torch.tensor([0.]))
        self.att_drop = nn.Dropout(att_drop)
        if act == 'sigmoid':
            self.act = torch.nn.Sigmoid()
        elif act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky_relu':
            self.act = torch.nn.LeakyReLU(0.2)
        elif act == 'none':
            self.act = lambda x: x
        else:
            assert 0, f'Unrecognized activation function {act} for class Transformer'

    def _conv(self,n_in, n_out):
        return torch.nn.utils.spectral_norm(nn.Conv1d(n_in, n_out, 1, bias=False))

    def forward(self, x, mask=None):
        #Notation from the paper.
        size = x.size()
        x = x.view(*size[:2],-1)
        f, g, h = self.query(x), self.key(x), self.value(x)
        beta = F.softmax(self.act(torch.bmm(f.transpose(1,2), g)), dim=1) # [batch_size, num_metapath(normalized), num_metapath]
        beta = self.att_drop(beta)
        if mask is not None:
            beta = beta * mask.unsqueeze(-1)
            beta = beta / (beta.sum(1, keepdim=True) + 1e-12)
        o = self.gamma * torch.bmm(h, beta) + x
        return o.view(*size).contiguous()


class Conv1d1x1(nn.Module):
    def __init__(self, cin, cout, groups, bias=True, cformat='channel-first'):
        super(Conv1d1x1, self).__init__()
        self.cin = cin
        self.cout = cout
        self.groups = groups
        self.cformat = cformat
        if not bias:
            self.bias = None
        if self.groups == 1: # different keypoints share same kernel
            self.W = nn.Parameter(torch.randn(self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(1, self.cout))
        else:
            self.W = nn.Parameter(torch.randn(self.groups, self.cin, self.cout))
            if bias:
                self.bias = nn.Parameter(torch.zeros(self.groups, self.cout))

    def reset_parameters(self):

        def xavier_uniform_(tensor, gain=1.):
            fan_in, fan_out = tensor.size()[-2:]
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            return torch.nn.init._no_grad_uniform_(tensor, -a, a)

        gain = nn.init.calculate_gain("relu")
        xavier_uniform_(self.W, gain=gain)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.groups == 1:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,mn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,mn->bnc', x, self.W) + self.bias.T
            else:
                assert False
        else:
            if self.cformat == 'channel-first':
                return torch.einsum('bcm,cmn->bcn', x, self.W) + self.bias
            elif self.cformat == 'channel-last':
                return torch.einsum('bmc,cmn->bnc', x, self.W) + self.bias.T
            else:
                assert False


class SeHGNN_mag(nn.Module):
    def __init__(self, dataset, nfeat, hidden, nclass, feats_list, tgt_key,
                 dropout, input_drop, att_drop, n_layers_1, n_layers_2,
                 act, residual=False, bns=False, lr_output=True):
        super(SeHGNN_mag, self).__init__()
        self.dataset = dataset
        self.residual = residual
        self.tgt_key = tgt_key
        num_feats = len(feats_list)

        def add_nonlinear_layers(hidden, dropout, bns='none'):
            layers = []
            assert bns in ['none', 'bn', 'ln']
            if bns == 'bn':
                layers.append(nn.BatchNorm1d(hidden))
            elif bns == 'ln':
                layers.append(nn.LayerNorm(hidden))
            layers.append(nn.PReLU())
            layers.append(nn.Dropout(dropout))
            return layers

        feat_project_layers = [
            Conv1d1x1(nfeat, hidden, num_feats, bias=True, cformat='channel-first')
            ] + add_nonlinear_layers(hidden, dropout, 'ln' if bns else 'none')
        for i in range(1, n_layers_1):
            feat_project_layers += [
                Conv1d1x1(hidden, hidden, num_feats, bias=True, cformat='channel-first')
                ] + add_nonlinear_layers(hidden, dropout, 'ln' if bns else 'none')
        self.feat_project_layers = nn.Sequential(*feat_project_layers)

        self.semantic_aggr_layers = Transformer(hidden, att_drop, act)
        self.concat_project_layer = nn.Linear(num_feats * hidden, hidden)

        if self.residual:
            self.res_fc = nn.Linear(nfeat, hidden, bias=False)

        if lr_output:
            lr_output_layers = [
                nn.Linear(hidden, hidden, bias=not bns)
                ] + add_nonlinear_layers(hidden, dropout, 'bn' if bns else 'none')
            for i in range(1, n_layers_2-1):
                lr_output_layers += [
                    nn.Linear(hidden, hidden, bias=not bns)
                    ] + add_nonlinear_layers(hidden, dropout, 'bn' if bns else 'none')
            lr_output_layers.append(nn.Linear(hidden, nclass, bias=True))
            self.lr_output = nn.Sequential(*lr_output_layers)
        else:
            self.lr_output = None

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")

        for layer in self.feat_project_layers:
            if isinstance(layer, Conv1d1x1) or isinstance(layer, nn.BatchNorm1d) \
                or isinstance(layer, nn.LayerNorm): 
                layer.reset_parameters()

        nn.init.xavier_uniform_(self.concat_project_layer.weight, gain=gain)
        nn.init.zeros_(self.concat_project_layer.bias)

        if self.residual:
            nn.init.xavier_uniform_(self.res_fc.weight, gain=gain)

        if self.lr_output is not None:
            for layer in self.lr_output:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight, gain=gain)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.LayerNorm): 
                    layer.reset_parameters()

    def forward(self, feats_dict):
        tgt_feat = self.input_drop(feats_dict[self.tgt_key])
        B = num_node = tgt_feat.size(0)
        feat_keys = sorted(list(feats_dict.keys()))

        x = [feats_dict[k] for k in feat_keys]
        x = self.input_drop(torch.stack(x, dim=1))
        x = self.feat_project_layers(x)

        x = x.transpose(1,2)
        x = self.semantic_aggr_layers(x, mask=None)
        x = self.concat_project_layer(x.reshape(B, -1))

        if self.residual:
            x = x + self.res_fc(tgt_feat)
        if self.lr_output is not None:
            x = self.dropout(self.prelu(x))
            x = self.lr_output(x)
        return x


class SeHGNN_mix(nn.Module):
    def __init__(self, dataset, nfeat, hidden, nclass, feats_list, tgt_key,
                 dropout, input_drop, att_drop, n_layers_1, n_layers_2,
                 act, residual=False, bns=False):
        super(SeHGNN_mix, self).__init__()

        self.feats_list_b = sorted([k for k in feats_list if k[0] == 'b'])
        self.feats_list_f = sorted([k for k in feats_list if k[0] == 'f'])

        self.model_b = SeHGNN_mag(dataset, nfeat, hidden, nclass, self.feats_list_b, 'b',
            dropout, input_drop, att_drop, n_layers_1, n_layers_2,
            act, residual, bns, lr_output=False)
        self.model_f = SeHGNN_mag(dataset, nfeat, hidden, nclass, self.feats_list_f, 'f',
            dropout, input_drop, att_drop, n_layers_1, n_layers_2,
            act, residual, bns, lr_output=False)

        self.project_i = nn.Sequential(
            nn.Linear(nfeat*2, hidden, bias=True),
            # nn.LayerNorm([hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden, bias=True),
            # nn.LayerNorm([hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
        )
        self.lr_output = nn.Sequential(
            nn.Linear(hidden*3, hidden, bias=True),
            # nn.BatchNorm1d([hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden, bias=True),
            # nn.BatchNorm1d([hidden]),
            nn.PReLU(),
            nn.Dropout(dropout),
        )

        self.prelu = nn.PReLU()
        self.dropout = nn.Dropout(dropout)
        self.input_drop = nn.Dropout(input_drop)

    def reset_parameters(self):
        self.model_b.reset_parameters()
        self.model_f.reset_parameters()

        gain = nn.init.calculate_gain("relu")
        for layer in self.lr_output:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=gain)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.LayerNorm): 
                layer.reset_parameters()

    def forward(self, feats_dict):
        feats_i = self.project_i(self.input_drop(feats_dict['i']))
        feats_b = {k: v for k, v in feats_dict.items() if k[0] == 'b'}
        feats_f = {k: v for k, v in feats_dict.items() if k[0] == 'f'}

        x_b = self.model_b(feats_b)
        x_f = self.model_f(feats_f)

        x = torch.cat((feats_i, x_b, x_f), dim=1)
        x = self.dropout(self.prelu(x))
        x = self.lr_output(x)

        return x
