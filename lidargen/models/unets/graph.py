#!/usr/bin/python
#
# Copyright 2018 Google LLC
# Modification copyright 2021 Helisa Dhamo, Fabian Manhardt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
from .layers import build_mlp

"""
PyTorch modules for dealing with scene graphs.
"""


def make_mlp(dim_list, activation='relu', batch_norm='none', dropout=0, norelu=False):
  return build_mlp(dim_list, activation, batch_norm, dropout, final_nonlinearity=(not norelu))


def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)


class WeightNetGCN(nn.Module):
    """ predict a weight array for the subject and the objects """
    def __init__(self, feat_dim_in1=256, feat_dim_in2=256, feat_dim=128, separate_s_o=True):
        super(WeightNetGCN, self).__init__()

        self.separate = separate_s_o

        if self.separate:
            self.Net_s = nn.Sequential(
                nn.Linear(3*feat_dim, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

            self.Net_o = nn.Sequential(
                    nn.Linear(3*feat_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                    )
        else:
            self.Net = nn.Sequential(
                    nn.Linear(3*feat_dim, 64),
                    nn.ReLU(inplace=True),
                    nn.Linear(64, 1),
                    nn.Sigmoid()
                    )

        self.down_sample_obj = nn.Linear(feat_dim_in1, feat_dim)
        self.down_sample_pred = nn.Linear(feat_dim_in2, feat_dim)

    def forward(self, s, p, o):

        s = self.down_sample_obj(s)
        p = self.down_sample_pred(p)
        o = self.down_sample_obj(o)

        if self.separate:
            feat1 = torch.cat([s, o, p], 1)
            w_s = self.Net_s(feat1)

            feat2 = torch.cat([s, o, p], 1)
            w_o = self.Net_o(feat2)
        else:
            feat = torch.cat([s, o, p], 1)
            w_o = self.Net(feat)
            w_s = w_o

        return w_s, w_o


class GraphTripleConv(nn.Module):
    """
    A single layer of scene graph convolution.
    """
    def __init__(self, input_dim_obj, input_dim_pred, output_dim=None, hidden_dim=512,
                             pooling='avg', mlp_normalization='none', residual=True):
        super(GraphTripleConv, self).__init__()
        if output_dim is None:
            output_dim = input_dim_obj
        self.input_dim_obj = input_dim_obj
        self.input_dim_pred = input_dim_pred
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.residual = residual

        assert pooling in ['sum', 'avg', 'wAvg'], 'Invalid pooling "%s"' % pooling

        self.pooling = pooling
        net1_layers = [2 * input_dim_obj + input_dim_pred, hidden_dim, 2 * hidden_dim + input_dim_pred]
        net1_layers = [l for l in net1_layers if l is not None]
        self.net1 = build_mlp(net1_layers, batch_norm=mlp_normalization)
        self.net1.apply(_init_weights)

        net2_layers = [hidden_dim, hidden_dim, output_dim]
        self.net2 = build_mlp(net2_layers, batch_norm=mlp_normalization)
        self.net2.apply(_init_weights)

        if self.residual:
            self.linear_projection = nn.Linear(input_dim_obj, output_dim)
            self.linear_projection_pred = nn.Linear(input_dim_pred, input_dim_pred) # TODO is there any better option?

        if self.pooling == 'wAvg':
            self.weightNet = WeightNetGCN(hidden_dim, output_dim, 128)

    def forward(self, obj_vecs, pred_vecs, edges):
        """
        Inputs:
        - obj_vecs: FloatTensor of shape (num_objs, D) giving vectors for all objects
        - pred_vecs: FloatTensor of shape (num_triples, D) giving vectors for all predicates
        - edges: LongTensor of shape (num_triples, 2) where edges[k] = [i, j] indicates the
            presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

        Outputs:
        - new_obj_vecs: FloatTensor of shape (num_objs, D) giving new vectors for objects
        - new_pred_vecs: FloatTensor of shape (num_triples, D) giving new vectors for predicates
        """

        dtype, device = obj_vecs.dtype, obj_vecs.device
        num_objs, num_triples = obj_vecs.size(0), pred_vecs.size(0)
        Din_obj, Din_pred, H, Dout = self.input_dim_obj, self.input_dim_pred, self.hidden_dim, self.output_dim

        # Break apart indices for subjects and objects; these have shape (num_triples,)
        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()

        # Get current vectors for subjects and objects; these have shape (num_triples, Din)
        cur_s_vecs = obj_vecs[s_idx]
        cur_o_vecs = obj_vecs[o_idx]

        # Get current vectors for triples; shape is (num_triples, 3 * Din)
        # Pass through net1 to get new triple vecs; shape is (num_triples, 2 * H + Dout)
        cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        new_t_vecs = self.net1(cur_t_vecs)

        # Break apart into new s, p, and o vecs; s and o vecs have shape (num_triples, H) and
        # p vecs have shape (num_triples, Dout)
        new_s_vecs = new_t_vecs[:, :H]
        new_p_vecs = new_t_vecs[:, H:(H+Din_pred)]
        new_o_vecs = new_t_vecs[:, (H+Din_pred):]
 
        # Allocate space for pooled object vectors of shape (num_objs, H)
        pooled_obj_vecs = torch.zeros(num_objs, H, dtype=dtype, device=device)

        if self.pooling == 'wAvg':

            s_weights, o_weights = self.weightNet(new_s_vecs.detach(),
                                                  new_p_vecs.detach(),
                                                  new_o_vecs.detach())

            new_s_vecs = s_weights * new_s_vecs
            new_o_vecs = o_weights * new_o_vecs

        # Use scatter_add to sum vectors for objects that appear in multiple triples;
        # we first need to expand the indices to have shape (num_triples, D)
        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)
        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        if self.pooling == 'wAvg':
            pooled_weight_sums = torch.zeros(num_objs, 1, dtype=dtype, device=device)
            pooled_weight_sums = pooled_weight_sums.scatter_add(0, o_idx.view(-1, 1), o_weights)
            pooled_weight_sums = pooled_weight_sums.scatter_add(0, s_idx.view(-1, 1), s_weights)

            pooled_obj_vecs = pooled_obj_vecs / (pooled_weight_sums + 0.0001)

        if self.pooling == 'avg':
            # Figure out how many times each object has appeared, again using
            # some scatter_add trickery.
            obj_counts = torch.zeros(num_objs, dtype=dtype, device=device)
            ones = torch.ones(num_triples, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)

            # Divide the new object vectors by the number of times they
            # appeared, but first clamp at 1 to avoid dividing by zero;
            # objects that appear in no triples will have output vector 0
            # so this will not affect them.
            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)

        # Send pooled object vectors through net2 to get output object vectors,
        # of shape (num_objs, Dout)
        new_obj_vecs = self.net2(pooled_obj_vecs)

        if self.residual:
            projected_obj_vecs = self.linear_projection(obj_vecs)
            new_obj_vecs = new_obj_vecs + projected_obj_vecs
            # new
            new_p_vecs = new_p_vecs + self.linear_projection_pred(pred_vecs)

        return new_obj_vecs, new_p_vecs


class GraphTripleConvNet(nn.Module):
    """ A sequence of scene graph convolution layers    """
    def __init__(self, input_dim_obj, input_dim_pred, num_layers=2, hidden_dim=512,
                             residual=False, pooling='avg',
                             mlp_normalization='none', output_dim=None):
        super(GraphTripleConvNet, self).__init__()

        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
            'input_dim_obj': input_dim_obj,
            'input_dim_pred': input_dim_pred,
            'hidden_dim': hidden_dim,
            'pooling': pooling,
            'residual': residual,
            'mlp_normalization': mlp_normalization,
        }
        gconv_kwargs_out = {
            'input_dim_obj': input_dim_obj,
            'input_dim_pred': input_dim_pred,
            'hidden_dim': hidden_dim,
            'pooling': pooling,
            'residual': residual,
            'mlp_normalization': mlp_normalization,
            'output_dim': output_dim
        }
        for i in range(self.num_layers):
            if output_dim is not None and i >=  self.num_layers - 1:
                self.gconvs.append(GraphTripleConv(**gconv_kwargs_out))
            else:
                self.gconvs.append(GraphTripleConv(**gconv_kwargs))

    def forward(self, obj_vecs, pred_vecs, edges):
        for i in range(self.num_layers):
            gconv = self.gconvs[i]
            obj_vecs, pred_vecs = gconv(obj_vecs, pred_vecs, edges)
        return obj_vecs, pred_vecs