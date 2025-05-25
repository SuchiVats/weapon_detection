# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
"""
Full implementation of TransformerDecoderLayer, including class wrapper for TransformerDecoder.
"""

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional
from rfdetr.models.attention import MultiheadAttention
from rfdetr.models.ops.modules import MSDeformAttn
import copy

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model=256, sa_nhead=8, ca_nhead=8, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, group_detr=1,
                 num_feature_levels=4, dec_n_points=4, skip_self_attn=False):
        super().__init__()
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=sa_nhead, dropout=dropout, batch_first=True)
        self.cross_attn = MSDeformAttn(d_model, n_levels=num_feature_levels, n_heads=ca_nhead, n_points=dec_n_points)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.group_detr = group_detr

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, *args, **kwargs):
        tgt = args[0]
        memory = args[1]
        tgt_mask = kwargs.get("tgt_mask", None)
        memory_mask = kwargs.get("memory_mask", None)
        tgt_key_padding_mask = kwargs.get("tgt_key_padding_mask", None)
        memory_key_padding_mask = kwargs.get("memory_key_padding_mask", None)
        pos = kwargs.get("pos", None)
        query_pos = kwargs.get("query_pos", None)
        query_sine_embed = kwargs.get("query_sine_embed", None)
        is_first = kwargs.get("is_first", False)
        reference_points = kwargs.get("reference_points", None)
        spatial_shapes = kwargs.get("spatial_shapes", None)
        level_start_index = kwargs.get("level_start_index", None)

        bs, num_queries, _ = tgt.shape
        q = k = self.with_pos_embed(tgt, query_pos)
        v = tgt
        if self.training:
            q = torch.cat(q.split(num_queries // self.group_detr, dim=1), dim=0)
            k = torch.cat(k.split(num_queries // self.group_detr, dim=1), dim=0)
            v = torch.cat(v.split(num_queries // self.group_detr, dim=1), dim=0)

        tgt2 = self.self_attn(q, k, v, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)[0]
        if self.training:
            tgt2 = torch.cat(tgt2.split(bs, dim=0), dim=1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos), reference_points, memory,
                               spatial_shapes, level_start_index, memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False,
                 d_model=256, lite_refpoint_refine=False, bbox_reparam=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        self.bbox_embed = MLP(d_model, d_model, 4, 3)
        self.bbox_reparam = bbox_reparam
        self.lite_refpoint_refine = lite_refpoint_refine
        self._export = False

    def export(self):
        self._export = True

    def refpoints_refine(self, refpoints_unsigmoid, new_refpoints_delta):
        if self.bbox_reparam:
            new_refpoints_cxcy = new_refpoints_delta[..., :2] * refpoints_unsigmoid[..., 2:] + refpoints_unsigmoid[..., :2]
            new_refpoints_wh = new_refpoints_delta[..., 2:].exp() * refpoints_unsigmoid[..., 2:]
            return torch.cat([new_refpoints_cxcy, new_refpoints_wh], dim=-1)
        else:
            return refpoints_unsigmoid + new_refpoints_delta

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, refpoints_unsigmoid=None, level_start_index=None,
                spatial_shapes=None, valid_ratios=None):
        output = tgt
        intermediate = []
        hs_refpoints_unsigmoid = [refpoints_unsigmoid]

        def get_reference(refpoints):
            obj_center = refpoints[..., :4]
            if self._export:
                query_sine_embed = torch.sin(obj_center)
                refpoints_input = obj_center[:, :, None]
            else:
                refpoints_input = obj_center[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[:, None]
                query_sine_embed = torch.sin(refpoints_input[:, :, 0, :])
            query_pos = self.ref_point_head(query_sine_embed)
            return obj_center, refpoints_input, query_pos, query_sine_embed

        if self.lite_refpoint_refine:
            refpoints_input = refpoints_unsigmoid if self.bbox_reparam else refpoints_unsigmoid.sigmoid()
            obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_input)

        for layer_id, layer in enumerate(self.layers):
            if not self.lite_refpoint_refine:
                refpoints_input = refpoints_unsigmoid if self.bbox_reparam else refpoints_unsigmoid.sigmoid()
                obj_center, refpoints_input, query_pos, query_sine_embed = get_reference(refpoints_input)

            output = layer(output, memory,
                           tgt_mask=tgt_mask, memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos, query_sine_embed=query_sine_embed, is_first=(layer_id == 0),
                           reference_points=refpoints_input, spatial_shapes=spatial_shapes,
                           level_start_index=level_start_index)

            if not self.lite_refpoint_refine:
                new_refpoints_delta = self.bbox_embed(output)
                refpoints_unsigmoid = self.refpoints_refine(refpoints_unsigmoid, new_refpoints_delta).detach()
                if layer_id != self.num_layers - 1:
                    hs_refpoints_unsigmoid.append(refpoints_unsigmoid)

            if self.return_intermediate:
                intermediate.append(self.norm(output) if self.norm else output)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(hs_refpoints_unsigmoid)
        return output.unsqueeze(0), refpoints_unsigmoid.unsqueeze(0)

DecoderLayer = TransformerDecoderLayer
Decoder = TransformerDecoder