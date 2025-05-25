# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import torch
from torch import nn
from rfdetr.util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    Standard sine positional embedding for 2D image inputs.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and not normalize:
            raise ValueError("normalize must be True if scale is provided")
        self.scale = scale if scale is not None else 2 * math.pi
        self._export = False
        print(f" PositionEmbeddingSine initialized with num_pos_feats = {num_pos_feats} → final dim = {num_pos_feats * 2}")

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

    def forward(self, tensor_list: NestedTensor, align_dim_orders=True):
        x = tensor_list.tensors
        mask = tensor_list.mask

        # Fallback for empty input
        if mask is None or (isinstance(mask, list) and len(mask) == 0) or \
           (isinstance(x, list) and len(x) == 0):
            print(" Warning: Empty mask or tensor list received in PositionEmbeddingSine. Returning dummy positional encoding.")
            if isinstance(x, list) and len(x) > 0:
                _, H, W = x[0].shape
                B = len(x)
                device = x[0].device
            elif isinstance(x, torch.Tensor):
                B, _, H, W = x.shape
                device = x.device
            else:
                B, H, W = 1, 64, 64
                device = torch.device("cpu")
            dummy = torch.zeros((B, self.num_pos_feats * 2, H, W), device=device)
            return dummy.permute(2, 3, 0, 1) if align_dim_orders else dummy

        if isinstance(mask, list):
            mask = torch.stack(mask, dim=0)

        # Squeeze channel if present
        # Ensure mask is [B, H, W]
        if mask.ndim == 4:
            if mask.shape[1] > 1:
                print(f"⚠️ Mask has {mask.shape[1]} channels, averaging across channel dim.")
                mask = mask.float().mean(dim=1) > 0.5  # Convert to float first
  # Convert to binary mask
            else:
                mask = mask[:, 0]


        assert mask.ndim == 3, f"Expected mask of shape [B, H, W], but got {mask.shape}"


        assert mask.ndim == 3, f"Expected mask of shape [B, H, W], but got {mask.shape}"
        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        device = x.device if isinstance(x, torch.Tensor) else torch.device("cpu")
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # Broadcast-aware safe division
        dim_t = dim_t.view(1, 1, 1, -1)
        x_embed = x_embed.to(dim_t.device)
        y_embed = y_embed.to(dim_t.device)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos.permute(1, 2, 0, 3) if align_dim_orders else pos.permute(0, 3, 1, 2)

    def forward_export(self, mask: torch.Tensor, align_dim_orders=True):
        assert mask is not None
        if mask.ndim == 4:
            mask = mask.squeeze(1)

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = dim_t.view(1, 1, 1, -1)
        
        x_embed = x_embed.to(dim_t.device)
        y_embed = y_embed.to(dim_t.device)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3)
        return pos.permute(1, 2, 0, 3) if align_dim_orders else pos.permute(0, 3, 1, 2)


class PositionEmbeddingLearned(nn.Module):
    """
    Learned absolute positional embedding.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()
        self._export = False

    def export(self):
        raise NotImplementedError

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).unsqueeze(2).repeat(1, 1, x.shape[0], 1)
        return pos  # (H, W, B, C)


def build_position_encoding(hidden_dim, position_embedding):
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        return PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ('v3', 'learned'):
        return PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported position embedding: {position_embedding}")


__all__ = ["build_position_encoding", "PositionEmbeddingSine", "PositionEmbeddingLearned"]
