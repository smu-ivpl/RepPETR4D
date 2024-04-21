# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# ------------------------------------------------------------------------
import math
import torch
import torch.nn as nn 
import numpy as np

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats) # sinusoidal transform
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_l = pos[..., 3, None] / dim_t
    pos_w = pos[..., 4, None] / dim_t
    pos_h = pos[..., 5, None] / dim_t
    pos_rot_x = pos[..., 6, None] / dim_t
    pos_rot_y = pos[..., 7, None] / dim_t
    pos_vel_x = pos[..., 8, None] / dim_t
    pos_vel_y = pos[..., 9, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_l = torch.stack((pos_l[..., 0::2].sin(), pos_l[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_w = torch.stack((pos_w[..., 0::2].sin(), pos_w[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_h = torch.stack((pos_h[..., 0::2].sin(), pos_h[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_rot_x = torch.stack((pos_rot_x[..., 0::2].sin(), pos_rot_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_rot_y = torch.stack((pos_rot_y[..., 0::2].sin(), pos_rot_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_vel_x = torch.stack((pos_vel_x[..., 0::2].sin(), pos_vel_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_vel_y = torch.stack((pos_vel_y[..., 0::2].sin(), pos_vel_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z, pos_l, pos_w, pos_h, pos_rot_x, pos_rot_y, pos_vel_x, pos_vel_y), dim=-1)
    return posemb

def pos2posemb1d(pos, num_pos_feats=256, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t

    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)

    return pos_x

def nerf_positional_encoding(
    tensor, num_encoding_functions=6, include_input=False, log_sampling=True
) -> torch.Tensor:
    r"""Apply positional encoding to the input.
    Args:
        tensor (torch.Tensor): Input tensor to be positionally encoded.
        encoding_size (optional, int): Number of encoding functions used to compute
            a positional encoding (default: 6).
        include_input (optional, bool): Whether or not to include the input in the
            positional encoding (default: True).
    Returns:
    (torch.Tensor): Positional encoding of the input tensor.
    """
    # TESTED
    # Trivially, the input tensor is added to the positional encoding.
    encoding = [tensor] if include_input else []
    frequency_bands = None
    if log_sampling:
        frequency_bands = 2.0 ** torch.linspace(
            0.0,
            num_encoding_functions - 1,
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )
    else:
        frequency_bands = torch.linspace(
            2.0 ** 0.0,
            2.0 ** (num_encoding_functions - 1),
            num_encoding_functions,
            dtype=tensor.dtype,
            device=tensor.device,
        )

    for freq in frequency_bands:
        for func in [torch.sin, torch.cos]:
            encoding.append(func(tensor * freq))

    # Special case, for no positional encoding
    if len(encoding) == 1:
        return encoding[0]
    else:
        return torch.cat(encoding, dim=-1)
