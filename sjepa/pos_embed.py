# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# Position embedding utils
# --------------------------------------------------------


# https://github.com/facebookresearch/AudioMAE/blob/main/util/pos_embed.py
import numpy as np
import torch


# --------------------------------------------------------
# 2D sine-cosine position embedding
# References:
# Transformer: https://github.com/tensorflow/models/blob/master/official/nlp/transformer/model_utils.py
# MoCo v3: https://github.com/facebookresearch/moco-v3
# --------------------------------------------------------
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token_num):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    if grid_size is int:
        gH = grid_size
        gW = grid_size
    else:
        gH = grid_size[0]
        gW = grid_size[1]
    grid_h = np.arange(gH, dtype=np.float64)
    grid_w = np.arange(gW, dtype=np.float64)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, gH, gW])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    for _ in range(cls_token_num):
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_flexible(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float64)
    grid_w = np.arange(grid_size[1], dtype=np.float64)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_1d_sincos_pos_embed(embed_dim, length):
    """
    Create 1D sinusoidal positional embeddings.

    Args:
        embed_dim: embedding dimension
        length: sequence length

    Returns:
        pos_embed: [length, embed_dim]
    """
    assert embed_dim % 2 == 0

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = np.arange(length, dtype=np.float64)  # (length,)
    out = np.einsum("m,d->md", pos, omega)  # (length, D/2)

    emb_sin = np.sin(out)  # (length, D/2)
    emb_cos = np.cos(out)  # (length, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (length, D)
    return emb


def get_binaural_pos_embed(embed_dim, time_steps=100):
    """
    Create positional embeddings for binaural audio.
    Same time encoding, different channel encoding.

    Args:
        embed_dim: embedding dimension
        time_steps: number of time steps per channel

    Returns:
        pos_embed: [2*time_steps, embed_dim] - for concatenated L+R channels
    """
    assert embed_dim % 2 == 0

    # Time dimension encoding (same for both channels)
    time_embed = get_1d_sincos_pos_embed(embed_dim // 2, time_steps)

    # Channel dimension encoding (different for L and R)
    channel_embed_left = np.zeros((time_steps, embed_dim // 2))  # Left channel = 0
    channel_embed_right = get_1d_sincos_pos_embed(
        embed_dim // 2, 1
    )  # Right channel = different
    channel_embed_right = np.tile(channel_embed_right, (time_steps, 1))

    # Combine time and channel embeddings
    left_pos_embed = np.concatenate([time_embed, channel_embed_left], axis=1)
    right_pos_embed = np.concatenate([time_embed, channel_embed_right], axis=1)

    # Concatenate left and right channel embeddings
    binaural_pos_embed = np.concatenate([left_pos_embed, right_pos_embed], axis=0)

    return binaural_pos_embed


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches**0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size, orig_size, new_size, new_size)
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size, orig_size, embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size, new_size),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def interpolate_pos_embed_img2audio(model, checkpoint_model, orig_size, new_size):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        # orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        # new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size[0], orig_size[1], new_size[0], new_size[1])
            )
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(
                -1, orig_size[0], orig_size[1], embedding_size
            ).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens,
                size=(new_size[0], new_size[1]),
                mode="bicubic",
                align_corners=False,
            )
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def interpolate_pos_embed_audio(model, checkpoint_model, orig_size, new_size):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1]
        if orig_size != new_size:
            print(
                "Position interpolate from %dx%d to %dx%d"
                % (orig_size[0], orig_size[1], new_size[0], new_size[1])
            )
            # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            cls_token = pos_embed_checkpoint[:, 0, :].unsqueeze(1)
            pos_tokens = pos_embed_checkpoint[:, 1:, :]  # remove
            pos_tokens = pos_tokens.reshape(
                -1, orig_size[0], orig_size[1], embedding_size
            )  # .permute(0, 3, 1, 2)
            # pos_tokens = torch.nn.functional.interpolate(
            #    pos_tokens, size=(new_size[0], new_size[1]), mode='bicubic', align_corners=False)

            # pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            pos_tokens = pos_tokens[:, :, : new_size[1], :]  # assume only time diff
            pos_tokens = pos_tokens.flatten(1, 2)
            new_pos_embed = torch.cat((cls_token, pos_tokens), dim=1)
            checkpoint_model["pos_embed"] = new_pos_embed


def interpolate_patch_embed_audio(
    model,
    checkpoint_model,
    orig_channel,
    new_channel=1,
    kernel_size=(16, 16),
    stride=(16, 16),
    padding=(0, 0),
):
    if orig_channel != new_channel:
        if "patch_embed.proj.weight" in checkpoint_model:
            # aggregate 3 channels in rgb ckpt to 1 channel for audio
            new_proj_weight = torch.nn.Parameter(
                torch.sum(checkpoint_model["patch_embed.proj.weight"], dim=1).unsqueeze(
                    1
                )
            )
            checkpoint_model["patch_embed.proj.weight"] = new_proj_weight
