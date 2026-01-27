import collections.abc
from itertools import repeat

import torch
import torch.nn as nn
import torch.nn.functional as F

from .droppath import DropPath
from .swin import Mlp


def constant_init(tensor, constant=0.0):
    nn.init.constant_(tensor, constant)
    return tensor


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


class Mlp(nn.Module):
    def __init__(
        self,
        in_features=None,
        hidden_features=None,
        out_features=None,
        activation=F.gelu,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = activation
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, train: bool = True):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x) if train else x
        x = self.fc2(x)
        x = self.drop(x) if train else x
        return x


class Attention(nn.Module):
    """
    Default multihead attention
    """

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)

    def forward(self, x, train: bool = True):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) if train else attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x) if train else x
        return x


def window_partition1d(x, window_size):
    B, W, C = x.shape
    x = x.view(B, W // window_size, window_size, C)
    windows = x.view(-1, window_size, C)
    return windows


def window_reverse1d(windows, window_size, W: int):
    B = int(windows.shape[0] / (W / window_size))
    x = windows.view(B, W // window_size, window_size, -1)
    x = x.view(B, W, -1)
    return x


def get_relative_position_index1d(win_w):
    # get pair-wise relative position index for each token inside the window
    coords = torch.stack(torch.meshgrid(torch.arange(win_w)))

    relative_coords = coords[:, :, None] - coords[:, None, :]  # 1, Ww, Ww
    relative_coords = relative_coords.permute(1, 2, 0)  # Ww, Ww, 1

    relative_coords[:, :, 0] += win_w - 1  # shift to start from 0

    return relative_coords.sum(-1)  # Ww*Ww


class WindowedAttentionHead(nn.Module):
    def __init__(self, head_dim, window_size, shift_windows=False, attn_drop=0.0):
        super().__init__()
        self.head_dim = head_dim
        self.window_size = window_size
        self.shift_windows = shift_windows
        self.attn_drop = attn_drop

        self.scale = self.head_dim**-0.5
        self.window_area = self.window_size * 1

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1, 1))
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Get relative position index
        self.register_buffer(
            "relative_position_index", get_relative_position_index1d(window_size)
        )

        self.drop_layer = nn.Dropout(attn_drop) if attn_drop > 0 else None

        if shift_windows:
            self.shift_size = window_size // 2
        else:
            self.shift_size = 0
        assert 0 <= self.shift_size < self.window_size, (
            "shift_size must in 0-window_size"
        )

    def forward(self, q, k, v, train: bool = True):
        B, W, C = q.shape

        mask = None
        if self.shift_size > 0:
            img_mask = torch.zeros((1, W, 1), device=q.device)
            cnt = 0
            for w in (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            ):
                img_mask[:, w, :] = cnt
                cnt += 1
            mask_windows = window_partition1d(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size)
            mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            mask = mask.masked_fill(mask != 0, -100.0).masked_fill(mask == 0, 0.0)

            q = torch.roll(q, shifts=-self.shift_size, dims=1)
            k = torch.roll(k, shifts=-self.shift_size, dims=1)
            v = torch.roll(v, shifts=-self.shift_size, dims=1)

        q = window_partition1d(q, self.window_size)
        k = window_partition1d(k, self.window_size)
        v = window_partition1d(v, self.window_size)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if train:
            attn = attn + self._get_rel_pos_bias()
        else:
            attn = attn + self._get_rel_pos_bias()

        if mask is not None:
            B_, N, _ = attn.shape
            num_win = mask.shape[0]
            attn = attn.view(B_ // num_win, num_win, N, N) + mask.unsqueeze(0)
            attn = attn.view(-1, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)

        if self.drop_layer is not None and train:
            attn = self.drop_layer(attn)

        x = attn @ v

        # merge windows
        shifted_x = window_reverse1d(x, self.window_size, W=W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x

        return x, attn

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_area, self.window_area, -1)  # Ww,Ww,1
        relative_position_bias = relative_position_bias.permute(2, 0, 1)  # 1, Ww, Ww
        return relative_position_bias


class AttentionHead(nn.Module):
    def __init__(self, head_dim, attn_drop=0.0):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim**-0.5
        self.drop_layer = nn.Dropout(attn_drop) if attn_drop > 0 else None

    def forward(self, q, k, v, train: bool = True):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        if self.drop_layer is not None and train:
            attn = self.drop_layer(attn)

        x = attn @ v
        return x, attn


class WindowedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim,
        window_sizes,
        shift_windows=False,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        nn.init.xavier_uniform_(self.qkv.weight)

        if isinstance(window_sizes, int):
            window_sizes = _ntuple(num_heads)(window_sizes)
        else:
            assert len(window_sizes) == num_heads

        self.attn_heads = nn.ModuleList()
        for i in range(num_heads):
            ws_i = window_sizes[i]
            if ws_i == 0:
                self.attn_heads.append(AttentionHead(self.head_dim, attn_drop))
            else:
                self.attn_heads.append(
                    WindowedAttentionHead(
                        self.head_dim,
                        window_size=ws_i,
                        shift_windows=shift_windows,
                        attn_drop=attn_drop,
                    )
                )

        self.proj = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.proj.weight)
        self.drop_layer = nn.Dropout(proj_drop) if proj_drop > 0 else None

    def forward(self, x, train: bool = True):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 3, 0, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        o = []
        for i in range(self.num_heads):
            head_i, attn_i = self.attn_heads[i](q[i], k[i], v[i], train=train)
            o.append(head_i.unsqueeze(0))

        o = torch.cat(o, dim=0)
        o = o.permute(1, 2, 0, 3).reshape(B, N, -1)
        o = self.proj(o)

        if self.drop_layer is not None and train:
            o = self.drop_layer(o)

        return o


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma


class BNWrapper(nn.Module):
    def __init__(
        self, num_features, use_running_average=True, use_bias=True, use_scale=True
    ):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features, affine=use_scale or use_bias)

    def forward(self, x, train=True):
        return self.bn(x, train)


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=F.gelu,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            activation=act_layer,
            drop=drop,
        )

        self.init_values = init_values
        if init_values is not None:
            self.layer_scale1 = LayerScale(dim, init_values)
            self.layer_scale2 = LayerScale(dim, init_values)

    def forward(self, x, train: bool = True):
        outputs1 = self.attn(self.norm1(x), train=train)

        if self.init_values is not None:
            outputs1 = self.layer_scale1(outputs1)

        x = x + self.drop_path(outputs1) if train else x + outputs1

        outputs2 = self.mlp(self.norm2(x), train=train)

        if self.init_values is not None:
            outputs2 = self.layer_scale2(outputs2)

        x = x + self.drop_path(outputs2) if train else x + outputs2
        return x


class MWMHABlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_sizes,
        shift_windows=False,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=F.gelu,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.wmha = WindowedMultiHeadAttention(
            dim,
            window_sizes=window_sizes,
            shift_windows=shift_windows,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            activation=act_layer,
            drop=drop,
        )

        self.init_values = init_values
        if init_values is not None:
            self.layer_scale1 = LayerScale(dim, init_values)
            self.layer_scale2 = LayerScale(dim, init_values)

    def forward(self, x, train: bool = True):
        outputs1 = self.wmha(self.norm1(x), train=train)

        if self.init_values is not None:
            outputs1 = self.layer_scale1(outputs1)

        x = x + self.drop_path(outputs1) if train else x + outputs1

        outputs2 = self.mlp(self.norm2(x), train=train)

        if self.init_values is not None:
            outputs2 = self.layer_scale2(outputs2)

        x = x + self.drop_path(outputs2) if train else x + outputs2
        return x
