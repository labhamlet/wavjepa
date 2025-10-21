from itertools import product
from math import prod

import pytest
import torch
import einops


from sjepa.functions import (
    batched_index_select,
    pos_encode_time,
    pos_encode_continuous,
    batched_index_unselect,
    pos_encode_continuous_batched,
)
from sjepa.masking import complement_index


@pytest.mark.parametrize("bdims", [(), (10,), (10, 11)])
@pytest.mark.parametrize("selected_dim_size", [1, 5])
@pytest.mark.parametrize("xdims", [(), (1,), (3,), (3, 4)])
@pytest.mark.parametrize("idims", [(), (1,), (3,), (3, 4)])
def test_batched_index_select(bdims, selected_dim_size, xdims, idims):
    xshape = bdims + (selected_dim_size,) + xdims
    idxshape = bdims + idims
    x = torch.randn(xshape)
    idx = torch.randint(0, selected_dim_size, idxshape)
    y = batched_index_select(x, idx, selected_dim=len(bdims))
    assert y.shape == idxshape + xdims
    for i_batch in product(*map(range, bdims)):
        for i_idx in product(*map(range, idims)):
            assert (y[i_batch + i_idx] == x[i_batch][idx[i_batch + i_idx]]).all()


@pytest.mark.parametrize("bdims", [(), (10,), (10, 11)])
@pytest.mark.parametrize("selected_dim_size", [1, 5])
@pytest.mark.parametrize("xdims", [(), (1,), (3,), (3, 4)])
def test_batched_index_unselect(bdims, selected_dim_size, xdims):
    xshape = bdims + (selected_dim_size,) + xdims
    bdims_str = " ".join([f"b{i}" for i in range(len(bdims))])
    x = torch.randn(xshape)
    n1 = min(selected_dim_size, 3)
    idx1 = torch.cat(
        [torch.randperm(selected_dim_size)[:n1] for _ in range(prod(bdims))]
    )
    if len(bdims) > 0:
        idx1 = einops.rearrange(
            idx1,
            f"({bdims_str} i) -> {bdims_str} i",
            **{f"b{i}": d for i, d in enumerate(bdims)},
        )
    idx2 = complement_index(idx1, selected_dim_size)
    y1 = batched_index_select(x, idx1, selected_dim=len(bdims))
    y2 = batched_index_select(x, idx2, selected_dim=len(bdims))
    y = torch.cat([y2, y1], dim=len(bdims))
    idx = torch.cat([idx2, idx1], dim=len(bdims))
    y_out = batched_index_unselect(y, idx)
    torch.testing.assert_close(x, y_out)


class TestPosEncodeTime:
    def test_shape(self):
        pos_encoding = pos_encode_time(10, 6, 20)
        assert pos_encoding.shape == (10, 6)

    def test_plot(self):
        import matplotlib.pyplot as plt

        n_times = 100
        n_dim = 100
        max_n_times = 100
        pos_encoding = pos_encode_time(n_times, n_dim, max_n_times)
        plt.imshow(pos_encoding.T, aspect=n_times / n_dim)
        plt.ylabel("embedding dimension")
        plt.xlabel("time")
        plt.colorbar()
        plt.show()


class TestPosEncodeContineous:
    def test_shape(self):
        pos_encoding = pos_encode_continuous(0.5, 0, 1, 6)
        assert pos_encoding.shape == (6,)

    def test_sanity(self):
        torch.testing.assert_close(
            pos_encode_continuous(0.75, 0, 1, 6),
            pos_encode_continuous(0.5, -1, 1, 6),
        )
        torch.testing.assert_close(
            pos_encode_continuous(0.75, 0, 1, 6),
            pos_encode_continuous(-0.25, -1, 0, 6),
        )

    @pytest.mark.parametrize("bdims", [(), (10,), (10, 11)])
    def test_batched(self, bdims):
        locs = torch.rand(bdims)
        out1 = torch.empty(bdims + (6,))
        out2 = torch.empty(bdims + (6,))
        for i_list in product(*map(range, bdims)):
            _ = pos_encode_continuous(locs[i_list], 0, 1, 6, out=out1[i_list])
        _ = pos_encode_continuous_batched(locs, 0, 1, 6, out=out2)
        torch.testing.assert_close(out1, out2)

    def test_plot(self):
        import matplotlib.pyplot as plt

        n_dim = 100
        x0 = -0.001
        x1 = 0.001
        steps = 100
        x = torch.linspace(x0, x1, steps)
        out = torch.empty((len(x), n_dim))
        for i, xx in enumerate(x):
            out[i] = pos_encode_continuous(xx, 0, 10 * x1, n_dim)
        plt.imshow(out.T, aspect=len(x) / n_dim)
        plt.ylabel("embedding dimension")
        plt.xlabel("x")
        plt.colorbar()
        plt.show()
