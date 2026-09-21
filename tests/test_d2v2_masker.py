"""``D2v2BlockMasker`` (wavjepa/masking.py) against fairseq's ``compute_block_mask_1d`` (data2vec 2.0).

The reference below is fairseq's function (fairseq/data/data_utils.py, MIT) restricted to the options the released
speech recipe uses (non_overlapping=False, expand_adjcent=False, mask_dropout=0); it draws from the global torch RNG,
so the comparison is statistical (per-position mask rate, masked count, number of masked runs). CPU, tiny.
"""
from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from wavjepa.masking import D2v2BlockMasker  # noqa: E402

torch.set_num_threads(min(4, torch.get_num_threads()))


def fairseq_compute_block_mask_1d(shape, mask_prob, mask_length, mask_prob_adjust=0.0, inverse_mask=False,
                                  require_same_masks=True):
    """Verbatim reference (fairseq ``compute_block_mask_1d``; non_overlapping / expand_adjcent / mask_dropout off)."""
    B, L = shape
    if inverse_mask:
        mask_prob = 1 - mask_prob
    mask = torch.zeros((B, L))
    mask_inds = torch.randint(0, L, size=(B, int(L * ((mask_prob + mask_prob_adjust) / mask_length))))
    mask.view(B, -1).scatter_(1, mask_inds, 1)
    centers = mask.nonzero(as_tuple=True)
    inds = ([], [])
    offset = mask_length // 2
    for i in range(mask_length):
        k1 = i - offset
        inds[0].append(centers[0])
        inds[1].append(centers[1] + k1)
    i0 = torch.cat(inds[0])
    i1 = torch.cat(inds[1]).clamp_(min=0, max=L - 1)
    mask[(i0, i1)] = 1
    mask = mask.view(B, -1)
    if require_same_masks:
        n_masks = mask.sum(dim=-1)
        final_target_len = int(L * mask_prob)
        for i in range(len(mask)):
            n = n_masks[i]
            m = mask[i]
            if n > final_target_len:
                to_unmask = torch.multinomial(m, int(n - final_target_len), replacement=False)
                m[to_unmask] = 0
            elif n < final_target_len:
                to_mask = torch.multinomial((1 - m), int(final_target_len - n), replacement=False)
                m[to_mask] = 1
    if inverse_mask:
        mask = 1 - mask
    return mask


def _runs(row: torch.Tensor) -> list[tuple[int, int]]:
    """(start, length) of every maximal run of True in a 1-D bool tensor."""
    out, start = [], None
    for i, v in enumerate(row.tolist() + [False]):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - start)); start = None
    return out


def test_exact_masked_count_and_shapes():
    S = 201
    for inverse, expect_masked in ((False, int(S * 0.4)), (True, S - int(S * 0.6))):
        m = D2v2BlockMasker(mask_prob=0.4, mask_length=10, mask_prob_adjust=0.05, inverse_mask=inverse)
        ctx, tgt, ctx_tgt = m.sample(64, S, in_channels=1, device="cpu", generator=torch.Generator().manual_seed(0))
        assert ctx.shape == (64, S) and tgt.shape == (64, 1, S) and ctx_tgt.shape == (64, 1, S)
        assert ctx.dtype == torch.bool and tgt.dtype == torch.bool and ctx_tgt.dtype == torch.bool
        assert torch.equal(tgt[:, 0], ctx) and not ctx_tgt.any()
        assert (ctx.sum(dim=1) == expect_masked).all(), (inverse, ctx.sum(dim=1)[:5])


def test_paper_speech_setting_masks_half():
    """R = 0.5, B = 10 (100 ms at 100 Hz), A = 0.05 on a 200-token crop: exactly 100 masked, blocks of 10."""
    m = D2v2BlockMasker(mask_prob=0.5, mask_length=10, mask_prob_adjust=0.05)
    ctx, _, _ = m.sample(256, 200, device="cpu", generator=torch.Generator().manual_seed(1))
    assert (ctx.sum(dim=1) == 100).all()
    # int(200 * 0.55 / 10) = 11 centres of width 10 -> up to 110 masked before the adjustment, so the typical row
    # is trimmed, never padded: most masked positions still sit in long runs
    lengths = torch.tensor([ln for row in ctx[:32] for _, ln in _runs(row)])
    assert lengths.float().mean() > 4.0


def test_blocks_before_adjustment_are_windows_of_mask_length():
    """Without the per-row adjustment every interior run of masked positions is at least one block wide."""
    m = D2v2BlockMasker(mask_prob=0.5, mask_length=10, mask_prob_adjust=0.05, require_same_masks=False)
    ctx = m.block_mask(64, 200, torch.device("cpu"), torch.Generator().manual_seed(2))
    for row in ctx:
        for start, length in _runs(row):
            if start > 0 and start + length < 200:          # runs touching an edge may be clamped shorter
                assert length >= 10, (start, length)


def test_determinism_and_device_generator():
    m = D2v2BlockMasker(mask_prob=0.5, mask_length=10, mask_prob_adjust=0.05)
    a = m.sample(8, 200, device="cpu", generator=torch.Generator().manual_seed(3))[0]
    b = m.sample(8, 200, device="cpu", generator=torch.Generator().manual_seed(3))[0]
    c = m.sample(8, 200, device="cpu", generator=torch.Generator().manual_seed(4))[0]
    assert torch.equal(a, b) and not torch.equal(a, c)
    f = m(8, 200, 1)[0]                                       # legacy forward = sample on CPU
    assert f.shape == (8, 200) and (f.sum(dim=1) == 100).all()


def test_channel_based_masking_layout():
    m = D2v2BlockMasker(mask_prob=0.5, mask_length=10, channel_based_masking=True)
    ctx, tgt, ctx_tgt = m.sample(4, 400, in_channels=2, device="cpu", generator=torch.Generator().manual_seed(5))
    assert ctx.shape == (4, 400) and tgt.shape == (4, 1, 400)
    assert torch.equal(ctx[:, 0::2], ctx[:, 1::2])              # (S C) layout: the two channels share the mask


def test_matches_fairseq_reference_statistics():
    """Same per-position mask rate, masked count and number of masked runs as fairseq (2 % relative / 5 sigma)."""
    torch.manual_seed(0)
    R, B_, A, S, N = 0.5, 10, 0.05, 200, 4000
    for inverse in (False, True):
        ref = fairseq_compute_block_mask_1d((N, S), R, B_, A, inverse_mask=inverse).bool()
        ours = D2v2BlockMasker(R, B_, A, inverse_mask=inverse).sample(
            N, S, device="cpu", generator=torch.Generator().manual_seed(11 + int(inverse)))[0]
        assert torch.equal(ref.sum(dim=1), ours.sum(dim=1))    # exact per-row count is deterministic
        rate_ref, rate_ours = ref.float().mean(dim=0), ours.float().mean(dim=0)
        assert (rate_ref - rate_ours).abs().max() < 0.05, (rate_ref - rate_ours).abs().max()
        runs_ref = torch.tensor([len(_runs(r)) for r in ref[:600]]).float()
        runs_ours = torch.tensor([len(_runs(r)) for r in ours[:600]]).float()
        se = (runs_ref.var() / 600 + runs_ours.var() / 600).sqrt()
        assert (runs_ref.mean() - runs_ours.mean()).abs() < max(5 * se, 0.02 * runs_ref.mean()), \
            (inverse, runs_ref.mean(), runs_ours.mean())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("ok", name)
