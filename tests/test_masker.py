"""
Tests for ``TimeInverseBlockMasker.sample`` (vectorised) against ``forward`` (loop reference).

pytest-style, but also runnable as ``python tests/test_masker.py`` (pytest is optional).
CPU only, tiny. The loop reference cannot be seeded (``compute_mask_indices`` uses
``np.random.default_rng(None)``), so the statistical tests are genuinely random on that
side. The AudioSet comparison uses the 2 % relative rule from DESIGN.md with row counts
that make it > 3.5 sigma; the edge cases use standard-error based (5 sigma) bounds and,
where the distribution is derivable by hand, exact values.
"""
from __future__ import annotations

import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from wavjepa.masking import TimeInverseBlockMasker  # noqa: E402

try:  # pytest is optional
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None

torch.set_num_threads(min(4, torch.get_num_threads()))

AUDIOSET_CFG = dict(
    target_masks_per_context=4,
    context_mask_prob=0.65,
    context_mask_length=10,
    target_prob=0.25,
    target_length=10,
    ratio_cutoff=0.1,
    channel_based_masking=False,
)
S = 200
# loop reference is ~1 ms/row (S=200) on the login node; 20k rows ~ 20 s
REF_ROWS = int(os.environ.get("WAVJEPA_MASKER_REF_ROWS", "20000"))
VEC_ROWS = int(os.environ.get("WAVJEPA_MASKER_VEC_ROWS", "100000"))
REL_TOL = 0.02
N_SIGMA = 5.0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _gen(seed: int = 0) -> torch.Generator:
    return torch.Generator(device="cpu").manual_seed(seed)


def _runs(mask: torch.Tensor):
    """mask (..., S) bool -> (n_runs (...,), run_starts (K,), run_lengths (K,)) of True runs."""
    S_ = mask.shape[-1]
    flat = mask.reshape(-1, S_)
    zeros = torch.zeros(flat.shape[0], 1, dtype=torch.bool)
    starts = flat & ~torch.cat([zeros, flat[:, :-1]], dim=1)
    ends = flat & ~torch.cat([flat[:, 1:], zeros], dim=1)
    n_runs = starts.sum(1).reshape(mask.shape[:-1])
    s_row, s_pos = torch.nonzero(starts, as_tuple=True)
    e_row, e_pos = torch.nonzero(ends, as_tuple=True)
    assert torch.equal(s_row, e_row), "run start/end bookkeeping broken"
    return n_runs, s_pos, e_pos - s_pos + 1


def _check_invariants(ctx, tgt, ctx_tgt, masker: TimeInverseBlockMasker, B: int, S_: int,
                      in_channels: int = 1):
    """Structural invariants shared by forward and sample."""
    N = masker.target_masks_per_context
    S_out = S_ * in_channels if masker.channel_based_masking else S_
    assert ctx.shape == (B, S_out) and ctx.dtype == torch.bool, ctx.shape
    assert tgt.shape == (B, N, S_out) and tgt.dtype == torch.bool, tgt.shape
    assert ctx_tgt.shape == (B, N, S_out) and ctx_tgt.dtype == torch.bool, ctx_tgt.shape

    if masker.channel_based_masking and in_channels > 1:
        # layout is (S C): every channel carries the same mask
        c_ = ctx.reshape(B, S_, in_channels)
        t_ = tgt.reshape(B, N, S_, in_channels)
        v_ = ctx_tgt.reshape(B, N, S_, in_channels)
        assert torch.equal(c_, c_[..., :1].expand_as(c_))
        assert torch.equal(t_, t_[..., :1].expand_as(t_))
        assert torch.equal(v_, v_[..., :1].expand_as(v_))
        ctx, tgt, ctx_tgt = c_[..., 0], t_[..., 0], v_[..., 0]

    is_ctx = ~ctx                                                # True = context
    # 1. no context ∩ target
    assert not (is_ctx[:, None, :] & tgt).any(), "context and target overlap"
    # 2. ratio >= cutoff (same float32 semantics as the masker)
    ratio = is_ctx.sum(1) / S_
    assert bool((ratio >= masker.ratio_cutoff).all()), "context ratio below cutoff"
    # 3. ctx_tgt == xor
    assert torch.equal(ctx_tgt, torch.logical_xor(ctx[:, None, :], tgt))
    # 4. target runs >= min(l, distance to end)
    l_t = masker.target_length
    _, t_start, t_len = _runs(tgt)
    assert bool((t_len >= torch.minimum(torch.full_like(t_start, l_t), S_ - t_start)).all()), \
        "target run shorter than min(target_length, distance to end)"
    # 5. runs of the final ctx mask are unions of ctx / tgt spans
    l_min = min(masker.context_mask_length, l_t)
    _, c_start, c_len = _runs(ctx)
    assert bool((c_len >= torch.minimum(torch.full_like(c_start, l_min), S_ - c_start)).all()), \
        "masked-context run shorter than min(span length, distance to end)"


def _stats(ctx, tgt):
    """(mean, std, n) of #context tokens, #target tokens per group, #target runs per group."""
    n_ctx = (~ctx).sum(1).double()
    n_tgt = tgt.sum(2).reshape(-1).double()
    n_runs, _, _ = _runs(tgt)
    n_runs = n_runs.reshape(-1).double()
    out = {}
    for key, x in (("ctx_count", n_ctx), ("tgt_count", n_tgt), ("tgt_runs", n_runs)):
        out[key] = (x.mean().item(), x.std().item(), x.numel())
    return out


def _assert_stats_close(ref, vec, rel_tol=None, n_sigma=N_SIGMA, label=""):
    """Compare (mean, std) per statistic.

    rel_tol given  -> relative tolerance (DESIGN.md rule), sigma is printed for information.
    rel_tol None   -> |diff| <= n_sigma * SE with SE from the normal approximation
                      (mean: s/sqrt(n); std: s/sqrt(2n)).
    """
    lines, ok = [], True
    for key in ref:
        mr, sr, nr = ref[key]
        mv, sv, nv = vec[key]
        for what, r, v, se in (
            ("mean", mr, mv, (sr ** 2 / nr + sv ** 2 / nv) ** 0.5),
            ("std", sr, sv, (sr ** 2 / (2 * nr) + sv ** 2 / (2 * nv)) ** 0.5),
        ):
            diff = abs(r - v)
            rel = diff / max(abs(r), 1e-12)
            z = diff / se if se > 0 else 0.0
            passed = rel <= rel_tol if rel_tol is not None else z <= n_sigma
            ok &= passed
            lines.append(f"  {label}{key:10s} {what:4s} loop={r:9.4f} vec={v:9.4f} "
                         f"rel={rel * 100:5.2f}% z={z:4.1f}{'' if passed else '   <-- FAIL'}")
    print("\n".join(lines))
    assert ok, "statistics differ:\n" + "\n".join(lines)


def _assert_prob_close(p_hat, p_exact, n, what, n_sigma=N_SIGMA):
    se = max((p_exact * (1 - p_exact) / n) ** 0.5, 1e-12)
    z = abs(p_hat - p_exact) / se
    print(f"  {what}: hat={p_hat:.4f} exact={p_exact:.4f} z={z:.1f}")
    assert z <= n_sigma, (what, p_hat, p_exact, se)


def _assert_raises(exc_type, fn):
    try:
        fn()
    except exc_type:
        return
    raise AssertionError(f"expected {exc_type.__name__}")


# --------------------------------------------------------------------------- #
# structural invariants for both implementations
# --------------------------------------------------------------------------- #
def test_invariants_forward_audioset():
    m = TimeInverseBlockMasker(**AUDIOSET_CFG)
    B = 256
    ctx, tgt, ctx_tgt = m(batch_size=B, n_times=S, in_channels=1)
    _check_invariants(ctx, tgt, ctx_tgt, m, B, S)


def test_invariants_sample_audioset():
    m = TimeInverseBlockMasker(**AUDIOSET_CFG)
    B = 4096
    ctx, tgt, ctx_tgt = m.sample(batch_size=B, n_times=S, in_channels=1, generator=_gen(1))
    _check_invariants(ctx, tgt, ctx_tgt, m, B, S)
    assert ctx.device.type == "cpu"


def test_sample_signature_and_determinism():
    m = TimeInverseBlockMasker(**AUDIOSET_CFG)
    a = m.sample(16, S, 1, device="cpu", generator=_gen(123))
    b = m.sample(16, S, 1, device=torch.device("cpu"), generator=_gen(123))
    c = m.sample(16, S, 1, generator=_gen(124))
    for x, y in zip(a, b):
        assert torch.equal(x, y), "same seed must give identical masks"
    assert not torch.equal(a[0], c[0]), "different seeds should give different masks"
    # no generator at all uses the global RNG and must also work
    d = m.sample(batch_size=8, n_times=S)
    _check_invariants(*d, m, 8, S)


# --------------------------------------------------------------------------- #
# statistical comparison, AudioSet config, S=200 (2 % relative, DESIGN.md)
# --------------------------------------------------------------------------- #
def test_statistics_match_audioset():
    m = TimeInverseBlockMasker(**AUDIOSET_CFG)
    t0 = time.time()
    ctx_r, tgt_r, ctt_r = m(batch_size=REF_ROWS, n_times=S, in_channels=1)
    t_ref = time.time() - t0
    t0 = time.time()
    ctx_v, tgt_v, ctt_v = m.sample(batch_size=VEC_ROWS, n_times=S, in_channels=1, generator=_gen(7))
    t_vec = time.time() - t0
    print(f"\n  loop: {REF_ROWS} rows in {t_ref:.1f}s ({t_ref / REF_ROWS * 1e3:.2f} ms/row); "
          f"vectorised: {VEC_ROWS} rows in {t_vec:.2f}s ({t_vec / VEC_ROWS * 1e6:.1f} us/row)")
    _check_invariants(ctx_r, tgt_r, ctt_r, m, REF_ROWS, S)
    _check_invariants(ctx_v, tgt_v, ctt_v, m, VEC_ROWS, S)
    _assert_stats_close(_stats(ctx_r, tgt_r), _stats(ctx_v, tgt_v), rel_tol=REL_TOL)


# --------------------------------------------------------------------------- #
# edge cases
# --------------------------------------------------------------------------- #
def test_edge_small_S_min_len_adjustment_all_rows():
    """S=12, l=11, p=1 -> a=12/11: num_mask in {1, 2}; sz-l=1 <= num_mask for every row,
    so min_len = sz-num_mask-1 and starts come from [0, num_mask+1). Non-integral a
    exercises the probabilistic rounding. ratio_cutoff=0 because context is ~empty.

    Exact: num_mask=1 (prob 10/11): start in {0,1} -> 11 masked. num_mask=2 (prob 1/11):
    starts are a 2-subset of {0,1,2}; {0,1},{0,2} cover all 12, {1,2} covers 11 -> P(12)=2/33.
    """
    S_ = 12
    m = TimeInverseBlockMasker(target_masks_per_context=2, context_mask_prob=1.0, context_mask_length=11,
                               target_prob=1.0, target_length=11, ratio_cutoff=0.0)
    B_ref, B_vec = 4000, 100000
    ctx_r, tgt_r, ctt_r = m(batch_size=B_ref, n_times=S_, in_channels=1)
    ctx_v, tgt_v, ctt_v = m.sample(batch_size=B_vec, n_times=S_, in_channels=1, generator=_gen(3))
    _check_invariants(ctx_r, tgt_r, ctt_r, m, B_ref, S_)
    _check_invariants(ctx_v, tgt_v, ctt_v, m, B_vec, S_)
    assert set(tgt_r.sum(2).unique().tolist()) <= {11, 12}
    assert set(tgt_v.sum(2).unique().tolist()) <= {11, 12}
    print()
    p_exact = 2 / 33
    _assert_prob_close((tgt_r.sum(2) == 12).double().mean().item(), p_exact, B_ref * 2, "loop P(12 masked)")
    _assert_prob_close((tgt_v.sum(2) == 12).double().mean().item(), p_exact, B_vec * 2, "vec  P(12 masked)")
    _assert_stats_close(_stats(ctx_r, tgt_r), _stats(ctx_v, tgt_v), label="smallS ")


def test_edge_min_len_adjustment_mixed_rows():
    """S=20, l=10, p=4.75 -> a=9.5: num_mask in {9, 10}; sz-l=10 <= num_mask only for
    num_mask=10, so the number of admissible starts differs between rows (10 vs 11).

    Exact: num_mask=9 -> starts = {0..9} minus one m: m=0 covers [1,19), m=9 covers [0,18),
    else [0,19). num_mask=10 -> starts = {0..10} minus m: m=0 -> [1,20), m=10 -> [0,19),
    else [0,20). Hence P(18)=0.1, P(19)=0.4+1/11, P(20)=9/22 masked per span-set, and
    E[#ctx] = sum_pos P(pos uncovered)^2 = (0.05+1/22)^2 + (0.5+1/22)^2 + 0.05^2.
    """
    S_ = 20
    m = TimeInverseBlockMasker(target_masks_per_context=1, context_mask_prob=4.75, context_mask_length=10,
                               target_prob=4.75, target_length=10, ratio_cutoff=0.0)
    B_ref, B_vec = 4000, 200000
    ctx_r, tgt_r, ctt_r = m(batch_size=B_ref, n_times=S_, in_channels=1)
    ctx_v, tgt_v, ctt_v = m.sample(batch_size=B_vec, n_times=S_, in_channels=1, generator=_gen(5))
    _check_invariants(ctx_r, tgt_r, ctt_r, m, B_ref, S_)
    _check_invariants(ctx_v, tgt_v, ctt_v, m, B_vec, S_)
    exact = {18: 0.1, 19: 0.4 + 1 / 11, 20: 9 / 22}
    cnt_r = tgt_r.sum(2).reshape(-1)
    cnt_v = tgt_v.sum(2).reshape(-1)
    assert set(cnt_r.unique().tolist()) <= set(exact) and set(cnt_v.unique().tolist()) <= set(exact)
    print()
    for v, p in exact.items():
        _assert_prob_close((cnt_r == v).double().mean().item(), p, cnt_r.numel(), f"loop P(count={v})")
        _assert_prob_close((cnt_v == v).double().mean().item(), p, cnt_v.numel(), f"vec  P(count={v})")
    e_ctx = (0.05 + 1 / 22) ** 2 + (0.5 + 1 / 22) ** 2 + 0.05 ** 2
    s_r, s_v = _stats(ctx_r, tgt_r), _stats(ctx_v, tgt_v)
    for name, st in (("loop", s_r), ("vec ", s_v)):
        mean, std, n = st["ctx_count"]
        z = abs(mean - e_ctx) / (std / n ** 0.5)
        print(f"  {name} E[#ctx]: hat={mean:.4f} exact={e_ctx:.4f} z={z:.1f}")
        assert z <= N_SIGMA, (name, mean, e_ctx)
    _assert_stats_close(s_r, s_v, label="mixed ")


def test_edge_non_integral_a_S200():
    """AudioSet-like but p_ctx=0.63 (a=12.6) and p_tgt=0.27 (a=5.4): num_mask varies per row."""
    cfg = dict(AUDIOSET_CFG, context_mask_prob=0.63, target_prob=0.27)
    m = TimeInverseBlockMasker(**cfg)
    B_ref, B_vec = 4000, 60000
    ctx_r, tgt_r, ctt_r = m(batch_size=B_ref, n_times=S, in_channels=1)
    ctx_v, tgt_v, ctt_v = m.sample(batch_size=B_vec, n_times=S, in_channels=1, generator=_gen(11))
    _check_invariants(ctx_r, tgt_r, ctt_r, m, B_ref, S)
    _check_invariants(ctx_v, tgt_v, ctt_v, m, B_vec, S)
    print()
    _assert_stats_close(_stats(ctx_r, tgt_r), _stats(ctx_v, tgt_v), label="nonint ")


def test_edge_channel_based_masking_two_channels():
    cfg = dict(AUDIOSET_CFG, channel_based_masking=True)
    m = TimeInverseBlockMasker(**cfg)
    C = 2
    n_times = S * C
    B_ref, B_vec = 1500, 20000
    ctx_r, tgt_r, ctt_r = m(batch_size=B_ref, n_times=n_times, in_channels=C)
    ctx_v, tgt_v, ctt_v = m.sample(batch_size=B_vec, n_times=n_times, in_channels=C, generator=_gen(9))
    assert ctx_v.shape == (B_vec, n_times) and tgt_v.shape == (B_vec, 4, n_times)
    assert ctx_r.shape == (B_ref, n_times) and tgt_r.shape == (B_ref, 4, n_times)
    _check_invariants(ctx_r, tgt_r, ctt_r, m, B_ref, S, in_channels=C)
    _check_invariants(ctx_v, tgt_v, ctt_v, m, B_vec, S, in_channels=C)
    # de-duplicated per-time masks follow the same statistics
    ctx_r1 = ctx_r.reshape(B_ref, S, C)[..., 0]
    tgt_r1 = tgt_r.reshape(B_ref, 4, S, C)[..., 0]
    ctx_v1 = ctx_v.reshape(B_vec, S, C)[..., 0]
    tgt_v1 = tgt_v.reshape(B_vec, 4, S, C)[..., 0]
    print()
    _assert_stats_close(_stats(ctx_r1, tgt_r1), _stats(ctx_v1, tgt_v1), label="chan ")

    # in_channels=2 without channel_based_masking: outputs have width n_times // 2
    m2 = TimeInverseBlockMasker(**AUDIOSET_CFG)
    ctx2, tgt2, ctt2 = m2.sample(batch_size=8, n_times=n_times, in_channels=C, generator=_gen(2))
    assert ctx2.shape == (8, S) and tgt2.shape == (8, 4, S) and ctt2.shape == (8, 4, S)
    _check_invariants(ctx2, tgt2, ctt2, m2, 8, S)


def test_edge_high_ratio_cutoff_redraws():
    """ratio_cutoff=0.2 with the AudioSet config rejects most raw draws, so redraw rounds
    are exercised; the conditioned distributions must still agree."""
    m0 = TimeInverseBlockMasker(**dict(AUDIOSET_CFG, ratio_cutoff=0.0))
    ctx0, _, _ = m0.sample(batch_size=20000, n_times=S, generator=_gen(21))
    p_fail = float(((~ctx0).sum(1) / S < 0.2).double().mean())
    print(f"\n  unconditioned P(ratio < 0.2) = {p_fail:.3f}")
    assert p_fail > 0.3, "cutoff not high enough to exercise redraws"

    m = TimeInverseBlockMasker(**dict(AUDIOSET_CFG, ratio_cutoff=0.2))
    # count the draw calls to prove the redraw loop ran
    calls = {"n": 0}
    orig = m._draw_rows

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    m._draw_rows = counting  # instance attribute shadows the bound method
    B_ref, B_vec = 2500, 30000
    ctx_v, tgt_v, ctt_v = m.sample(batch_size=B_vec, n_times=S, generator=_gen(22))
    assert calls["n"] > 1, "no redraw round happened"
    print(f"  redraw rounds (vectorised): {calls['n'] - 1}")
    ctx_r, tgt_r, ctt_r = m(batch_size=B_ref, n_times=S, in_channels=1)
    _check_invariants(ctx_r, tgt_r, ctt_r, m, B_ref, S)
    _check_invariants(ctx_v, tgt_v, ctt_v, m, B_vec, S)
    _assert_stats_close(_stats(ctx_r, tgt_r), _stats(ctx_v, tgt_v), label="cutoff ")


def test_unsatisfiable_cutoff_raises():
    m = TimeInverseBlockMasker(**dict(AUDIOSET_CFG, ratio_cutoff=1.0))  # >= 1 span always masked
    _assert_raises(RuntimeError, lambda: m.sample(batch_size=4, n_times=S, generator=_gen(0),
                                                  max_redraw_rounds=20))


def test_num_mask_zero_raises_like_reference():
    """a = p*S/l < 1 -> num_mask can be 0 -> compute_mask_indices raises ValueError; so do we."""
    m = TimeInverseBlockMasker(**dict(AUDIOSET_CFG, target_prob=0.25, target_length=10))
    S_ = 20  # a_tgt = 0.5
    _assert_raises(ValueError, lambda: m(batch_size=64, n_times=S_, in_channels=1))
    _assert_raises(ValueError, lambda: m.sample(batch_size=64, n_times=S_, generator=_gen(0)))


def test_zero_target_groups():
    m = TimeInverseBlockMasker(**dict(AUDIOSET_CFG, target_masks_per_context=0))
    ctx, tgt, ctt = m.sample(batch_size=8, n_times=S, generator=_gen(0))
    ctx_r, tgt_r, ctt_r = m(batch_size=8, n_times=S, in_channels=1)
    assert tgt.shape == (8, 0, S) and ctt.shape == (8, 0, S)
    assert tgt_r.shape == (8, 0, S) and ctt_r.shape == (8, 0, S)


# --------------------------------------------------------------------------- #
# plain-python runner
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            failures += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {e}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
