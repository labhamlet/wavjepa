"""
Tests for ``hear_api/streaming.py`` (DESIGN2.md E4).  CPU only, no checkpoints, < 2 min.

Runnable as ``python tests/test_streaming.py`` (pytest is optional).

(a) frame bookkeeping: for a 7 s clip and several (W, C) -- W = C, W > C, C not a multiple
    of the token hop, W/C not multiples of each other, and W = C = inf -- and both emission
    rules (``rf_end``: a chunk emits the frames whose receptive field ends inside it;
    ``centre``: the frames whose centre lies inside it), the emitted frames are exactly the
    analytically expected frames of the contract's windows ``wave[t + C - W : t + C]``, each
    exactly once, with strictly increasing timestamps at the token hop (``rf_end`` with
    W >= C + rf emits the complete token grid; otherwise a grid frame is missing only where a
    receptive field straddles an uncovered window boundary), and every emitted embedding was
    computed from exactly the planned window;
(b) look-ahead bound: with a base model whose frame features depend on the whole window,
    perturbing the samples >= t + C leaves every frame with centre < t + C unchanged;
(c) W = C = the whole clip (finite, and inf) reproduces the base model's output;
(d) WavJEPA path: the real ``RuntimeJEPA`` (around a tiny random JEPA) wrapped in
    ``StreamingWrapper(W = C = process_seconds)`` equals ``RuntimeJEPA.get_timestamp_embeddings``
    (allclose 1e-4) for both pad modes; the window snaps to the runtime's ``unit_frames``
    (``int(unit_frames / sr * sr)`` truncates to ``unit_frames - 1``); short windows equal a
    manual padded+masked forward;
(e) batching invariance, scene embedding, latency / RTF accounting, argument validation;
(f) BEATs path on a tiny random BEATs (``third_party/BEATs``): ``8 * (fbank_frames // 16)``
    tokens (2 s -> 96, 10 s -> 496) become one frame per time patch (mean / concat of the 8
    band tokens) stamped ``(16 j + 8) x 10 ms``; W = C = inf equals the pooled
    ``extract_features``; C < 0.25 s is refused by ``hear_configs.BEATs_stream``.
"""
from __future__ import annotations

import math
import os
import sys
import time
import warnings

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from hear_api.runtime import normalize  # noqa: E402
from hear_api.streaming import (  # noqa: E402
    BEATsWindowModel,
    StreamingWrapper,
    WavJEPAWindowModel,
    WindowModel,
    cpu_safe_runtime,
    runtime_from_jepa,
)
from wavjepa.extractors import ConvFeatureExtractor  # noqa: E402
from wavjepa.jepa import JEPA  # noqa: E402
from wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG  # noqa: E402

try:  # pytest is optional
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None

torch.set_num_threads(min(4, torch.get_num_threads()))
warnings.filterwarnings("ignore")  # torch.masked prototype warning inside the runtime
SR = 16000


# --------------------------------------------------------------------------- synthetic base
class SyntheticBase(WindowModel):
    """Parameter-free window model with exact, inspectable frame features.

    Frame ``i`` of a window ``w`` of ``Lw`` samples -> ``[mean(w[support(i)]), mean(w), Lw, i]``:
    feature 0 depends on the frame's receptive field only, feature 1 on the whole window (so
    any window that leaks future samples is detectable), 2 and 3 expose the window length
    and the frame index.
    """

    def __init__(self, hop: int, rf: int, centre_offset=None, n_frames_fn=None, name="synthetic",
                 max_gap_hops=None, support_fn=None):
        super().__init__()
        self.name = name
        self.sample_rate = SR
        self.embedding_dim = 4
        self.hop, self.rf = hop, rf
        self.centre_offset = rf / 2.0 if centre_offset is None else centre_offset
        self.max_window = None
        self._n_frames_fn = n_frames_fn
        self._support_fn = support_fn
        # largest gap (in hops) between consecutive emitted frames: a receptive field that
        # straddles an uncovered window boundary (+1 for the grid alignment of the start)
        self.max_gap_hops = (-(-rf // hop) + 1) if max_gap_hops is None else max_gap_hops

    def n_frames(self, n_samples: int) -> int:
        if self._n_frames_fn is not None:
            return self._n_frames_fn(n_samples)
        return super().n_frames(n_samples)

    def support(self, i: int):
        """``(start, end)`` samples (relative to the window) frame ``i`` truly depends on."""
        if self._support_fn is not None:
            return self._support_fn(i)
        return i * self.hop, i * self.hop + self.rf

    def embed_window(self, wave: torch.Tensor) -> torch.Tensor:
        n = self.n_frames(wave.shape[-1])
        if n == 0:
            return wave.new_zeros((wave.shape[0], 0, 4))
        f0 = torch.stack([wave[:, a:b].mean(-1) for a, b in (self.support(i) for i in range(n))], dim=1)
        f1 = wave.mean(-1, keepdim=True).expand(-1, n)
        f2 = torch.full_like(f0, float(wave.shape[-1]))
        f3 = torch.arange(n, dtype=wave.dtype).unsqueeze(0).expand_as(f0)
        return torch.stack([f0, f1, f2, f3], dim=-1)


def _wrap(base, W, C, **kw):
    return StreamingWrapper(
        base, W, C, SR, base.token_hop_ms, base.front_rf_ms, base.max_window_s, **kw
    )


RULES = ("rf_end", "centre")


def _belongs(rule, t, Cs, start, rf, centre):
    """Does the frame with receptive field ``[start, start + rf)`` / centre ``centre`` belong to
    the chunk ``[t, t + Cs)`` under ``rule``?"""
    if rule == "rf_end":  # receptive field inside the stream and ending inside the chunk
        return start >= 0 and t < start + rf <= t + Cs
    return t <= centre < t + Cs


def _expected_frames(base, W, C, L, left_pad=True, rule="rf_end"):
    """Independent re-derivation of the schedule: list of (chunk, s, e, i, centre) for every
    emitted frame in emission order (``i`` = frame index inside its window, ``centre`` in
    samples).  Windows are the contract's ``wave[t + C - W : t + C]`` (clipped at the clip
    end), seconds -> samples with the runtime's truncating ``int()``."""
    hop, rf = base.hop, base.rf
    off = base.centre_offset
    if math.isinf(W):
        n = base.n_frames(L)
        return [(0, 0, L, i, i * hop + off) for i in range(n)]
    Ws, Cs = int(W * SR), int(C * SR)
    if abs(W - C) < 1e-9:
        Cs = Ws
    out = []
    k = 0
    while k * Cs < L:
        t, e = k * Cs, min(k * Cs + Cs, L)
        s = t + Cs - Ws
        if not left_pad:
            s = max(s, 0)
        assert e - s <= Ws
        n = base.n_frames(e - s)
        for i in range(n):
            centre = s + i * hop + off
            if _belongs(rule, t, Cs, s + i * hop, rf, centre):
                out.append((k, s, e, i, centre))
        k += 1
    return out


# BEATs-like frames: one per 16-frame time patch of a 25 ms / 10 ms fbank (hop 2560 samples,
# support 2800 samples, timestamp (16 j + 8) x 10 ms -> centre offset 1280 samples).
def _beats_like_n_frames(L: int) -> int:
    return (1 + (L - 400) // 160) // 16 if L >= 400 else 0


def _beats_like_support(i: int):
    return i * 2560, i * 2560 + 2800


BASES = {
    "wavjepa100": lambda: SyntheticBase(160, 240),
    "wavjepa50": lambda: SyntheticBase(320, 400),
    "tiny": lambda: SyntheticBase(80, 100),
    # up to 15 fbank frames (+ the 25 ms window) at a window end yield no patch -> the gap
    # across a chunk boundary is < 2 patch hops
    "beats": lambda: SyntheticBase(2560, 2800, centre_offset=1280.0, n_frames_fn=_beats_like_n_frames,
                                   max_gap_hops=2, support_fn=_beats_like_support),
}
CONFIGS = [  # (W, C) in seconds
    (2.01, 2.01),   # the real model's window: int(2.01 * 16000) = 32159 samples, not a whole number of hops
    (2.0, 2.0), (1.0, 1.0), (2.0, 1.0), (1.0, 0.5), (2.0, 0.25), (2.0, 0.1), (1.0, 0.1),
    (0.5, 0.125),   # C = 12.5 hops of 10 ms -> interleaved grids
    (0.37, 0.13),   # W not a multiple of C
    (0.75, 0.31),
    (math.inf, math.inf),
]


# --------------------------------------------------------------------------- (a) bookkeeping
def _check_bookkeeping(base, W, C, L, x, left_pad=True, rule="rf_end"):
    wrapper = _wrap(base, W, C, batch_windows=7, left_pad=left_pad, emit_rule=rule)
    emb, ts = wrapper.get_timestamp_embeddings(x)
    B = x.shape[0]
    expected = _expected_frames(base, W, C, L, left_pad, rule)
    assert emb.shape == (B, len(expected), 4), (emb.shape, len(expected))
    assert ts.shape == (B, len(expected)) and ts.dtype == torch.float32 and ts.device.type == "cpu"
    assert wrapper.n_emitted(L) == len(expected)
    hop_ms = base.token_hop_ms
    # timestamps: the planned centres, each (chunk, i) exactly once, strictly increasing
    keys = [(k, i) for k, _s, _e, i, _c in expected]
    assert len(set(keys)) == len(keys), "a frame was emitted twice"
    centres = [c for *_, c in expected]
    assert all(b > a for a, b in zip(centres, centres[1:])), "planned centres not increasing"
    exp_ts = torch.tensor(centres, dtype=torch.float64) / SR * 1000.0
    assert torch.allclose(ts[0].double(), exp_ts, atol=1e-3), (ts[0][:5], exp_ts[:5])
    assert torch.equal(ts[0], ts[-1])
    d = ts[0, 1:].double() - ts[0, :-1].double()
    assert (d > 0).all(), "timestamps not strictly increasing"
    steps = d / hop_ms
    # inside a chunk the spacing is exactly one hop
    same_chunk = torch.tensor([a[0] == b[0] for a, b in zip(expected, expected[1:])])
    assert torch.allclose(steps[same_chunk], torch.ones_like(steps[same_chunk]), atol=1e-4)
    n_grid = base.n_frames(L)
    if math.isinf(W):
        assert len(keys) == n_grid
    else:
        Cs = wrapper.chunk
        if Cs % base.hop == 0:
            # C = whole hops -> one global grid: diffs are whole hops, gaps only at chunk ends
            assert torch.allclose(steps, steps.round(), atol=1e-4), "timestamp diff not a multiple of the hop"
            assert int(steps.max().round()) <= max(1, base.max_gap_hops), (W, C, steps.max())
            assert len(keys) >= n_grid - (L // Cs + 1) * base.max_gap_hops
            if rule == "rf_end" and wrapper.window >= Cs + base.rf and left_pad:
                # every grid frame's receptive field fits in the window of the chunk its end
                # falls in -> the complete grid, no frame lost
                assert len(keys) == n_grid and torch.allclose(steps, torch.ones_like(steps), atol=1e-4), (W, C, len(keys), n_grid)
            elif rule == "centre" and wrapper.window > Cs:
                # the frame straddling each chunk end is lost: fewer frames than the grid
                assert len(keys) < n_grid, (W, C)
        else:
            # interleaved grids (window starts differ by C, not a whole number of hops): the
            # spacing across a chunk end is at least gcd(C, hop), never a near-duplicate
            assert steps.min() > math.gcd(Cs, base.hop) / base.hop - 1e-4, (W, C, steps.min())
    # the embeddings were computed from exactly the planned window
    for pos, (k, s, e, i, _c) in enumerate(expected):
        a, z = base.support(i)
        assert z <= e - s, "a frame's true support must fit in its window"
        for b in range(B):
            win = x.new_zeros(e - s)
            win[max(s, 0) - s :] = x[b, max(s, 0) : e]
            assert abs(emb[b, pos, 0].item() - win[a:z].mean().item()) < 1e-5
            assert abs(emb[b, pos, 1].item() - win.mean().item()) < 1e-6
            assert emb[b, pos, 2].item() == e - s and emb[b, pos, 3].item() == i
    return wrapper, emb, ts


def test_frame_bookkeeping():
    torch.manual_seed(0)
    L = 7 * SR
    x = torch.randn(2, L)
    for name, make in BASES.items():
        base = make()
        for rule in RULES:
            for W, C in CONFIGS:
                if name == "beats" and not math.isinf(C) and C < 0.25:
                    continue
                _check_bookkeeping(base, W, C, L, x, rule=rule)
            # no left padding variant
            _check_bookkeeping(base, 2.0, 0.5, L, x, left_pad=False, rule=rule)
        print(f"  bookkeeping ok for {name} ({len(CONFIGS)} configs x {len(RULES)} rules)")
    # the trailing chunk: window = wave[t + C - W : L] (shorter than W), no zero padding at the end
    base = BASES["wavjepa100"]()
    for rule in RULES:
        wrapper = _wrap(base, 2.0, 0.5, emit_rule=rule)
        last = wrapper.plan(L + 3000)[-1]
        assert last.t == 7 * SR and last.e == L + 3000 and last.s == last.t + wrapper.chunk - wrapper.window
        assert last.length == last.e - last.s == 27000 < wrapper.window
        assert last.n_emit == sum(
            1 for i in range(last.n_frames)
            if _belongs(rule, last.t, wrapper.chunk, last.s + i * base.hop, base.rf, last.s + i * base.hop + base.centre_offset)
        )
    # the two rules differ exactly by the frames whose receptive field straddles a chunk end
    a = _wrap(base, 2.0, 0.1, emit_rule="rf_end").n_emitted(L)
    b = _wrap(base, 2.0, 0.1, emit_rule="centre").n_emitted(L)
    assert a == base.n_frames(L) == 699 and b == 699 - 69, (a, b)  # centre: 1 frame lost per chunk end (10 %)
    # with W = C the rules coincide (no window overlap)
    for W in (2.0, 2.01, 0.5):
        assert _wrap(base, W, W, emit_rule="rf_end").frame_centres_ms(L).equal(_wrap(base, W, W, emit_rule="centre").frame_centres_ms(L))


def test_dense_when_window_overlaps():
    """With W - C >= rf the only missing grid frames are the ones whose receptive field
    crosses a chunk end (< rf/2 before the boundary): the grid is otherwise dense."""
    base = BASES["wavjepa100"]()
    L = 7 * SR
    x = torch.randn(1, L)
    for W, C in [(2.0, 1.0), (2.0, 0.5), (2.0, 0.1), (1.0, 0.25)]:
        for rule in RULES:
            wrapper, emb, ts = _check_bookkeeping(base, W, C, L, x, rule=rule)
            n_chunks = len(wrapper.plan(L))
            missing = base.n_frames(L) - ts.shape[1]
            if rule == "rf_end":
                assert missing == 0, (W, C, missing)
            else:
                assert 0 < missing <= n_chunks, (W, C, missing, n_chunks)
            # frames per second close to 1/hop
            assert abs(ts.shape[1] / 7.0 - 100.0) < 100.0 * missing / base.n_frames(L) + 1.0


# --------------------------------------------------------------------------- (b) look-ahead
def test_lookahead_bound():
    torch.manual_seed(1)
    L = 5 * SR
    x = torch.randn(1, L)
    for name in ("wavjepa100", "beats"):
        base = BASES[name]()
        for rule in RULES:
            for W, C in [(2.0, 0.5), (1.0, 0.25), (2.0, 2.0), (0.5, 0.5), (0.75, 0.31)]:
                if name == "beats" and C < 0.25:
                    continue
                wrapper = _wrap(base, W, C, emit_rule=rule)
                emb, ts = wrapper.get_timestamp_embeddings(x)
                Cs = wrapper.chunk
                # frames emitted before the chunk end b = k*C: centre < b (centre rule) /
                # receptive-field end <= b (rf_end rule); they must not see samples >= b
                ts_samples = ts[0].double() * SR / 1000.0
                changed_any = False
                for k in range(1, L // Cs + 1):
                    b = k * Cs
                    if b >= L:
                        break
                    x2 = x.clone()
                    x2[:, b:] += 10.0 * torch.randn(L - b)
                    emb2, ts2 = wrapper.get_timestamp_embeddings(x2)
                    assert torch.equal(ts, ts2)
                    if rule == "centre":
                        past = ts_samples < b - 1e-6
                    else:
                        past = ts_samples + (base.rf - base.centre_offset) <= b + 1e-6
                    assert torch.equal(emb[0, past], emb2[0, past]), (name, rule, W, C, k)
                    future = ~past
                    if future.any():
                        changed_any |= not torch.equal(emb[0, future], emb2[0, future])
                        # ... and those do depend on the perturbed samples (window mean feature)
                        assert not torch.equal(emb[0, future, 1], emb2[0, future, 1])
                assert changed_any, "the perturbation test is vacuous"
        print(f"  look-ahead bound ok for {name} (both rules)")


# --------------------------------------------------------------------------- (c) full window
def test_full_window_reproduces_base():
    torch.manual_seed(2)
    for name in ("wavjepa100", "tiny", "beats"):
        base = BASES[name]()
        L = 3 * SR
        x = torch.randn(2, L)
        ref = base.embed_window(x)
        for (W, C), rule in [((3.0, 3.0), "rf_end"), ((3.0, 3.0), "centre"), ((math.inf, math.inf), "rf_end"), ((math.inf, math.inf), "centre")]:
            wrapper = _wrap(base, W, C, emit_rule=rule)
            emb, ts = wrapper.get_timestamp_embeddings(x)
            assert torch.equal(emb, ref)
            exp = torch.arange(ref.shape[1], dtype=torch.float64) * base.token_hop_ms + base.centre_offset_ms
            assert torch.allclose(ts[0].double(), exp, atol=1e-3)
            assert math.isinf(wrapper.latency_ms()) == math.isinf(C)
            if not math.isinf(C):
                assert abs(wrapper.latency_ms() - (C * 1000 + base.front_rf_ms)) < 1e-9


# --------------------------------------------------------------------------- (d) WavJEPA path
TINY_CONV = [(64, 10, 5), (64, 3, 2), (64, 3, 2), (64, 2, 2), (64, 2, 2)]  # hop 80, rf 100


def _audio_len_for_tokens(n_tokens: int) -> int:
    length = n_tokens
    for _dim, k, stride in reversed(TINY_CONV):
        length = (length - 1) * stride + k
    return length


def _tiny_runtime(S: int = 200, seed: int = 0):
    """RuntimeJEPA around a tiny random JEPA built like ``train.default_transformer_cfgs``
    (same ``TransformerLayerCFG.create`` / ``TransformerEncoderCFG.create`` calls, small sizes).
    ``process_audio_seconds`` >= 1 s because the runtime floors ``target_length // sr``, and
    the window is a whole number of hops (201 x 80 = 16080 samples, like the real model's
    2.01 s = 201 x 160) so that the runtime's window starts lie on the token grid."""
    torch.manual_seed(seed)
    extractor = ConvFeatureExtractor(conv_layers_spec=TINY_CONV, in_channels=1, depthwise=False)
    hop = 80
    audio_len = -(-_audio_len_for_tokens(S) // hop) * hop  # 16020 -> 16080, still S frames
    assert audio_len >= SR, "the runtime needs process_seconds >= 1 s"
    model = JEPA(
        feature_extractor=extractor,
        transformer_encoder_cfg=TransformerEncoderCFG.create(num_layers=3),
        transformer_encoder_layers_cfg=TransformerLayerCFG.create(d_model=64, nhead=4),
        transformer_decoder_cfg=TransformerEncoderCFG.create(num_layers=2),
        transformer_decoder_layers_cfg=TransformerLayerCFG.create(d_model=32, nhead=4),
        decoder_embedding_dim=32,
        average_top_k_layers=2,
        resample_sr=SR,
        process_audio_seconds=(audio_len + 0.5) / SR,
        nr_samples_per_audio=1,
        compile_modules=False,
    )
    assert model.target_length == audio_len and model.total_patches == S
    return cpu_safe_runtime(runtime_from_jepa(model.eval(), SR, in_channels=1))


def test_wavjepa_full_equals_runtime():
    runtime = _tiny_runtime()
    P = runtime.unit_frames / SR  # 1.005 s; int(P * SR) = 16079, the wrapper must snap to 16080
    assert int(P * SR) == runtime.unit_frames - 1, "the test relies on the truncation pitfall"
    torch.manual_seed(3)
    x = 0.3 * torch.randn(2, 3 * runtime.unit_frames)  # exactly three windows
    ref, ref_ts = runtime.get_timestamp_embeddings(x)
    assert ref.shape == (2, 600, 64), ref.shape
    for pad_mode, rule in [("zero", "rf_end"), ("trim", "rf_end"), ("zero", "centre"), ("trim", "centre")]:
        base = WavJEPAWindowModel(runtime, pad_mode=pad_mode)
        assert base.hop == 80 and base.rf == 100 and base.n_frames(runtime.unit_frames) == 200
        assert base.max_window == runtime.unit_frames == runtime.model.target_length
        wrapper = _wrap(base, P, P, batch_windows=2, emit_rule=rule)
        assert wrapper.window == wrapper.chunk == runtime.unit_frames, (wrapper.window, wrapper.chunk)
        emb, ts = wrapper.get_timestamp_embeddings(x)
        assert emb.shape == ref.shape and ts.shape == ref_ts.shape
        diff = (emb - ref).abs().max().item()
        assert torch.allclose(emb, ref, atol=1e-4, rtol=1e-4), f"{pad_mode}/{rule}: max abs diff {diff:.2e}"
        # our timestamps are frame centres; the runtime's are (approximate) frame starts.
        # Every window boundary loses one grid frame (200 frames per 201-hop window).
        j = torch.tensor([w * (runtime.unit_frames // base.hop) + i for w in range(3) for i in range(200)])
        exp = j.double() * base.token_hop_ms + base.front_rf_ms / 2
        assert torch.allclose(ts[0].double(), exp, atol=1e-3)
        assert (ts[0].double() - ref_ts[0].double()).abs().max() < base.token_hop_ms + base.front_rf_ms
        print(f"  WavJEPA W=C=P ({pad_mode}, {rule}) == RuntimeJEPA: max abs diff {diff:.2e}")
    # a non-multiple clip: the full windows still agree; the trailing chunk is the runtime's
    # last (short) window, zero-padded to process_seconds behind the key-padding mask
    L2 = 2 * runtime.unit_frames + 7000
    x2 = 0.3 * torch.randn(1, L2)
    ref2, _ = runtime.get_timestamp_embeddings(x2)
    base = WavJEPAWindowModel(runtime, pad_mode="zero")
    wrapper = _wrap(base, P, P)
    emb2, ts2 = wrapper.get_timestamp_embeddings(x2)
    assert torch.allclose(emb2[:, :400], ref2[:, :400], atol=1e-4, rtol=1e-4)
    last = wrapper.plan(L2)[-1]
    assert last.s == 2 * runtime.unit_frames and last.e == L2 and last.length == 7000
    assert last.i_min == 0 and last.n_emit == base.n_frames(7000) == (7000 - 100) // 80 + 1
    assert emb2.shape[1] == 400 + last.n_emit
    # the same frames as a manual padded + masked forward of the short window
    win = base.prepare(x2)[:, 2 * runtime.unit_frames :]
    manual = base.embed_window(win)
    assert torch.allclose(emb2[:, 400:], manual, atol=1e-5, rtol=1e-5)
    # seconds -> samples follows the runtime's truncating int(): 2.01 s = 32159 samples for a
    # base without a maximum window ...
    assert _wrap(BASES["wavjepa100"](), 2.01, 2.01).window == int(2.01 * SR) == 32159
    # ... and snaps to the base model's exact maximum window when it has one
    snap = SyntheticBase(160, 240)
    snap.max_window = 32159
    assert _wrap(snap, 2.01, 2.01).window == 32159 and _wrap(snap, 32159 / SR, 32159 / SR).window == 32159
    assert _wrap(snap, 2.01, 1.0).window == 32159 and _wrap(snap, 2.01, 1.0).chunk == 16000


def test_wavjepa_short_windows():
    """Windows shorter than process_seconds: zero mode == manual pad + key-padding mask via
    get_audio_representation; trim mode == the manual pipeline on the unpadded window; the
    padded frames never appear in the output."""
    runtime = _tiny_runtime(seed=4)
    jepa = runtime.model
    torch.manual_seed(5)
    Lw = 6000  # 0.375 s window (S = 200 frames per 1.005 s window)
    win = 0.2 * torch.randn(3, Lw)
    n_real = (Lw - 100) // 80 + 1
    x = normalize(win.unsqueeze(1))
    with torch.inference_mode():
        xp = torch.nn.functional.pad(x, (0, runtime.unit_frames - Lw))
        mask = (torch.arange(200) >= n_real).unsqueeze(0).expand(3, 200)
        ref_zero = jepa.get_audio_representation(xp, mask)[:, :n_real]
        local = jepa.feature_norms(jepa.extract_audio(x))
        if jepa.post_extraction_mapper is not None:
            local = jepa.post_extraction_mapper(local)
        ref_trim = jepa.encoder_forward(local + jepa.pos_encoding_encoder[:, :n_real], None)
    for pad_mode, ref in (("zero", ref_zero), ("trim", ref_trim)):
        base = WavJEPAWindowModel(runtime, pad_mode=pad_mode)
        out = base.embed_window(win)
        assert out.shape == (3, n_real, 64)
        assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5), pad_mode
        # streaming with W < P: frame counts and timestamps follow the bookkeeping
        wrapper = _wrap(base, 0.375, 0.125, batch_windows=5)
        emb, ts = wrapper.get_timestamp_embeddings(0.2 * torch.randn(2, 2 * SR))
        assert emb.shape[1] == ts.shape[1] == wrapper.n_emitted(2 * SR) and emb.shape[1] > 0
        assert torch.isfinite(emb).all()
    # W > process_seconds is refused
    try:
        _wrap(WavJEPAWindowModel(runtime), 1.5, 0.5)
    except ValueError as e:
        assert "maximum window" in str(e)
    else:  # pragma: no cover
        raise AssertionError("W > process_seconds must be refused")


# --------------------------------------------------------------------------- (e) misc
def test_batching_scene_rtf_and_validation():
    base = BASES["wavjepa100"]()
    torch.manual_seed(6)
    x = torch.randn(3, 4 * SR)
    w_small = _wrap(base, 2.0, 0.5, batch_windows=1)
    w_big = _wrap(base, 2.0, 0.5, batch_windows=64)
    e1, t1 = w_small.get_timestamp_embeddings(x)
    e2, t2 = w_big.get_timestamp_embeddings(x)
    assert torch.equal(e1, e2) and torch.equal(t1, t2)
    e_row, t_row = w_big.get_timestamp_embeddings(x[1:2])
    assert torch.equal(e_row[0], e2[1]) and torch.equal(t_row[0], t2[1])
    assert torch.equal(w_big.scene(x), e2.mean(1))
    assert torch.equal(w_big.get_scene_embeddings(x), e2.mean(1))
    # 1-D input
    e_1d, _ = w_big.get_timestamp_embeddings(x[0])
    assert torch.equal(e_1d[0], e2[0])
    # RTF accounting: 5 calls on 12 + 4 + 12 + 12 + 4 s of audio
    assert w_big.n_calls == 5 and abs(w_big.total_audio_s - 44.0) < 1e-9
    assert w_big.rtf is not None and w_big.rtf > 0 and w_big.last_rtf > 0
    rep = w_big.rtf_report()
    assert rep["calls"] == 5 and rep["W"] == 2.0 and rep["C"] == 0.5 and rep["latency_ms"] == 515.0
    assert rep["window_samples"] == 32000 and rep["chunk_samples"] == 8000 and rep["centre_offset_ms"] == 7.5
    assert rep["emit_rule"] == "rf_end" and _wrap(base, 2.0, 0.5, emit_rule="centre").rtf_report()["emit_rule"] == "centre"
    # rtf json
    import tempfile, json
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "rtf.json")
        w = _wrap(base, 1.0, 0.5, rtf_json=path)
        w.get_timestamp_embeddings(x)
        rep = json.load(open(path))
        assert rep["calls"] == 1 and rep["rtf"] > 0 and rep["latency_ms"] == 515.0
    w_inf = _wrap(base, math.inf, math.inf)
    assert w_inf.rtf_report()["W"] == "inf" and w_inf.rtf_report()["latency_ms"] == "inf"
    # validation
    for bad in [dict(W=0.5, C=1.0), dict(W=math.inf, C=1.0), dict(W=0.005, C=0.005), dict(W=-1, C=-1)]:
        try:
            _wrap(base, bad["W"], bad["C"])
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"{bad} must be refused")
    try:
        StreamingWrapper(base, 1.0, 0.5, SR, 7.0, base.front_rf_ms)  # wrong token hop
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("mismatching token hop must be refused")
    try:
        _wrap(base, 1.0, 0.5, emit_rule="middle")
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("an unknown emit_rule must be refused")
    print("  batching / scene / rtf / validation ok; describe:", w_big.describe())


# --------------------------------------------------------------------------- (f) BEATs path
def _tiny_beats():
    """A tiny random BEATs (1 layer, 32-d) from ``third_party/BEATs`` -- the real fbank /
    patch-embedding front end, so token counts and flattening are the real ones."""
    beats_dir = os.path.join(_REPO_ROOT, "third_party", "BEATs")
    if not os.path.isfile(os.path.join(beats_dir, "BEATs.py")):
        return None
    if beats_dir not in sys.path:
        sys.path.insert(0, beats_dir)
    from BEATs import BEATs, BEATsConfig  # noqa: WPS433

    torch.manual_seed(7)
    cfg = BEATsConfig({
        "input_patch_size": 16, "embed_dim": 32, "encoder_layers": 1, "encoder_embed_dim": 32,
        "encoder_ffn_embed_dim": 64, "encoder_attention_heads": 4, "conv_pos": 16, "conv_pos_groups": 4,
        "dropout": 0.0, "attention_dropout": 0.0, "dropout_input": 0.0, "encoder_layerdrop": 0.0,
    })
    return BEATs(cfg).eval()


def test_beats_window_model():
    beats = _tiny_beats()
    if beats is None:  # pragma: no cover
        print("  SKIP: third_party/BEATs not available")
        return
    base = BEATsWindowModel(beats)
    assert base.hop == 2560 and base.rf == 2800 and base.centre_offset == 1280 and base.embedding_dim == 32
    assert base.token_hop_ms == 160.0 and base.front_rf_ms == 175.0 and base.centre_offset_ms == 80.0
    # token counts: 8 * (fbank_frames // 16); 2 s -> 96, 10 s -> 496 (not the wrapper's (ms - 5) // 20)
    assert base.n_tokens(32000) == 96 and base.n_frames(32000) == 12
    assert base.n_tokens(160000) == 496 and base.n_frames(160000) == 62
    # a frame needs 16 fbank frames = 400 + 15 x 160 = 2800 samples (175 ms): C = 0.1 s (1600) yields
    # none, C = 0.25 s (4000) exactly one -> hence the C >= 0.25 s rule of hear_configs/BEATs_stream.py
    assert base.n_frames(2799) == 0 and base.n_frames(2800) == 1 and base.n_frames(1600) == 0 and base.n_frames(4000) == 1
    assert (base.frame_timestamps_ms(32000) == 80.0 + 160.0 * torch.arange(12, dtype=torch.float64)).all()
    torch.manual_seed(8)
    x = 0.1 * torch.randn(2, 32000)
    with torch.inference_mode():
        tokens = beats.extract_features(x, padding_mask=None)[0]
    assert tokens.shape == (2, 96, 32), tokens.shape
    out = base.embed_window(x)
    assert out.shape == (2, 12, 32)
    assert torch.allclose(out, tokens.reshape(2, 12, 8, 32).mean(2), atol=1e-6)  # token k = 8 j + b
    cat = BEATsWindowModel(beats, band_pool="concat")
    assert cat.embedding_dim == 256 and torch.equal(cat.embed_window(x), tokens.reshape(2, 12, 256))
    # whole-clip reference and streaming through the wrapper
    L = 3 * SR
    y = 0.1 * torch.randn(1, L)
    full = _wrap(base, math.inf, math.inf)
    emb, ts = full.get_timestamp_embeddings(y)
    assert emb.shape == (1, base.n_frames(L), 32) and torch.equal(emb, base.embed_window(y))
    assert torch.allclose(ts[0].double(), base.frame_timestamps_ms(L), atol=1e-3)
    for rule in RULES:
        for W, C in [(2.0, 0.5), (1.0, 0.25), (2.0, 2.0)]:
            w = _wrap(base, W, C, batch_windows=4, emit_rule=rule)
            e, t = w.get_timestamp_embeddings(y)
            assert e.shape[1] == t.shape[1] == w.n_emitted(L) > 0 and torch.isfinite(e).all()
            assert (t[0, 1:] > t[0, :-1]).all()
            assert all(ch.n_emit >= 1 for ch in w.plan(L)[:-1]), "every full chunk yields >= 1 frame"
            assert w.latency_ms() == C * 1000 + 175.0
    # C = 0.25 s, W = 1 s: the rf_end rule keeps 2 frames per chunk (8 / s; only 1 in the first
    # chunk, whose other candidate starts in the left padding), the centre rule 1 (4 / s)
    assert _wrap(base, 1.0, 0.25, emit_rule="rf_end").n_emitted(L) == 2 * 12 - 1
    assert _wrap(base, 1.0, 0.25, emit_rule="centre").n_emitted(L) == 12
    # the HEAR config refuses C < 0.25 s before touching any checkpoint
    from hear_configs import BEATs_stream

    try:
        BEATs_stream.build_streaming_model("/nonexistent.pt", window_s=2.0, hop_s=0.1)  # below one 160 ms patch
    except ValueError as e:
        assert "0.16" in str(e)
    else:  # pragma: no cover
        raise AssertionError("BEATs C = 0.1 s must be refused")
    print("  BEATs window model ok (96 tokens / 12 frames per 2 s, timestamps 80 + 160 j ms)")


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
