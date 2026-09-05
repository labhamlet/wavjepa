"""
Tests for the packed student path of ``wavjepa.jepa.JEPA`` (DESIGN.md sections C / F)
and for the model side of the native batch contract (section A).

pytest-style, but also runnable as ``python tests/test_packing.py`` (pytest is optional).
CPU only, fp32, tiny model (< 1 min).

What is checked
---------------
* ``pack_tokens`` invariants: ``L`` rule, kept positions gathered in original order,
  pad slots in range and zeroed, row-major ``~pad`` order == ``x[keep]`` order.
* packed vs masked forward on real ``TimeInverseBlockMasker`` (AudioSet config) masks:
  same loss (atol 1e-4), same encoder outputs at the context positions, same decoder
  predictions at every decoder-visible (and hence every target) position. Run for
  ``S=200`` (pad_multiple 32) and ``S=137`` (pad_multiple 64 and 32), in eval/no_grad
  (fast attention path) and with grad enabled (training numerics + backward, and the
  parameter gradients of both paths agree).
* ``_prepare_wave16k`` (GPU-side resample + RMS normalisation, here on CPU) equals the
  legacy per-sample ``torchaudio.transforms.Resample`` -> ``pre_process`` path.
* ``on_after_batch_transfer`` with a native dict batch (stub masker with a ``sample``
  method) returns the legacy tuple layout, is deterministic under the per-rank
  generator, and the legacy tuple batch path still works.
* constructing ``JEPA`` without a masker keeps the state_dict keys / hparams clean.
"""
from __future__ import annotations

import copy
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
import torchaudio  # noqa: E402
from einops import rearrange  # noqa: E402

from data_modules.dataset_functions import pre_process  # noqa: E402
from wavjepa.extractors import ConvFeatureExtractor  # noqa: E402
from wavjepa.jepa import JEPA, RESAMPLE_KWARGS, pack_tokens  # noqa: E402
from wavjepa.masking import TimeInverseBlockMasker  # noqa: E402
from wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG  # noqa: E402

try:  # pytest is optional
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None

torch.set_num_threads(min(4, torch.get_num_threads()))

TINY_CONV = [(64, 10, 5), (64, 3, 2), (64, 3, 2)]
AUDIOSET_CFG = dict(
    target_masks_per_context=4,
    context_mask_prob=0.65,
    context_mask_length=10,
    target_prob=0.25,
    target_length=10,
    ratio_cutoff=0.1,
    channel_based_masking=False,
)
SR = 16000


# --------------------------------------------------------------------------- helpers
def _audio_len_for_tokens(n_tokens: int) -> int:
    """Invert ``TINY_CONV`` so that the extractor yields exactly ``n_tokens`` frames."""
    length = n_tokens
    for _dim, k, stride in reversed(TINY_CONV):
        length = (length - 1) * stride + k
    return length


def _tiny_model(
    S: int = 200,
    pad_multiple: int = 32,
    use_packing: bool = True,
    nr_samples_per_audio: int = 2,
    masker=None,
    seed: int = 0,
) -> JEPA:
    """Small JEPA on CPU/fp32 (d_model 64, 3 enc layers, decoder d_model 32, 2 layers)."""
    torch.manual_seed(seed)
    extractor = ConvFeatureExtractor(conv_layers_spec=TINY_CONV, in_channels=1, depthwise=False)
    audio_len = _audio_len_for_tokens(S)
    model = JEPA(
        feature_extractor=extractor,
        transformer_encoder_layers_cfg=TransformerLayerCFG.create(d_model=64, nhead=4),
        transformer_encoder_cfg=TransformerEncoderCFG.create(num_layers=3),
        transformer_decoder_layers_cfg=TransformerLayerCFG.create(d_model=32, nhead=4),
        transformer_decoder_cfg=TransformerEncoderCFG.create(num_layers=2),
        decoder_embedding_dim=32,
        average_top_k_layers=2,
        resample_sr=SR,
        # +0.5 sample so that int(SR * seconds) never truncates to audio_len - 1
        process_audio_seconds=(audio_len + 0.5) / SR,
        nr_samples_per_audio=nr_samples_per_audio,
        compile_modules=False,
        use_packing=use_packing,
        pad_multiple=pad_multiple,
        masker=masker,
    )
    assert model.target_length == audio_len
    assert model.total_patches == S, (model.total_patches, S)
    return model.eval()


def _audioset_masks(B: int, S: int, seed: int = 0):
    """Reference (loop) masker; ``compute_mask_indices`` uses numpy's global-less rng."""
    torch.manual_seed(seed)
    masker = TimeInverseBlockMasker(**AUDIOSET_CFG)
    ctx, tgt, ctx_tgt = masker(batch_size=B, n_times=S, in_channels=1)
    return ctx.bool(), tgt.bool(), ctx_tgt.bool()


def _unpack(packed: torch.Tensor, idx: torch.Tensor, pad: torch.Tensor, S: int):
    """Scatter a packed ``(R, L, E)`` tensor back to ``(R, S, E)``; also return the
    ``(R, S)`` bool mask of positions that were actually filled."""
    R, L, E = packed.shape
    full = torch.zeros(R, S, E, dtype=packed.dtype)
    filled = torch.zeros(R, S, dtype=torch.bool)
    rows = torch.arange(R)[:, None].expand(R, L)
    keep = ~pad
    full[rows[keep], idx[keep]] = packed[keep]
    filled[rows[keep], idx[keep]] = True
    return full, filled


def _random_keep(B: int, S: int, ratio: float, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    keep = torch.rand(B, S, generator=g) < ratio
    keep[:, 0] = True  # never an empty row
    return keep


def _both_paths(model: JEPA, audio, ctx, tgt, ctx_tgt):
    """Run the masked (old) and packed (new) forward on the same inputs."""
    model.use_packing = False
    out_old = model(audio, ctx, tgt, ctx_tgt)
    model.use_packing = True
    out_new = model(audio, ctx, tgt, ctx_tgt)
    return out_old, out_new


def _compare_paths(model: JEPA, out_old, out_new, ctx, tgt, ctx_tgt, atol=1e-4, rtol=1e-4):
    """Loss / context outputs / predictions of the two paths must agree."""
    B, N, S = tgt.shape
    # loss
    assert torch.isfinite(out_new["loss"]) and torch.isfinite(out_old["loss"])
    assert torch.allclose(out_new["loss"], out_old["loss"], atol=atol, rtol=0), (
        f"loss packed={out_new['loss'].item():.6f} masked={out_old['loss'].item():.6f}"
    )
    # encoder outputs at the context positions (already mapped to the decoder width)
    cf_old, cf_new = out_old["contextual_features"], out_new["contextual_features"]
    assert cf_old.shape == cf_new.shape == ((~ctx).sum().item(), model.decoder_embedding_dim)
    assert torch.allclose(cf_old, cf_new, atol=atol, rtol=rtol), (
        f"context features max abs diff {(cf_old - cf_new).abs().max().item():.2e}"
    )
    # targets: the teacher is identical in both paths
    assert torch.allclose(out_old["targets"], out_new["targets"], atol=0, rtol=0)
    # predictions: packed (B*N, L2, E) -> full (B*N, S, E); compare on target positions
    preds_old = out_old["preds"]
    preds_full, filled = _unpack(out_new["preds"], out_new["preds_idx"], out_new["preds_pad"], S)
    assert preds_old.shape == preds_full.shape == (B * N, S, model.encoder_embedding_dim)
    tgt_flat = rearrange(tgt, "B N S -> (B N) S")
    visible = ~rearrange(ctx_tgt, "B N S -> (B N) S")
    assert torch.equal(filled, visible), "packed decoder slots != decoder-visible positions"
    assert bool((visible | ~tgt_flat).all()), "a target position is not decoder-visible"
    diff_tgt = (preds_full[tgt_flat] - preds_old[tgt_flat]).abs().max().item()
    assert torch.allclose(preds_full[tgt_flat], preds_old[tgt_flat], atol=atol, rtol=rtol), (
        f"preds at target positions max abs diff {diff_tgt:.2e}"
    )
    assert torch.allclose(preds_full[visible], preds_old[visible], atol=atol, rtol=rtol), (
        f"preds at visible positions max abs diff "
        f"{(preds_full[visible] - preds_old[visible]).abs().max().item():.2e}"
    )
    # packed length rule
    L2 = out_new["preds"].shape[1]
    n_vis_max = int(visible.sum(1).max().item())
    assert L2 >= n_vis_max and (L2 == S or L2 % model.pad_multiple == 0)
    assert L2 - n_vis_max < model.pad_multiple or L2 == S


# --------------------------------------------------------------------------- pack_tokens
def test_pack_tokens_invariants():
    B, S, E, pm = 5, 137, 8, 32
    x = torch.randn(B, S, E)
    keep = _random_keep(B, S, 0.4, seed=3)
    keep[1] = True                       # a full row
    keep[2] = False
    keep[2, 17] = True                   # a single kept token
    n_keep = keep.sum(1)
    xp, idx, pad = pack_tokens(x, keep, pm)

    # L rule
    L = xp.shape[1]
    assert L == min(S, -(-int(n_keep.max()) // pm) * pm) == idx.shape[1] == pad.shape[1]
    assert idx.dtype == torch.int64 and pad.dtype == torch.bool
    assert idx.min() >= 0 and idx.max() < S, "pad slots must point at in-range positions"
    for b in range(B):
        nb = int(n_keep[b])
        assert int((~pad[b]).sum()) == nb
        assert torch.equal(pad[b], torch.arange(L) >= nb)
        # kept positions, in original order, no duplicates
        assert torch.equal(idx[b, :nb], torch.nonzero(keep[b], as_tuple=True)[0])
        assert idx[b, :nb].unique().numel() == nb
        assert torch.equal(xp[b, :nb], x[b, keep[b]])
        assert bool((xp[b, nb:] == 0).all()), "pad slots are zeroed"
    # row-major order of ~pad == row-major order of keep
    assert torch.equal(xp[~pad], x[keep])

    # pad_multiple larger than S -> L == S
    xp2, idx2, pad2 = pack_tokens(x, keep, 1024)
    assert xp2.shape[1] == S and idx2.shape[1] == S
    # every row full -> no pad at all
    xp3, _, pad3 = pack_tokens(x, torch.ones(B, S, dtype=torch.bool), pm)
    assert xp3.shape[1] == S and not bool(pad3.any()) and torch.equal(xp3, x)


# --------------------------------------------------------------------------- packed == masked
def _check_equivalence(S: int, pad_multiple: int, B: int = 4, seed: int = 0):
    model = _tiny_model(S=S, pad_multiple=pad_multiple, seed=seed)
    torch.manual_seed(seed + 1)
    audio = torch.randn(B, 1, model.target_length)
    ctx, tgt, ctx_tgt = _audioset_masks(B, S, seed=seed)
    with torch.no_grad():
        # encoder alone: packed output == masked output at the context positions
        local = model.extract_audio(audio)
        local = model.feature_norms(local)
        if model.post_extraction_mapper is not None:
            local = model.post_extraction_mapper(local)
        local = local + model.pos_encoding_encoder
        enc_old = model.encoder_forward(local, src_key_padding_mask=ctx)[~ctx]
        enc_new = model.encoder_forward_packed(local, ctx)
        assert enc_old.shape == enc_new.shape == ((~ctx).sum().item(), model.encoder_embedding_dim)
        assert torch.allclose(enc_old, enc_new, atol=1e-4, rtol=1e-4), (
            f"encoder max abs diff {(enc_old - enc_new).abs().max().item():.2e}"
        )
        out_old, out_new = _both_paths(model, audio, ctx, tgt, ctx_tgt)
    _compare_paths(model, out_old, out_new, ctx, tgt, ctx_tgt)

    # report the packed shapes / fractions (visible in `pytest -s` and the plain run)
    T = tgt.shape[1]
    L = -(-int((~ctx).sum(1).max()) // pad_multiple) * pad_multiple
    L = min(S, L)
    L2 = out_new["preds"].shape[1]
    print(
        f"[S={S} pad_multiple={pad_multiple} B={B} T={T}] "
        f"local_features {tuple(out_new['local_features'].shape)} -> "
        f"encoder packed (B, L, E)=({B}, {L}, {model.encoder_embedding_dim}) "
        f"ctx_flat {tuple(out_new['contextual_features'].shape)} -> "
        f"decoder packed (B*T, L2, E_dec)=({B * T}, {L2}, {model.decoder_embedding_dim}) -> "
        f"preds {tuple(out_new['preds'].shape)} idx/pad {tuple(out_new['preds_idx'].shape)} | "
        f"mean ctx/S={((~ctx).sum(1).float().mean() / S).item():.3f} L/S={L / S:.3f} "
        f"mean visible/S={((~ctx_tgt).sum((1, 2)).float().mean() / (T * S)).item():.3f} L2/S={L2 / S:.3f}"
    )
    return model, audio, (ctx, tgt, ctx_tgt)


def test_packed_equals_masked_S200_pad32():
    _check_equivalence(S=200, pad_multiple=32)


def test_packed_equals_masked_S137_pad64():
    _check_equivalence(S=137, pad_multiple=64, seed=1)


def test_packed_equals_masked_S137_pad32():
    _check_equivalence(S=137, pad_multiple=32, B=3, seed=2)


def test_packed_equals_masked_with_grad_and_backward():
    """Grad enabled -> the regular (non fast-path) attention kernels, like training.
    Loss, outputs and the parameter gradients of both paths must agree."""
    S, B = 200, 3
    model = _tiny_model(S=S, pad_multiple=32, seed=5)
    torch.manual_seed(6)
    audio = torch.randn(B, 1, model.target_length)
    ctx, tgt, ctx_tgt = _audioset_masks(B, S, seed=5)

    model.use_packing = False
    out_old = model(audio, ctx, tgt, ctx_tgt)
    out_old["loss"].backward()
    grads_old = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    model.use_packing = True
    out_new = model(audio, ctx, tgt, ctx_tgt)
    out_new["loss"].backward()
    grads_new = {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)

    with torch.no_grad():
        _compare_paths(model, out_old, out_new, ctx, tgt, ctx_tgt)
    assert "mask_token" in grads_new and torch.isfinite(grads_new["mask_token"]).all()
    assert set(grads_old) == set(grads_new), "different parameters received gradients"
    for name, g_old in grads_old.items():
        g_new = grads_new[name]
        assert torch.isfinite(g_new).all(), name
        scale = g_old.abs().max().item() + 1e-6
        assert torch.allclose(g_old, g_new, atol=1e-4 * scale + 1e-6, rtol=1e-3), (
            f"grad mismatch for {name}: max abs diff {(g_old - g_new).abs().max().item():.2e} "
            f"(scale {scale:.2e})"
        )


def test_forward_return_keys():
    model, audio, (ctx, tgt, ctx_tgt) = _check_equivalence(S=137, pad_multiple=32, B=2, seed=3)
    with torch.no_grad():
        out_old, out_new = _both_paths(model, audio, ctx, tgt, ctx_tgt)
    base = {"loss", "local_features", "contextual_features", "preds", "targets"}
    assert base <= set(out_old) and base <= set(out_new)
    assert {"preds_idx", "preds_pad"} <= set(out_new)
    assert out_new["preds_idx"].dtype == torch.int64 and out_new["preds_pad"].dtype == torch.bool


# --------------------------------------------------------------------------- native batch path
def _legacy_resampler(sr: int) -> torchaudio.transforms.Resample:
    return torchaudio.transforms.Resample(sr, SR, dtype=torch.float32, **RESAMPLE_KWARGS)


def _native_batch(seed: int = 0):
    """Two 10 s rows at 44.1k / 48k, a short 16 kHz row and a silent row."""
    g = torch.Generator().manual_seed(seed)
    t44 = torch.arange(441000, dtype=torch.float32) / 44100
    a = 0.3 * torch.sin(2 * torch.pi * 440.0 * t44) + 0.05 * torch.randn(441000, generator=g)
    b = 0.2 * torch.randn(480000, generator=g)
    c = 0.1 * torch.randn(100000, generator=g)            # 6.25 s at 16 kHz, zero padded
    d = torch.zeros(160000)                                # rms == 0 -> unchanged
    rows = [(a, 44100), (b, 48000), (c, SR), (d, SR)]
    l_max = max(sr * 10 for _, sr in rows)
    audio = torch.zeros(len(rows), l_max)
    for i, (x, _) in enumerate(rows):
        audio[i, : x.numel()] = x
    batch = {
        "audio": audio,
        "sr": torch.tensor([sr for _, sr in rows], dtype=torch.int64),
        "length": torch.tensor([x.numel() for x, _ in rows], dtype=torch.int64),
    }
    return batch, rows


def _legacy_prepare(x: torch.Tensor, sr: int) -> torch.Tensor:
    """The legacy per-sample CPU path: cached Resample -> pre_process (normalise, pad/crop)."""
    if sr != SR:
        x = _legacy_resampler(sr)(x)
    return pre_process(x, SR)[0]


def test_prepare_wave16k_matches_legacy_path():
    model = _tiny_model(S=137)
    batch, rows = _native_batch()
    wave, valid = model._prepare_wave16k(batch)
    assert wave.shape == (len(rows), SR * 10) and wave.dtype == torch.float32
    assert valid.tolist() == [160000, 160000, 100000, 160000]
    for i, (x, sr) in enumerate(rows):
        ref = _legacy_prepare(x, sr)
        diff = (wave[i] - ref).abs().max().item()
        assert torch.allclose(wave[i], ref, atol=1e-4, rtol=1e-4), f"row {i} (sr={sr}) max abs diff {diff:.2e}"
        if i < 3:
            rms = wave[i, : valid[i]].pow(2).mean().sqrt()
            assert abs(20 * torch.log10(rms).item() + 14.0) < 1e-3, "not -14 dBFS over the valid samples"
            assert bool((wave[i, valid[i]:] == 0).all())
    assert bool((wave[3] == 0).all()), "silent row must be left unchanged"
    # a second call reuses the cached resamplers (one per foreign sr)
    assert sorted(model._resamplers) == [44100, 48000]
    wave2, _ = model._prepare_wave16k(batch)
    assert torch.equal(wave, wave2)
    # the resamplers are not registered as submodules / in the state_dict
    assert "_resamplers" not in dict(model.named_modules())
    assert not any(k.startswith(("_resamplers", "masker")) for k in model.state_dict())


class _StubMasker(torch.nn.Module):
    """Minimal stand-in for ``TimeInverseBlockMasker.sample`` (random masks, right layout)."""

    def __init__(self, n_targets: int = 4):
        super().__init__()
        self.n_targets = n_targets
        self.calls: list[dict] = []

    def sample(self, batch_size, n_times, in_channels=1, device=None, generator=None):
        self.calls.append(dict(batch_size=batch_size, n_times=n_times, in_channels=in_channels,
                               device=device, generator=generator))
        S = n_times // in_channels
        tgt = torch.rand(batch_size, self.n_targets, S, device=device, generator=generator) < 0.25
        ctx = (torch.rand(batch_size, S, device=device, generator=generator) < 0.5) & ~tgt.any(1)
        ctx[:, 0] = True
        ctx_mask = ~ctx
        return ctx_mask, tgt, torch.logical_xor(ctx_mask[:, None], tgt)


def test_on_after_batch_transfer_dict_batch():
    S, n_crops = 137, 3
    masker = _StubMasker()
    model = _tiny_model(S=S, nr_samples_per_audio=n_crops, masker=masker)
    assert "masker" not in model.hparams and not any(k.startswith("masker") for k in model.state_dict())
    batch, rows = _native_batch()
    B = len(rows)

    torch.manual_seed(123)
    model._generator = None
    audio, ctx, tgt, ctx_tgt = model.on_after_batch_transfer(batch, 0)
    assert audio.shape == (B * n_crops, 1, model.target_length) and audio.dtype == torch.bfloat16
    assert ctx.shape == (B * n_crops, S) and ctx.dtype == torch.bool
    assert tgt.shape == (B * n_crops, masker.n_targets, S) and tgt.dtype == torch.bool
    assert ctx_tgt.shape == (B * n_crops, masker.n_targets, S) and ctx_tgt.dtype == torch.bool
    assert torch.equal(ctx_tgt, torch.logical_xor(ctx[:, None], tgt))
    assert torch.isfinite(audio.float()).all()
    # per-crop standardisation happened (mean ~0, std ~1) for the non-silent crops.
    # Rows a/b are 10 s of signal (all crops non-silent); row c is 6.25 s zero padded
    # (a crop may fall into the padding); row d is silent -> std 0.
    a = audio.float()
    non_silent = a.std(dim=(-2, -1)) > 0.5
    assert 2 * n_crops <= int(non_silent.sum()) <= 3 * n_crops
    assert (a[non_silent].mean(dim=(-2, -1)).abs() < 0.05).all()
    assert ((a[non_silent].std(dim=(-2, -1)) - 1).abs() < 0.05).all()
    # masker got the device / generator of the batch, per-rank generator lives on the CPU here
    call = masker.calls[-1]
    assert call["batch_size"] == B * n_crops and call["n_times"] == S and call["in_channels"] == 1
    assert torch.device(call["device"]).type == "cpu" and call["generator"] is model._generator
    assert torch.device(model._generator.device).type == "cpu"

    # deterministic under the same seed (generator is re-created from torch.initial_seed())
    torch.manual_seed(123)
    model._generator = None
    audio2, ctx2, tgt2, ctx_tgt2 = model.on_after_batch_transfer(batch, 0)
    assert torch.equal(audio, audio2) and torch.equal(ctx, ctx2) and torch.equal(tgt, tgt2)
    assert torch.equal(ctx_tgt, ctx_tgt2)
    # ... and different draws on the next call (the generator advances, not re-seeded)
    audio3, *_ = model.on_after_batch_transfer(batch, 0)
    assert not torch.equal(audio, audio3)

    # the legacy tuple path is still available and untouched in layout
    N = masker.n_targets
    legacy = (
        torch.randn(B, 1, SR * 10),
        torch.rand(B, n_crops, S) < 0.5,
        torch.rand(B, n_crops, N, S) < 0.3,
        torch.rand(B, n_crops, N, S) < 0.5,
    )
    la, lc, lt, lct = model.on_after_batch_transfer(legacy, 0)
    assert la.shape == audio.shape and la.dtype == torch.bfloat16
    assert lc.shape == ctx.shape and lt.shape == tgt.shape and lct.shape == ctx_tgt.shape

    # a dict batch without a masker is a clear error
    model_no_masker = _tiny_model(S=S, nr_samples_per_audio=n_crops)
    try:
        model_no_masker.on_after_batch_transfer(batch, 0)
    except RuntimeError as e:
        assert "masker" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected a RuntimeError without a masker")


def test_model_without_masker_loads_own_state_dict():
    """hear_api / denoise construct JEPA without a masker and load old checkpoints."""
    model = _tiny_model(S=137)
    sd = copy.deepcopy(model.state_dict())
    other = _tiny_model(S=137, seed=1)
    missing, unexpected = other.load_state_dict(sd, strict=True)
    assert not missing and not unexpected
    assert model.masker is None and model.use_packing is True and model.pad_multiple == 32
    with torch.no_grad():
        x = torch.randn(2, 1, model.target_length)
        rep = other.get_audio_representation(x, torch.zeros(2, 137, dtype=torch.bool))
    assert rep.shape == (2, 137, 64)


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
