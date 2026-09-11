"""
Tests for the E3 latency adapters (``benchmarks/latency/models.py``) and the analytic delay
table (``benchmarks/latency/delay_floor.py``).

CPU only, 1 s inputs, tiny compute; the real checkpoints are loaded when available.  A model
whose checkpoint / bundle is missing is SKIPPED with the reason printed -- the file never
fails on missing weights.  Runnable without pytest::

    python tests/test_latency_adapters.py                    # all models
    python tests/test_latency_adapters.py --models beats,wavjepa100 --threads 4

Checks per adapter:
  * ``transformer(front_end(1 s noise))`` has shape ``(1, n_tokens(1.0), D)`` and ``n_tokens``
    matches the analytic count (BEATs also: 2 s -> 96 tokens; +-1 vs the HEAR wrapper formula);
  * WavJEPA: output equals ``RuntimeJEPA.get_timestamp_embeddings`` on the same input
    (allclose 1e-4) for one window and for a multi-window input;
  * BEATs / torchaudio: the two phases reproduce the model's own end-to-end forward;
  * ``count_phase_flops`` returns numbers (or "n/a") without raising.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Sequence

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from benchmarks.latency import delay_floor  # noqa: E402
from benchmarks.latency import models as M  # noqa: E402

os.environ.setdefault("TORCH_HOME", M.DEFAULT_TORCH_HOME)
SR = M.SAMPLE_RATE


def _wave(seconds: float, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return 0.1 * torch.randn(1, int(round(seconds * SR)), generator=g)


# --------------------------------------------------------------------------- #
# Weight-free checks
# --------------------------------------------------------------------------- #
def test_analytic_formulas() -> None:
    """Token / receptive-field formulas and the delay-floor constants (no weights needed)."""
    assert M.conv_out_len(32160, M.WAVJEPA100_SPEC) == 200  # 2.01 s window, 100 Hz
    assert M.conv_out_len(64320, M.WAVJEPA50_SPEC) == 200  # 4.02 s window, 50 Hz
    assert M.conv_out_len(32160, M.WAVJEPA50_SPEC) == 100  # 2.01 s / 50 Hz checkpoint
    assert M.conv_out_len(16000, M.W2V2_SPEC) == 49
    assert M.conv_receptive_field(M.WAVJEPA100_SPEC) == 240 and M.conv_stride(M.WAVJEPA100_SPEC) == 160
    assert M.conv_receptive_field(M.WAVJEPA50_SPEC) == 400 and M.conv_stride(M.WAVJEPA50_SPEC) == 320
    assert M.beats_n_frames(32000) == 198 and M.beats_n_tokens(32000) == 96
    assert M.beats_n_tokens(16000) == 48 and M.beats_n_tokens(160000) == 496
    # runtime bookkeeping mirror. NB: RuntimeJEPA's window is int(2.01 * 16000) = 32159 samples
    # (float rounding), which is what the checkpoint was trained with; 1 s -> 1 window, 200
    # tokens processed, 100 emitted (values verified against RuntimeJEPA in _check_wavjepa).
    unit = int(2.01 * SR)
    assert unit == 32159 and M.conv_out_len(unit, M.WAVJEPA100_SPEC) == 200
    assert M.wavjepa_windows_and_cut_off(16000, unit, 200, SR, unit) == (1, 200, 100)
    assert M.wavjepa_windows_and_cut_off(32000, unit, 200, SR, unit) == (1, 200, 200)
    assert M.wavjepa_windows_and_cut_off(40000, unit, 200, SR, unit) == (2, 400, 249)
    assert M.wavjepa_windows_and_cut_off(72000, unit, 200, SR, unit) == (3, 600, 448)
    assert M.wavjepa_windows_and_cut_off(160000, unit, 200, SR, unit) == (5, 1000, 996)
    floors = {n: s.delay_floor_ms for n, s in M.MODEL_SPECS.items()}
    assert floors == {"wavjepa100": 15.0, "wavjepa50": 25.0, "beats": 175.0, "wav2vec2": 25.0, "hubert": 25.0, "wavlm": 25.0}, floors
    assert abs(M.MODEL_SPECS["beats"].tokens_per_s - 50.0) < 1e-9
    assert abs(M.MODEL_SPECS["wavjepa100"].tokens_per_s - 100.0) < 1e-9
    delay_floor.verify()  # ConvFeatureExtractor.receptive_fields / forward assertions
    rows = delay_floor.compute_rows()
    by_name = {r["model"]: r for r in rows}
    assert by_name["beats"]["tokens @2 s"] == 96 and by_name["beats"]["tokens @10 s"] == 496
    assert by_name["wavjepa100"]["tokens @2 s"] == 200 and by_name["wavjepa100"]["tokens @10 s"] == 996
    assert by_name["wav2vec2"]["tokens @10 s"] == 499
    assert "EAT" in by_name and by_name["EAT"]["delay floor ms"] == "not run"


# --------------------------------------------------------------------------- #
# Adapter checks (skip on missing weights)
# --------------------------------------------------------------------------- #
def _cpu_safe_feature_extractor(in_channels: int = 1) -> torch.nn.Module:
    """``hear_api.feature_helper.FeatureExtractor`` calls ``.cuda()`` unconditionally; this test-only
    subclass keeps the tensors on the CPU when there is no GPU (the runtime code is untouched)."""
    from hear_api.feature_helper import FeatureExtractor  # noqa: WPS433

    class _CpuSafe(FeatureExtractor):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self._wav2feature(x)
            return out.cuda() if torch.cuda.is_available() else out

    return _CpuSafe(in_channels=in_channels)


def _check_common(adapter: M.Adapter, seconds: float = 1.0) -> torch.Tensor:
    wave = _wave(seconds)
    with torch.inference_mode():
        feats = adapter.front_end(wave)
        out = adapter.transformer(feats)
    expected = (1, adapter.n_tokens(seconds), adapter.embed_dim)
    assert tuple(out.shape) == expected, f"{adapter.name}: got {tuple(out.shape)}, expected {expected}"
    assert torch.isfinite(out).all(), f"{adapter.name}: non-finite output"
    assert adapter.params > 1e6 and adapter.params_front > 0 and adapter.params_transformer > 0
    return out


def _check_wavjepa(adapter: M.WavJEPAAdapter) -> None:
    rt = adapter.runtime
    rt.feature_extractor = _cpu_safe_feature_extractor(in_channels=1)  # test-only CPU shim, runtime code untouched
    for seconds in (1.0, 1.5 * adapter.fixed_window_s):  # one window, and a multi-window input
        wave = _wave(seconds, seed=1)
        with torch.inference_mode():
            out = adapter.transformer(adapter.front_end(wave))
            ref, ts = rt.get_timestamp_embeddings(wave)
        assert out.shape == ref.shape == (1, adapter.n_tokens(seconds), adapter.embed_dim), (out.shape, ref.shape)
        assert ts.shape[1] == out.shape[1]
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), f"max diff {(out - ref).abs().max().item():.3e}"
        n_win = adapter.n_tokens_processed(seconds) // adapter.output_steps
        assert n_win == (1 if seconds == 1.0 else 2), (seconds, n_win)
        print(f"    {seconds:.2f} s: adapter == RuntimeJEPA.get_timestamp_embeddings ({n_win} window(s), {out.shape[1]} tokens)")


def _check_beats(adapter: M.BEATsAdapter) -> None:
    assert adapter.n_tokens(2.0) == 96, adapter.n_tokens(2.0)
    assert adapter.n_tokens(1.0) == 48 and adapter.n_tokens(10.0) == 496
    assert abs(adapter.n_tokens(10.0) / 10.0 - adapter.spec.tokens_per_s) <= 1.0  # ~50 tokens/s (8 per 160 ms)
    wave = _wave(2.0, seed=2)
    with torch.inference_mode():
        out = adapter.transformer(adapter.front_end(wave))
        ref = adapter.model.extract_features(wave)[0]
    assert out.shape == (1, 96, adapter.embed_dim) and torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    print("    2 s: 96 tokens; phases == BEATs.extract_features")


def _check_torchaudio(adapter: M.TorchaudioAdapter) -> None:
    assert adapter.n_tokens(1.0) == 49 and adapter.n_tokens(10.0) == 499
    wave = _wave(1.0, seed=3)
    with torch.inference_mode():
        out = adapter.transformer(adapter.front_end(wave))
        ref = adapter.model(wave)[0]
    assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4)
    print("    1 s: 49 tokens; phases == Wav2Vec2Model.forward")


def _check_flops(adapter: M.Adapter) -> None:
    fe, tr = M.count_phase_flops(adapter, _wave(1.0))
    assert fe is None or fe > 0
    assert tr is None or tr > 1e9, tr  # a 12-layer base transformer on >= 48 tokens is > 1 GFLOP
    print(f"    FLOPs (1 s): FE {'n/a' if fe is None else f'{fe / 1e9:.2f} G'}  TR {'n/a' if tr is None else f'{tr / 1e9:.2f} G'}")


def test_adapter(name: str, ckpt: Optional[str] = None) -> Optional[str]:
    """Returns None on pass, or the skip reason (raises on a real failure)."""
    t0 = time.time()
    try:
        adapter = M.build_adapter(name, "cpu", ckpt=ckpt)
    except M.ModelUnavailable as exc:
        return str(exc)
    print(f"  {name}: loaded in {time.time() - t0:.1f} s ({adapter.params / 1e6:.1f} M params, D={adapter.embed_dim}, "
          f"window={adapter.fixed_window_s})")
    try:
        _check_common(adapter, 1.0)
        if isinstance(adapter, M.WavJEPAAdapter):
            _check_wavjepa(adapter)
        elif isinstance(adapter, M.BEATsAdapter):
            _check_beats(adapter)
        elif isinstance(adapter, M.TorchaudioAdapter):
            _check_torchaudio(adapter)
        _check_flops(adapter)
    finally:
        del adapter
        gc.collect()
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default="all", help=f"comma-separated subset of {M.MODEL_NAMES} or 'all'")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--wavjepa100-ckpt", default=None)
    p.add_argument("--wavjepa50-ckpt", default=None)
    p.add_argument("--beats-ckpt", default=None)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    names = M.MODEL_NAMES if args.models == "all" else [m.strip() for m in args.models.split(",") if m.strip()]
    ckpts: Dict[str, Optional[str]] = {"wavjepa100": args.wavjepa100_ckpt, "wavjepa50": args.wavjepa50_ckpt, "beats": args.beats_ckpt}

    passed: List[str] = []
    skipped: Dict[str, str] = {}
    failed: Dict[str, str] = {}
    print("test_analytic_formulas ...", flush=True)
    try:
        test_analytic_formulas()
        passed.append("analytic_formulas")
        print("  ok")
    except Exception:  # noqa: BLE001
        failed["analytic_formulas"] = traceback.format_exc()
        print(failed["analytic_formulas"])
    for name in names:
        print(f"test_adapter[{name}] ...", flush=True)
        try:
            why = test_adapter(name, ckpts.get(name))
        except Exception:  # noqa: BLE001
            failed[name] = traceback.format_exc()
            print(failed[name])
            continue
        if why is None:
            passed.append(name)
            print("  ok")
        else:
            skipped[name] = why
            print(f"  SKIP: {why}")
    print()
    print(f"passed: {passed}")
    print(f"skipped: {skipped}")
    print(f"failed: {list(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
