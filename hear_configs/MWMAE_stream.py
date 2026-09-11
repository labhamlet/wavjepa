"""HEAR config: MW-MAE (Yadav et al., ICLR 2024; JAX/Flax, ``third_party/MWMAE``) under streaming inference.

Run inside the ``mwmae-eval`` conda environment. Environment variables:

    MW_MAE_MODEL_DIR   run directory with the Flax checkpoint (default
                       /projects/0/prjs1338/models/MWMAE/mwmae_base_200_4x16_8x128_default_bfloat16_run1;
                       the path given to load_model() wins)
    STREAM_W           window W in seconds, or ``inf`` = whole clip (default 1.99 = one 200-frame unit)
    STREAM_C           chunk C in seconds (default = W); finite C must be >= 0.04 s (one 40 ms column)
    STREAM_LEFT_PAD    1 = zero-pad windows before the clip start (default 0: trim)
    MWMAE_BAND_POOL    concat (default, the official wrapper: 5 x 768 = 3840-d frames) or mean (768-d)
    MWMAE_FRAMING      unit (default: the authors' pipeline, a column emitted once its full 55 ms receptive
                       field has arrived, C >= 0.04), all (every column incl. the half-reflected last one,
                       30 ms declared; record only) or end (record only: W = 0.055 + k x 0.04 s, any C)
    MWMAE_NORM         clip (default, official: log-mel mean/std of the whole clip) or window
    MWMAE_UNIT_PAD     reflect (default, official) or zero: how a partial unit is completed (and how the
                       future is padded, see next)
    MWMAE_FUTURE_PAD_MS  end framing only, default 0: pad this much of each emitted column's future
                       (multiple of 10 ms, <= 40) instead of waiting for it; the front-end delay becomes
                       55 - MWMAE_FUTURE_PAD_MS (40 -> 15 ms = WavJEPA's, latency C + 15 ms)
    STREAM_EMIT, STREAM_BATCH_WINDOWS, STREAM_RTF_JSON  as for the other stream configs

Frames: one 3840-d vector (5 mel bands x 768, concatenated as in the original wrapper) per 40 ms
column, delay floor 55 ms (25 ms window + 3 x 10 ms), timestamps at the true column centres.
Scene embeddings = mean over the emitted frames. ``W = C = 1.99`` is the non-streaming reference
(concatenated 200-frame units, the original HEAR wrapper's scheme without its per-clip normalisation).
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from hear_api.streaming import StreamingWrapper  # noqa: E402
from hear_api.streaming_mwmae import DEFAULT_WEIGHTS_DIR, MWMAEWindowModel  # noqa: E402

SR = 16000
UNIT_S = 31840 / 16000  # 199 hops + 1 = 200 fbank frames = 1.99 s (unit framing)
END_W_S = 31600 / 16000  # 880 + 48 x 640 samples = 49 columns = 1.975 s (end framing)
MIN_HOP_S = 0.04  # one 4-frame column
MIN_WINDOW_S = 0.055  # one column's receptive field
TOKEN_HOP_MS = 40.0
FRONT_RF_MS = {"unit": 55.0, "all": 30.0, "end": 55.0}  # declared front-end delay per framing (end: minus the padded future)
CENTRE_OFFSET_MS = {"unit": 15.0, "end": 27.5}


def _parse_seconds(value: str) -> float:
    return float("inf") if value.strip().lower() in ("inf", "full") else float(value)


def build_streaming_model(
    weights_dir: str,
    window_s: float = UNIT_S,
    hop_s: Optional[float] = None,
    emit_rule: str = "rf_end",
    batch_windows: int = 32,
    rtf_json: Optional[str] = None,
    left_pad: bool = False,
    band_pool: str = "concat",
    framing: str = "unit",
    unit_pad: str = "reflect",
    future_pad_ms: float = 0.0,
    norm: str = "clip",
) -> StreamingWrapper:
    future_pad = int(round(future_pad_ms * SR / 1000.0))
    hop_s = window_s if hop_s is None else hop_s
    if framing in ("unit", "all") and hop_s != float("inf") and hop_s < MIN_HOP_S - 1e-9:
        raise ValueError(f"MW-MAE {framing} framing needs C >= {MIN_HOP_S} s (one 40 ms column); C = {hop_s} s was requested")
    if window_s != float("inf") and window_s < MIN_WINDOW_S - 1e-9:
        raise ValueError(f"MW-MAE streaming needs W >= {MIN_WINDOW_S} s; W = {window_s} s was requested")
    if framing == "end" and window_s != float("inf"):
        w = int(round(window_s * SR))
        if (w + future_pad - 880) % 640 != 0:
            k = max((w + future_pad - 880) // 640, 0)
            raise ValueError(f"end framing needs W + future_pad = 880 + k x 640 samples so that the last column ends at the "
                             f"(padded) window end; W = {window_s} s = {w} samples, future_pad = {future_pad} was requested "
                             f"(nearest W: {(880 + k * 640 - future_pad) / SR} s)")
    base = MWMAEWindowModel(weights_dir, band_pool=band_pool, framing=framing, unit_pad=unit_pad, future_pad=future_pad, norm=norm)
    assert base.token_hop_ms == TOKEN_HOP_MS and abs(base.front_rf_ms - (FRONT_RF_MS[framing] - future_pad_ms)) < 1e-6
    model = StreamingWrapper(
        base,
        window_s=window_s,
        hop_s=hop_s,
        sample_rate=SR,
        token_hop_ms=base.token_hop_ms,
        front_rf_ms=base.front_rf_ms,
        max_window_s=None,
        centre_offset_ms=base.centre_offset / SR * 1000.0,  # 15 ms (unit) / rf / 2 (end framing)
        batch_windows=batch_windows,
        rtf_json=rtf_json,
        emit_rule=emit_rule,
        left_pad=left_pad,
    )
    print(f"[MWMAE_stream] {model.describe()} band_pool={band_pool} framing={framing} norm={norm} unit_pad={unit_pad} "
          f"future_pad={future_pad} samples weights={weights_dir}")
    return model


def load_model(*args, **kwargs) -> StreamingWrapper:
    weights_dir = args[0] if args else os.environ.get("MW_MAE_MODEL_DIR", DEFAULT_WEIGHTS_DIR)
    if not os.path.isdir(weights_dir):
        raise FileNotFoundError(f"MW-MAE run directory not found: {weights_dir}")
    framing = os.environ.get("MWMAE_FRAMING", "unit")
    window_s = _parse_seconds(os.environ.get("STREAM_W", str(UNIT_S if framing == "unit" else END_W_S)))
    hop_s = _parse_seconds(os.environ.get("STREAM_C", "inf" if window_s == float("inf") else str(window_s)))
    return build_streaming_model(
        weights_dir,
        window_s=window_s,
        hop_s=hop_s,
        emit_rule=os.environ.get("STREAM_EMIT", "rf_end"),
        batch_windows=int(os.environ.get("STREAM_BATCH_WINDOWS", "32")),
        rtf_json=os.environ.get("STREAM_RTF_JSON") or None,
        left_pad=os.environ.get("STREAM_LEFT_PAD", "0") == "1",
        band_pool=os.environ.get("MWMAE_BAND_POOL", "concat"),
        framing=framing,
        unit_pad=os.environ.get("MWMAE_UNIT_PAD", "reflect"),
        future_pad_ms=float(os.environ.get("MWMAE_FUTURE_PAD_MS", "0")),
        norm=os.environ.get("MWMAE_NORM", "clip"),
    )


def get_scene_embeddings(audio: torch.Tensor, model: StreamingWrapper) -> torch.Tensor:
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio: torch.Tensor, model: StreamingWrapper):
    return model.get_timestamp_embeddings(audio)
