"""HEAR config: EAT-base (Chen et al., IJCAI 2024; ``third_party/EAT``) under streaming inference.

Environment (same conventions as ``hear_configs/BEATs_stream.py``):

    EAT_MODEL_DIR      local snapshot of worstchan/EAT-base_epoch30_pretrain (default
                       /projects/0/prjs1338/models/EAT/EAT-base_epoch30_pretrain; the path given
                       to load_model() wins)
    STREAM_W           window W in seconds, or ``inf`` = the whole clip (default 10, its pre-training length)
    STREAM_C           chunk C in seconds, or ``inf`` (default = W); finite C must be >= 0.16 s
    STREAM_LEFT_PAD    1 = zero-pad windows before the clip start (default 0: trim)
    EAT_BAND_POOL      mean (default) | concat (8 x 768-d frames)
    STREAM_EMIT        rf_end (default) | centre
    STREAM_BATCH_WINDOWS, STREAM_RTF_JSON  as for the other stream configs

Frames = one per 16-frame time patch (hop 160 ms, receptive field 175 ms, stamped 160 j + 80 ms),
exactly the BEATs geometry; the CLS token is dropped and the 8 mel-band tokens of a column are
mean-pooled.  Scene embeddings = mean over the emitted frames.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from hear_api.streaming import EATWindowModel, StreamingWrapper  # noqa: E402
from third_party.EAT.eat_loader import DEFAULT_MODEL_DIR, load_eat  # noqa: E402

SR = 16000
MIN_HOP_S = 0.16  # one 16-frame time patch
MIN_WINDOW_S = 0.175  # one patch's receptive field
TOKEN_HOP_MS = 160.0
FRONT_RF_MS = 175.0
PRETRAIN_S = 10.0


def _parse_seconds(value: str) -> float:
    return float("inf") if value.strip().lower() in ("inf", "full") else float(value)


def build_streaming_model(
    model_dir: str,
    window_s: float = PRETRAIN_S,
    hop_s: Optional[float] = None,
    band_pool: str = "mean",
    emit_rule: str = "rf_end",
    batch_windows: int = 32,
    rtf_json: Optional[str] = None,
    left_pad: bool = False,
) -> StreamingWrapper:
    hop_s = window_s if hop_s is None else hop_s
    if hop_s != float("inf") and hop_s < MIN_HOP_S - 1e-9:
        raise ValueError(f"EAT streaming needs C >= {MIN_HOP_S} s (one 160 ms time patch); C = {hop_s} s was requested")
    if window_s != float("inf") and window_s < MIN_WINDOW_S - 1e-9:
        raise ValueError(f"EAT streaming needs W >= {MIN_WINDOW_S} s; W = {window_s} s was requested")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = EATWindowModel(load_eat(model_dir, device=device), band_pool=band_pool)
    assert base.token_hop_ms == TOKEN_HOP_MS and base.front_rf_ms == FRONT_RF_MS
    model = StreamingWrapper(
        base,
        window_s=window_s,
        hop_s=hop_s,
        sample_rate=SR,
        token_hop_ms=base.token_hop_ms,
        front_rf_ms=base.front_rf_ms,
        max_window_s=None,
        centre_offset_ms=base.centre_offset_ms,
        batch_windows=batch_windows,
        rtf_json=rtf_json,
        emit_rule=emit_rule,
        left_pad=left_pad,
    )
    print(f"[EAT_stream] {model.describe()} weights={model_dir}")
    return model


def load_model(*args, **kwargs) -> StreamingWrapper:
    model_dir = args[0] if args else os.environ.get("EAT_MODEL_DIR", DEFAULT_MODEL_DIR)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"EAT model directory not found: {model_dir}")
    window_s = _parse_seconds(os.environ.get("STREAM_W", str(PRETRAIN_S)))
    hop_s = _parse_seconds(os.environ.get("STREAM_C", "inf" if window_s == float("inf") else str(window_s)))
    return build_streaming_model(
        model_dir,
        window_s=window_s,
        hop_s=hop_s,
        band_pool=os.environ.get("EAT_BAND_POOL", "mean"),
        emit_rule=os.environ.get("STREAM_EMIT", "rf_end"),
        batch_windows=int(os.environ.get("STREAM_BATCH_WINDOWS", "32")),
        rtf_json=os.environ.get("STREAM_RTF_JSON") or None,
        left_pad=os.environ.get("STREAM_LEFT_PAD", "0") == "1",
    )


def get_scene_embeddings(audio: torch.Tensor, model: StreamingWrapper) -> torch.Tensor:
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio: torch.Tensor, model: StreamingWrapper):
    return model.get_timestamp_embeddings(audio)
