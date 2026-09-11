"""HEAR config: ATST-Clip base under streaming inference.\n\nEnv: ATST_MODEL_PATH, STREAM_W (default 6 s = the pre-training segment; inf = whole clip), STREAM_C (>= 0.04),\nSTREAM_EMIT, STREAM_BATCH_WINDOWS, STREAM_RTF_JSON, STREAM_LEFT_PAD. Frames: one 768-d token per 40 ms (64 x 4 patches),\nfront-end delay 94 ms (64 ms analysis window + 3 x 10 ms). band_pool / norm are accepted and ignored."""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from hear_api.streaming import StreamingWrapper  # noqa: E402
from hear_api.streaming_patch import DEFAULT_ATST as DEFAULT_WEIGHTS, ATSTWindowModel  # noqa: E402

SR = 16000
UNIT_S = 6.0
MIN_HOP_S = 0.04
MIN_WINDOW_S = 0.094
TOKEN_HOP_MS = 40.0
FRONT_RF_MS = 94.0
CENTRE_OFFSET_MS = 15.0


def _parse_seconds(value: str) -> float:
    return float("inf") if value.strip().lower() in ("inf", "full") else float(value)


def build_streaming_model(weights: str, window_s: float = UNIT_S, hop_s: Optional[float] = None, emit_rule: str = "rf_end",
                          batch_windows: int = 32, rtf_json: Optional[str] = None, left_pad: bool = False,
                          band_pool: str = "mean", norm: str = "authors") -> StreamingWrapper:
    hop_s = window_s if hop_s is None else hop_s
    if hop_s != float("inf") and hop_s < MIN_HOP_S - 1e-9:
        raise ValueError(f"ATST streaming needs C >= {MIN_HOP_S} s (one token / patch column per chunk); C = {hop_s} s was requested")
    if window_s != float("inf") and window_s < MIN_WINDOW_S - 1e-9:
        raise ValueError(f"ATST streaming needs W >= {MIN_WINDOW_S} s; W = {window_s} s was requested")
    base = ATSTWindowModel(weights)
    assert abs(base.token_hop_ms - TOKEN_HOP_MS) < 1e-6 and abs(base.front_rf_ms - FRONT_RF_MS) < 1e-6
    model = StreamingWrapper(
        base, window_s=window_s, hop_s=hop_s, sample_rate=SR, token_hop_ms=base.token_hop_ms, front_rf_ms=base.front_rf_ms,
        max_window_s=None, centre_offset_ms=CENTRE_OFFSET_MS, batch_windows=batch_windows, rtf_json=rtf_json,
        emit_rule=emit_rule, left_pad=left_pad,
    )
    print(f"[ATST_stream] {model.describe()} band_pool={band_pool} norm={norm} weights={weights}")
    return model


def load_model(*args, **kwargs) -> StreamingWrapper:
    weights = args[0] if args else os.environ.get("ATST_MODEL_PATH", DEFAULT_WEIGHTS)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"ATST weights not found: {weights}")
    window_s = _parse_seconds(os.environ.get("STREAM_W", str(UNIT_S)))
    hop_s = _parse_seconds(os.environ.get("STREAM_C", "inf" if window_s == float("inf") else str(window_s)))
    return build_streaming_model(
        weights, window_s=window_s, hop_s=hop_s, emit_rule=os.environ.get("STREAM_EMIT", "rf_end"),
        batch_windows=int(os.environ.get("STREAM_BATCH_WINDOWS", "32")), rtf_json=os.environ.get("STREAM_RTF_JSON") or None,
        left_pad=os.environ.get("STREAM_LEFT_PAD", "0") == "1",
    )


def get_scene_embeddings(audio: torch.Tensor, model: StreamingWrapper) -> torch.Tensor:
    """Clip-level tasks use ATST-Clip as the authors do: the 12-block CLS + mean concatenation (18432-d,
    ``ATST_SCENE=clip``, default). ``ATST_SCENE=mean`` gives the mean of the streamed 40 ms tokens instead."""
    if os.environ.get("ATST_SCENE", "clip") == "clip":
        if audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio[:, 0, :]
        return model.base.clip_embedding(audio.float())
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio: torch.Tensor, model: StreamingWrapper):
    return model.get_timestamp_embeddings(audio)
