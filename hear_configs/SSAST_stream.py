"""HEAR config: SSAST-Base-Patch-400 under streaming inference (run in the ssast-eval env: timm 0.4.5).\n\nEnv: SSAST_MODEL_PATH, STREAM_W (default = one unit), STREAM_C (>= one token hop),\nSSAST_BAND_POOL mean (default) | concat, SSAST_NORM authors (default) | awsome, SSAST_STATS audioset (default) | esc50, SSAST_UNIT 1024 (default, the pre-training length) | 512 frames (the third-party wrapper), SSAST_STRIDE 16 (default, the pre-training grid) | 10 (the official fine-tuning recipes), STREAM_EMIT, STREAM_BATCH_WINDOWS,\nSTREAM_RTF_JSON, STREAM_LEFT_PAD. Frames: one per 160 ms time patch, front-end delay 175 ms."""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from hear_api.streaming import StreamingWrapper  # noqa: E402
from hear_api.streaming_patch import DEFAULT_SSAST as DEFAULT_WEIGHTS, SSASTWindowModel  # noqa: E402

SR = 16000
UNIT_S = int(os.environ.get("SSAST_UNIT", "1024")) * 160 / 16000  # 10.24 s (the pre-training input length; the third-party wrapper used 512)
MIN_HOP_S = 0.16
MIN_WINDOW_S = 0.175
TOKEN_HOP_MS = 10.0 * int(os.environ.get("SSAST_STRIDE", "16"))  # 160 ms at stride 16
FRONT_RF_MS = 175.0
CENTRE_OFFSET_MS = 80.0


def _parse_seconds(value: str) -> float:
    return float("inf") if value.strip().lower() in ("inf", "full") else float(value)


def build_streaming_model(weights: str, window_s: float = UNIT_S, hop_s: Optional[float] = None, emit_rule: str = "rf_end",
                          batch_windows: int = 32, rtf_json: Optional[str] = None, left_pad: bool = False,
                          band_pool: str = "mean", norm: str = "authors") -> StreamingWrapper:
    hop_s = window_s if hop_s is None else hop_s
    if hop_s != float("inf") and hop_s < MIN_HOP_S - 1e-9:
        raise ValueError(f"SSAST streaming needs C >= {MIN_HOP_S} s (one token / patch column per chunk); C = {hop_s} s was requested")
    if window_s != float("inf") and window_s < MIN_WINDOW_S - 1e-9:
        raise ValueError(f"SSAST streaming needs W >= {MIN_WINDOW_S} s; W = {window_s} s was requested")
    base = SSASTWindowModel(weights, band_pool=band_pool, norm=norm, stats=os.environ.get("SSAST_STATS", "audioset"),
                            unit_frames=int(os.environ.get("SSAST_UNIT", "1024")), stride=int(os.environ.get("SSAST_STRIDE", "16")))
    assert abs(base.token_hop_ms - TOKEN_HOP_MS) < 1e-6 and abs(base.front_rf_ms - FRONT_RF_MS) < 1e-6
    model = StreamingWrapper(
        base, window_s=window_s, hop_s=hop_s, sample_rate=SR, token_hop_ms=base.token_hop_ms, front_rf_ms=base.front_rf_ms,
        max_window_s=None, centre_offset_ms=CENTRE_OFFSET_MS, batch_windows=batch_windows, rtf_json=rtf_json,
        emit_rule=emit_rule, left_pad=left_pad,
    )
    print(f"[SSAST_stream] {model.describe()} band_pool={band_pool} norm={norm} weights={weights}")
    return model


def load_model(*args, **kwargs) -> StreamingWrapper:
    weights = args[0] if args else os.environ.get("SSAST_MODEL_PATH", DEFAULT_WEIGHTS)
    if not os.path.exists(weights):
        raise FileNotFoundError(f"SSAST weights not found: {weights}")
    window_s = _parse_seconds(os.environ.get("STREAM_W", str(UNIT_S)))
    hop_s = _parse_seconds(os.environ.get("STREAM_C", "inf" if window_s == float("inf") else str(window_s)))
    return build_streaming_model(
        weights, window_s=window_s, hop_s=hop_s, emit_rule=os.environ.get("STREAM_EMIT", "rf_end"),
        batch_windows=int(os.environ.get("STREAM_BATCH_WINDOWS", "32")), rtf_json=os.environ.get("STREAM_RTF_JSON") or None,
        left_pad=os.environ.get("STREAM_LEFT_PAD", "0") == "1",
        band_pool=os.environ.get("SSAST_BAND_POOL", "mean"), norm=os.environ.get("SSAST_NORM", "authors"),
    )


def get_scene_embeddings(audio: torch.Tensor, model: StreamingWrapper) -> torch.Tensor:
    """Clip-level tasks use the authors' clip encoder (``ft_avgtok``: mean over ALL tokens of the padded unit after
    the final LayerNorm; units averaged; ``SSAST_SCENE=clip``, default). ``SSAST_SCENE=mean`` averages the
    per-time-step frames of real audio instead."""
    if os.environ.get("SSAST_SCENE", "clip") == "clip":
        if audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio[:, 0, :]
        return model.base.clip_embedding(audio.float())
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio: torch.Tensor, model: StreamingWrapper):
    return model.get_timestamp_embeddings(audio)
