"""HEAR config: BEATs (Chen et al. 2023, ``third_party/BEATs``) under streaming inference.

DESIGN2.md E4.  Environment:

    MODEL_PATH         BEATs checkpoint (default /projects/0/prjs1338/models/BEATs/BEATs_iter3.pt;
                       the path given to load_model() wins)
    STREAM_W           window W in seconds, or ``inf`` = the whole clip (default inf)
    STREAM_C           chunk C in seconds, or ``inf`` (default = W); finite C must be >= 0.25 s
    BEATS_BAND_POOL    mean (default: one 768-d frame per 160 ms time patch = mean of its 8
                       mel-band tokens) | concat (8 x 768 = 6144-d frames)
    STREAM_EMIT        rf_end (default: a chunk emits the frames whose 175 ms receptive field
                       ends inside it -- no frame loss) | centre (DESIGN2's literal rule:
                       frames whose 80 ms centre lies in the chunk; loses the frames in the
                       last 95 ms of every chunk)
    STREAM_BATCH_WINDOWS  windows per forward pass (default 32)
    STREAM_RTF_JSON    if set, the accumulated real-time factor is written there

Frames and timestamps come from BEATs' patch grid, not from the reference HEAR wrapper's
``(ms - 5) // 20`` pseudo-frames (whose count is off by 3 per 2 s): ``extract_features``
emits ``8 * (fbank_frames // 16)`` tokens (2 s -> 96, 10 s -> 496), token ``k`` = mel band
``k % 8`` of time patch ``k // 8``; a frame is one time patch (hop 160 ms, receptive field
175 ms = 25 ms window + 15 x 10 ms) stamped at ``(16 j + 8) x 10 ms = 160 j + 80 ms``.
``W = C = inf`` reproduces the whole-clip non-streaming reference.  ``C < 0.25 s`` is refused:
a chunk shorter than one 160 ms patch + the 25 ms window cannot yield a frame.
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_BEATS_DIR = os.path.join(_REPO, "third_party", "BEATs")
if _BEATS_DIR not in sys.path:
    sys.path.insert(0, _BEATS_DIR)

from BEATs import BEATs, BEATsConfig  # noqa: E402
from hear_api.streaming import BEATsWindowModel, StreamingWrapper  # noqa: E402

SR = 16000
DEFAULT_MODEL_PATH = "/projects/0/prjs1338/models/BEATs/BEATs_iter3.pt"
MIN_HOP_S = 0.16  # one 16-frame time patch = the smallest chunk that can carry a new frame
MIN_WINDOW_S = 0.175  # a window must cover one patch's receptive field (25 ms window + 15 hops)
TOKEN_HOP_MS = 160.0  # one frame per 16-frame time patch
FRONT_RF_MS = 175.0  # 25 ms fbank window + 15 x 10 ms hops


def _parse_seconds(value: str) -> float:
    return float("inf") if value.strip().lower() in ("inf", "full") else float(value)


def build_beats(model_path: str, map_location: Optional[torch.device] = None) -> BEATs:
    """Load a BEATs checkpoint (``{"cfg": ..., "model": state_dict}``) in eval mode."""
    if map_location is None:
        map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)
    cfg = BEATsConfig(checkpoint["cfg"])
    model = BEATs(cfg)
    model.load_state_dict(checkpoint["model"])
    if torch.cuda.is_available():
        model.cuda()
    return model.eval()


def build_streaming_model(
    model_path: str,
    window_s: float = float("inf"),
    hop_s: Optional[float] = None,
    band_pool: str = "mean",
    emit_rule: str = "rf_end",
    batch_windows: int = 32,
    rtf_json: Optional[str] = None,
    map_location: Optional[torch.device] = None,
) -> StreamingWrapper:
    """Checkpoint -> ``StreamingWrapper``; ``hop_s`` defaults to ``window_s``."""
    hop_s = window_s if hop_s is None else hop_s
    # A window of 2800 + 2560 k samples (e.g. 158960 = 9.935 s = 62 patch columns) ends exactly on a patch
    # column, so every chunk - however short - emits the column ending at the chunk end ("dense framing",
    # latency C + 175 ms). Any other W leaves up to 159 ms of a window without a complete column, hence the
    # C >= 0.16 s rule there.
    column_aligned = window_s != float("inf") and (int(round(window_s * SR)) - 2800) % 2560 == 0
    if hop_s != float("inf") and hop_s < MIN_HOP_S - 1e-9 and not column_aligned:
        raise ValueError(
            f"BEATs streaming needs C >= {MIN_HOP_S} s (one 160 ms time patch per chunk) unless W = 2800 + 2560 k "
            f"samples (e.g. 9.935 s); C = {hop_s} s, W = {window_s} s was requested"
        )
    if window_s != float("inf") and window_s < MIN_WINDOW_S - 1e-9:
        raise ValueError(
            f"BEATs streaming needs W >= {MIN_WINDOW_S} s (one patch's 175 ms receptive field); "
            f"W = {window_s} s was requested"
        )
    base = BEATsWindowModel(build_beats(model_path, map_location), band_pool=band_pool)
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
        left_pad=os.environ.get("STREAM_LEFT_PAD", "0") == "1",  # default: no zeros before the clip start
    )
    print(f"[BEATs_stream] {model.describe()} weights={model_path}")
    return model


def load_model(*args, **kwargs) -> StreamingWrapper:
    """HEAR entry point; the streaming configuration is read from the environment."""
    model_path = args[0] if args else os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"BEATs checkpoint not found: {model_path}")
    window_s = _parse_seconds(os.environ.get("STREAM_W", "inf"))
    hop_s = _parse_seconds(os.environ.get("STREAM_C", "inf" if window_s == float("inf") else str(window_s)))
    return build_streaming_model(
        model_path,
        window_s=window_s,
        hop_s=hop_s,
        band_pool=os.environ.get("BEATS_BAND_POOL", "mean"),
        emit_rule=os.environ.get("STREAM_EMIT", "rf_end"),
        batch_windows=int(os.environ.get("STREAM_BATCH_WINDOWS", "32")),
        rtf_json=os.environ.get("STREAM_RTF_JSON") or None,
    )


def get_scene_embeddings(audio: torch.Tensor, model: StreamingWrapper) -> torch.Tensor:
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio: torch.Tensor, model: StreamingWrapper):
    return model.get_timestamp_embeddings(audio)
