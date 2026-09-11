"""HEAR config: WavJEPA under streaming (windowed, bounded look-ahead) inference.

DESIGN2.md E4.  ``load_model(ckpt_path)`` builds ``hear_api.runtime.RuntimeJEPA`` and wraps
it in ``hear_api.streaming.StreamingWrapper``; the configuration comes from the environment
(the HEAR eval kit only passes the checkpoint path):

    WAVJEPA_EXTRACTOR  wavjepa (10 ms tokens, default) | wav2vec2 (20 ms tokens)
    WAVJEPA_SECONDS    process_seconds of the checkpoint (default 2.01 / 4.02 per extractor)
    STREAM_W           window W in seconds (default = WAVJEPA_SECONDS)
    STREAM_C           chunk C in seconds (default = W)
    STREAM_PAD         zero (default: pad short windows to process_seconds + key-padding
                       mask) | trim (run the unpadded window, fewer tokens)
    STREAM_EMIT        rf_end (default: a chunk emits the frames whose receptive field ends
                       inside it -- no frame loss) | centre (DESIGN2's literal rule: frames
                       whose centre lies in the chunk; loses the frame straddling each chunk end)
    STREAM_BATCH_WINDOWS  windows per forward pass (default 32)
    STREAM_RTF_JSON    if set, the accumulated real-time factor is written there
    WAVJEPA_CKPT       checkpoint path used when load_model() gets no path

Default checkpoints: the PAPER model (375k steps, 100 Hz wavjepa extractor, 2.01 s windows --
exactly what the paper's Tables I-II report) for ``wavjepa``; the 4.02 s wav2vec2-extractor
revision run for ``wav2vec2``.  The revision 2.01 s checkpoints are in ``REVISION_CKPTS``.

``W = C = WAVJEPA_SECONDS`` reproduces the non-streaming ``hear_configs/WavJEPA.py`` numbers
(timestamps are frame centres, i.e. shifted by rf/2 = 7.5 ms w.r.t. the old frame starts).
The window snaps to the runtime's ``unit_frames`` (``int(2.01 * 16000)`` = 32159 samples).
"""
from __future__ import annotations

import os
import sys
from typing import Optional

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import wavjepa  # noqa: E402

sys.modules["sjepa"] = wavjepa  # old checkpoints pickle objects under the ``sjepa`` name

from hear_api.runtime import RuntimeJEPA  # noqa: E402
from hear_api.streaming import StreamingWrapper, WavJEPAWindowModel  # noqa: E402
from wavjepa.extractors import ConvFeatureExtractor  # noqa: E402

SR = 16000
EXTRACTOR_SPECS = {
    # 100 Hz tokens: hop 160 samples (10 ms), receptive field 240 samples (15 ms)
    "wavjepa": [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)],
    # 50 Hz tokens: hop 320 samples (20 ms), receptive field 400 samples (25 ms)
    "wav2vec2": [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] + [(512, 2, 2)],
}
# The paper's model (Tables I-II): 375k steps, 2.01 s, 100 Hz wavjepa extractor.
PAPER_CKPT = (
    "/projects/0/prjs1261/saved_models_jepa_reproduce/SR=16000/LibriRatio=0.0/BatchSize=32/NrSamples=8/"
    "NrGPUs=2/ModelSize=base/LR=0.0004/Masking=time-inverse-masker/TargetProb=0.25/TargetLen=10/"
    "ContextLen=10/TopK=8/step=375000.ckpt"
)
_RUNS = "/scratch-shared/gyuksel2/wavjepa_runs/saved_models_jepa_new_masking/Data=AudioSet16k"
_TAIL = (
    "BatchSize=256/NrSamples=1/NrGPUs=1/LR=0.0004/TargetProb=0.25/TargetLen=10/ContextProb=0.65/"
    "ContextLen=10/MinContextBlock=1/ContextRatio=0.1/Packed=True"
)
REVISION_CKPTS = {
    "wavjepa": f"{_RUNS}/Extractor=wavjepa/InSeconds=2.01/{_TAIL}/step=130000.ckpt",
    "wav2vec2": f"{_RUNS}/Extractor=wav2vec2/InSeconds=4.02/{_TAIL}/step=130000.ckpt",
}
DEFAULT_CKPTS = {"wavjepa": PAPER_CKPT, "wav2vec2": REVISION_CKPTS["wav2vec2"]}
DEFAULT_SECONDS = {"wavjepa": 2.01, "wav2vec2": 4.02}


def _parse_seconds(value: str) -> float:
    return float("inf") if value.strip().lower() in ("inf", "full") else float(value)


def build_runtime(
    ckpt_path: str,
    extractor: str = "wavjepa",
    process_seconds: float = 2.01,
    map_location: Optional[torch.device] = None,
) -> RuntimeJEPA:
    """The very same construction as ``hear_configs/WavJEPA.py`` / ``WavJEPA_w2v2.py``."""
    if extractor not in EXTRACTOR_SPECS:
        raise ValueError(f"WAVJEPA_EXTRACTOR must be one of {list(EXTRACTOR_SPECS)}, got {extractor!r}")
    if map_location is None:
        map_location = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weights = torch.load(ckpt_path, weights_only=False, map_location=map_location)
    conv = ConvFeatureExtractor(conv_layers_spec=EXTRACTOR_SPECS[extractor], in_channels=1)
    return RuntimeJEPA(
        in_channels=1,
        process_seconds=process_seconds,
        weights=weights,
        sr=SR,
        model_size="base",
        is_spectrogram=False,
        extractor=conv,
    )


def build_streaming_model(
    ckpt_path: str,
    extractor: str = "wavjepa",
    process_seconds: float = 2.01,
    window_s: Optional[float] = None,
    hop_s: Optional[float] = None,
    pad_mode: str = "trim",
    emit_rule: str = "rf_end",
    batch_windows: int = 32,
    rtf_json: Optional[str] = None,
    name: Optional[str] = None,
    map_location: Optional[torch.device] = None,
    left_pad: bool = False,
) -> StreamingWrapper:
    """Checkpoint -> ``StreamingWrapper``; ``window_s``/``hop_s`` default to ``process_seconds``.

    Defaults (review resolution R-A): ``pad_mode="trim"`` runs the conv stack on the exact
    window (no zeros reach the GroupNorm statistics) and ``left_pad=False`` trims the first
    windows of a clip at the clip start instead of zero-padding them, so no padding is ever
    seen by the model; a full 2.01 s window is bit-identical to ``RuntimeJEPA``.
    """
    runtime = build_runtime(ckpt_path, extractor, process_seconds, map_location)
    if name is None:
        name = "wavjepa100" if extractor == "wavjepa" else "wavjepa50"
    base = WavJEPAWindowModel(runtime, pad_mode=pad_mode, name=name)
    window_s = base.max_window_s if window_s is None else window_s
    hop_s = window_s if hop_s is None else hop_s
    model = StreamingWrapper(
        base,
        window_s=window_s,
        hop_s=hop_s,
        sample_rate=SR,
        token_hop_ms=base.token_hop_ms,
        front_rf_ms=base.front_rf_ms,
        max_window_s=base.max_window_s,
        batch_windows=batch_windows,
        rtf_json=rtf_json,
        emit_rule=emit_rule,
        left_pad=left_pad,
    )
    print(f"[WavJEPA_stream] {model.describe()} ckpt={ckpt_path}")
    return model


def load_model(*args, **kwargs) -> StreamingWrapper:
    """HEAR entry point; the streaming configuration is read from the environment."""
    extractor = os.environ.get("WAVJEPA_EXTRACTOR", "wavjepa")
    ckpt = args[0] if args else os.environ.get("WAVJEPA_CKPT", DEFAULT_CKPTS.get(extractor))
    if ckpt is None or not os.path.isfile(ckpt):
        raise FileNotFoundError(f"WavJEPA checkpoint not found: {ckpt}")
    seconds = float(os.environ.get("WAVJEPA_SECONDS", DEFAULT_SECONDS.get(extractor, 2.01)))
    window_s = _parse_seconds(os.environ.get("STREAM_W", str(seconds)))
    hop_s = _parse_seconds(os.environ.get("STREAM_C", str(window_s)))
    return build_streaming_model(
        ckpt,
        extractor=extractor,
        process_seconds=seconds,
        window_s=window_s,
        hop_s=hop_s,
        pad_mode=os.environ.get("STREAM_PAD", "trim"),
        emit_rule=os.environ.get("STREAM_EMIT", "rf_end"),
        batch_windows=int(os.environ.get("STREAM_BATCH_WINDOWS", "32")),
        rtf_json=os.environ.get("STREAM_RTF_JSON") or None,
        left_pad=os.environ.get("STREAM_LEFT_PAD", "0") == "1",
    )


def get_scene_embeddings(audio: torch.Tensor, model: StreamingWrapper) -> torch.Tensor:
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio: torch.Tensor, model: StreamingWrapper):
    return model.get_timestamp_embeddings(audio)
