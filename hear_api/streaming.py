"""
Streaming (windowed, bounded look-ahead) inference for HEAR-style embedding models.

Design contract: DESIGN2.md section E4.  ``hear_api/runtime.py`` is untouched; this module
adds a wrapper that re-uses ``RuntimeJEPA`` (WavJEPA) or a BEATs model as a *window model*
and turns any clip into a sequence of frame embeddings whose look-ahead is bounded.

Terminology (all in seconds unless stated; ``sr`` samples per second)
---------------------------------------------------------------------
* ``W`` (``window_s``): length of the audio window handed to the model for one chunk.
* ``C`` (``hop_s``): chunk length = how often the stream is advanced.  A frame is emitted
  when the chunk it belongs to is complete, so a frame sees at most ``C`` s of look-ahead
  (plus the front-end receptive field) and ``W - C`` s of left context.
* token hop ``hop`` (samples) and front-end receptive field ``rf`` (samples): frame ``i`` of a
  window that starts at sample ``s`` depends on samples ``[s + i*hop, s + i*hop + rf)`` and
  is stamped with the centre ``s + i*hop + centre_offset`` (``centre_offset = rf / 2`` unless
  the base model says otherwise).

Sliding-window algorithm (``StreamingWrapper.plan``)
---------------------------------------------------
Seconds are converted to samples with ``int(x * sr)`` -- the truncating convention of
``RuntimeJEPA`` / ``JEPA`` (``2.01 * 16000`` is ``32159.999...`` -> 32159 samples, ``4.02 s``
-> 64319) -- and a window within one sample of the base model's maximum window
(``RuntimeJEPA.unit_frames`` = ``JEPA.target_length``) snaps to exactly that length, so that
``W = C = process_seconds`` uses precisely the runtime's windows whatever the floating-point
representation of the seconds.  For chunk ``k`` (``t = k*C``, while ``t < duration``):

1. window end ``e = min(t + C, duration)`` -- the look-ahead limit;
2. window start ``s = t + C - W`` (the contract's ``wave[t + C - W : t + C]``); ``s < 0``
   means the window is left-padded with zeros (silence before the stream started); the
   trailing chunk (``t + C > duration``) therefore gets a window shorter than ``W``, which
   the base model zero-pads (with its key-padding mask) exactly like the non-streaming
   runtime pads its last window;
3. run ``base.embed_window`` on the window -> one embedding per frame whose receptive
   field fits in the window;
4. keep the frames that belong to chunk ``k`` (``emit_rule``), append them in time order:

   * ``"rf_end"`` (default): the frames whose receptive field END lies in ``(t, t + C]``
     (and whose receptive field starts inside the stream, not in the left zero padding),
     i.e. a frame is emitted by the first chunk whose end covers its whole receptive
     field.  With ``W >= C + rf`` every frame of the token grid ``[i*hop, i*hop + rf)``,
     ``i >= 0``, is emitted exactly once (no frame loss); a frame becomes available at most
     ``C`` after its receptive field ends, i.e. at most ``C + rf - centre_offset`` after its
     timestamp.
   * ``"centre"`` (the literal DESIGN2 rule): the frames whose CENTRE lies in
     ``[t, t + C)``.  The frame whose receptive field straddles the chunk end cannot be
     computed from the window and is lost: one frame per chunk for the conv front ends
     (10 % of the frames at C = 0.1 s for WavJEPA-100, 20 % for WavJEPA-50) and every
     frame whose centre lies in the last ``rf - centre_offset`` = 95 ms of a chunk for
     BEATs (36 % at C = 0.25 s).  Look-ahead: at most ``C`` after the timestamp.

   Both rules bound the look-ahead by ``C + rf`` (= ``latency_ms``) and both reproduce the
   non-streaming runtime at ``W = C = process_seconds`` (then no window overlaps and the
   two rules select the same frames).

Chunks partition the time axis (by receptive-field end or by centre), so every frame is
emitted exactly once and timestamps are strictly increasing (spacing = the token hop inside
a chunk).  When ``C`` is a whole number of hops all windows share one global grid; a grid
frame is then missing only where its receptive field straddles a window boundary no window
covers (e.g. with ``W = C`` the frames whose receptive field crosses ``k*C``), exactly like
the non-streaming runtime, which loses one frame per window boundary.

``window_s = hop_s = inf`` means "one window = the whole clip" (the non-streaming
reference for models without a fixed window, e.g. BEATs).

Base models
-----------
* :class:`WavJEPAWindowModel` wraps a :class:`hear_api.runtime.RuntimeJEPA`.  A window
  shorter than ``process_seconds`` is standardised over its real samples only, zero-padded
  to ``process_seconds`` and run through ``JEPA.get_audio_representation`` with the
  runtime's key-padding mask (``pad_mode="zero"``, the contract's default), or -- opt-in
  ``pad_mode="trim"`` -- run through the identical extractor -> feature_norms ->
  post_extraction_mapper -> +pos -> encoder pipeline on the unpadded window (fewer tokens,
  no padding influence on the conv GroupNorm statistics).  Either way only the real frames
  are returned.  With ``W = C = process_seconds`` both modes are bit-identical to
  ``RuntimeJEPA.get_timestamp_embeddings`` on full windows.
* :class:`BEATsWindowModel` wraps a BEATs model (``extract_features``).  BEATs emits
  ``8 * (fbank_frames // 16)`` tokens (2 s -> 96, 10 s -> 496): one token per
  16 x 16 patch of the 128-mel / 10 ms fbank, flattened time-major (token ``k`` = mel band
  ``k % 8`` of time patch ``k // 8``).  A *frame* here is one time patch: the 8 band tokens
  are pooled (mean, or concatenated with ``band_pool="concat"``), the hop is 160 ms, the
  receptive field 175 ms (25 ms window + 15 x 10 ms) and the timestamp of time patch ``j``
  is the centre of its 16-frame hop grid, ``(16 j + 8) x 10 ms = 160 j + 80 ms`` (the EAT
  convention) -- NOT the reference BEATs HEAR wrapper's ``12.5 + 20 k`` ms / ``(ms - 5) // 20``
  pseudo-frames, whose count is off by 3 per 2 s.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn

try:
    from hear_api.feature_helper import FeatureExtractor
    from hear_api.runtime import RuntimeJEPA, normalize
except ImportError:  # pragma: no cover - e.g. the mwmae-eval env (no `transformers`, which wavjepa imports);
    # only the WavJEPA window model needs these, the wrapper and the BEATs/EAT/MW-MAE models do not.
    FeatureExtractor = RuntimeJEPA = normalize = None  # type: ignore[assignment]


# --------------------------------------------------------------------------- helpers
def _ceil_div(a: int, b: int) -> int:
    """Ceiling division for (possibly negative) integers."""
    return -((-a) // b)


def _samples(seconds: float, sr: int, snap_to: Optional[int] = None) -> int:
    """Seconds -> samples with the truncating ``int(x * sr)`` convention of ``RuntimeJEPA`` /
    ``JEPA`` (``2.01 * 16000`` is ``32159.999...`` in floating point -> 32159 samples).

    ``snap_to``: a reference length in samples (e.g. the runtime's ``unit_frames``); a result
    within one sample of it is replaced by it, so that ``unit_frames / sr`` seconds map back
    to exactly ``unit_frames`` samples (``int(1.005 * 16000)`` is 16079, not 16080).
    """
    n = int(seconds * sr)
    if snap_to is not None and abs(n - snap_to) <= 1:
        return int(snap_to)
    return n


def _ms_to_samples_exact(ms: float, sr: int, what: str) -> int:
    """Milliseconds -> samples; the value must be an integer number of samples."""
    exact = ms * sr / 1000.0
    samples = int(round(exact))
    if abs(exact - samples) > 1e-6:
        raise ValueError(f"{what} = {ms} ms is not an integer number of samples at {sr} Hz")
    return samples


def _is_cuda(device: torch.device) -> bool:
    return device.type == "cuda"


def _atomic_json_dump(path: str, payload: dict) -> None:
    """Write ``payload`` to ``path`` via a temporary file + rename."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def _cpu_name() -> str:
    try:
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "cpu"


# --------------------------------------------------------------------------- window models
class WindowModel(nn.Module):
    """Interface every base model of :class:`StreamingWrapper` implements.

    Attributes (set by the subclass):
        name: short model name (used in reports).
        sample_rate: input sample rate (Hz).
        embedding_dim: ``D`` of the returned embeddings.
        hop: token hop in samples.
        rf: front-end receptive field in samples (a frame's true temporal support).
        centre_offset: offset (samples) of a frame's timestamp from the start of its
            receptive field; ``rf / 2`` (the centre) unless the model defines otherwise.
        max_window: longest window (samples) ``embed_window`` accepts, or ``None``.

    Frames returned by ``embed_window`` for a window of ``Lw`` samples are the ``n_frames(Lw)``
    frames ``i = 0, 1, ...`` whose receptive field ``[i*hop, i*hop + rf)`` fits in the window.
    """

    name: str = "window-model"
    sample_rate: int = 16000
    embedding_dim: int = 0
    hop: int = 1
    rf: int = 1
    centre_offset: float = 0.5
    max_window: Optional[int] = None

    @property
    def token_hop_ms(self) -> float:
        return self.hop / self.sample_rate * 1000.0

    @property
    def front_rf_ms(self) -> float:
        return self.rf / self.sample_rate * 1000.0

    @property
    def centre_offset_ms(self) -> float:
        return self.centre_offset / self.sample_rate * 1000.0

    @property
    def max_window_s(self) -> Optional[float]:
        return None if self.max_window is None else self.max_window / self.sample_rate

    @property
    def device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:  # a parameter-free synthetic model
            return torch.device("cpu")

    def n_frames(self, n_samples: int) -> int:
        """Number of frames ``embed_window`` returns for a window of ``n_samples`` samples."""
        if n_samples < self.rf:
            return 0
        return (n_samples - self.rf) // self.hop + 1

    def prepare(self, audio: torch.Tensor) -> torch.Tensor:
        """Clip-level front end applied once per clip (e.g. gain); ``(B, L) -> (B, L)``."""
        return audio

    def embed_window(self, wave: torch.Tensor) -> torch.Tensor:
        """Full-context embeddings of one batch of equal-length windows ``(B, Lw) -> (B, Tw, D)``."""
        raise NotImplementedError


class WavJEPAWindowModel(WindowModel):
    """``RuntimeJEPA`` as a window model (see the module docstring for ``pad_mode``)."""

    PAD_MODES = ("zero", "trim")

    def __init__(self, runtime: RuntimeJEPA, pad_mode: str = "zero", name: str = "wavjepa") -> None:
        super().__init__()
        if pad_mode not in self.PAD_MODES:
            raise ValueError(f"pad_mode must be one of {self.PAD_MODES}, got {pad_mode!r}")
        extractor = runtime.model.extract_audio
        if not hasattr(extractor, "conv_layers_spec") or not hasattr(extractor, "receptive_fields"):
            raise TypeError("WavJEPAWindowModel needs a ConvFeatureExtractor-like extractor")
        if int(getattr(runtime.feature_extractor, "in_channels", 1)) != 1:
            raise ValueError("only mono (in_channels=1) runtimes are supported")
        self.runtime = runtime
        self.pad_mode = pad_mode
        self.name = name
        self.sample_rate = int(runtime.sample_rate)
        self.embedding_dim = int(runtime.embedding_size)
        # hop = product of the conv strides, rf from ConvFeatureExtractor.receptive_fields
        self.hop = int(math.prod(stride for _dim, _k, stride in extractor.conv_layers_spec))
        self.rf = int(extractor.receptive_fields[0])
        self.centre_offset = self.rf / 2.0
        # the runtime's window: int(process_seconds * sr) = JEPA.target_length (32159 for 2.01 s)
        self.max_window = int(runtime.unit_frames)
        self.output_steps = int(runtime.output_steps)
        if self.n_frames(self.max_window) != self.output_steps:
            raise RuntimeError(
                f"frame bookkeeping mismatch: closed form gives {self.n_frames(self.max_window)} "
                f"frames for {self.max_window} samples, the runtime has {self.output_steps}"
            )

    @property
    def jepa(self):
        return self.runtime.model

    @property
    def device(self) -> torch.device:
        return next(self.jepa.parameters()).device

    def prepare(self, audio: torch.Tensor) -> torch.Tensor:
        """The runtime's clip-level front end: -14 dBFS RMS gain + mono handling.

        Done once per clip exactly like ``RuntimeJEPA.to_feature`` (without its hard-coded
        ``.cuda()``).  The gain is a scalar per clip that the per-window standardisation in
        :meth:`embed_window` cancels, so it does not leak information across chunks.
        """
        feats = self.runtime.feature_extractor._wav2feature(audio)  # (B, 1, L)
        return feats[:, 0, :].to(self.device)

    def embed_window(self, wave: torch.Tensor) -> torch.Tensor:
        n_samples = int(wave.shape[-1])
        if n_samples > self.max_window:
            raise ValueError(f"window of {n_samples} samples exceeds process_seconds ({self.max_window})")
        n_real = self.n_frames(n_samples)
        batch = wave.shape[0]
        if n_real == 0:
            return wave.new_zeros((batch, 0, self.embedding_dim))
        # Per-window standardisation over the REAL samples only (the runtime's ``normalize``).
        x = normalize(wave.to(self.device).unsqueeze(1))  # (B, 1, Lw)
        if n_samples == self.max_window or self.pad_mode == "zero":
            pad = self.max_window - n_samples
            if pad > 0:
                x = torch.nn.functional.pad(x, (0, pad), mode="constant", value=0.0)
            # True = padded frame -> excluded from attention (key padding mask), as in the runtime.
            mask = torch.arange(self.output_steps, device=x.device) >= n_real
            mask = mask.unsqueeze(0).expand(batch, self.output_steps)
            out = self.jepa.get_audio_representation(x, mask)  # (B, S, D)
            return out[:, :n_real]
        # pad_mode == "trim": the very same pipeline as get_audio_representation, on the
        # unpadded window (n_real tokens, the first n_real positional embeddings).
        jepa = self.jepa
        with torch.inference_mode():
            jepa.eval()
            local = jepa.extract_audio(x)
            local = jepa.feature_norms(local)
            if jepa.post_extraction_mapper is not None:
                local = jepa.post_extraction_mapper(local)
            local = local + jepa.pos_encoding_encoder[:, : local.shape[1]]
            out = jepa.encoder_forward(local, src_key_padding_mask=None)
        if out.shape[1] != n_real:
            raise RuntimeError(f"extractor produced {out.shape[1]} frames, expected {n_real}")
        return out


class BEATsWindowModel(WindowModel):
    """A BEATs model (``third_party/BEATs``) as a window model.

    ``extract_features`` gives one token per 16 x 16 patch of the 128-mel Kaldi fbank
    (25 ms window / 10 ms hop), flattened time-major: token ``k`` = mel band ``k % 8`` of
    time patch ``k // 8``.  A frame of this window model is one *time patch*: its 8 band
    tokens are mean-pooled (``band_pool="mean"``, ``D`` = 768) or concatenated
    (``band_pool="concat"``, ``D`` = 8 x 768).  Time patch ``j`` covers fbank frames
    ``16 j .. 16 j + 15`` = samples ``[2560 j, 2560 j + 2800)`` (``hop`` 160 ms, ``rf`` 175 ms)
    and is stamped ``(16 j + 8) x 10 ms = 160 j + 80 ms``, the centre of its 16-frame hop grid
    (``centre_offset`` = 1280 samples).  ``n_frames(L) = (1 + (L - 400) // 160) // 16``,
    i.e. 12 frames (96 tokens) for 2 s and 62 frames (496 tokens) for 10 s.
    """

    FBANK_WINDOW = 400  # 25 ms Kaldi fbank window (samples at 16 kHz)
    FBANK_HOP = 160  # 10 ms
    N_MEL = 128
    BAND_POOLS = ("mean", "concat")

    def __init__(self, beats: nn.Module, band_pool: str = "mean", name: str = "beats") -> None:
        super().__init__()
        if band_pool not in self.BAND_POOLS:
            raise ValueError(f"band_pool must be one of {self.BAND_POOLS}, got {band_pool!r}")
        self.beats = beats.eval()
        self.band_pool = band_pool
        self.name = name
        self.sample_rate = 16000
        self.patch = int(beats.cfg.input_patch_size)
        if self.N_MEL % self.patch != 0:
            raise ValueError(f"input_patch_size {self.patch} does not divide {self.N_MEL} mel bins")
        self.bands = self.N_MEL // self.patch  # 8 band tokens per time patch
        self.token_dim = int(beats.cfg.encoder_embed_dim)
        self.embedding_dim = self.token_dim if band_pool == "mean" else self.token_dim * self.bands
        self.hop = self.FBANK_HOP * self.patch  # 2560 samples = 160 ms per time patch
        self.rf = self.FBANK_WINDOW + (self.patch - 1) * self.FBANK_HOP  # 2800 samples = 175 ms
        self.centre_offset = self.patch * self.FBANK_HOP / 2.0  # (16 j + 8) x 10 ms -> 1280 samples = 80 ms
        self.max_window = None

    def fbank_frames(self, n_samples: int) -> int:
        """Kaldi fbank frames for ``n_samples`` (``snip_edges=True``): ``1 + (L - 400) // 160``."""
        if n_samples < self.FBANK_WINDOW:
            return 0
        return 1 + (n_samples - self.FBANK_WINDOW) // self.FBANK_HOP

    def n_frames(self, n_samples: int) -> int:
        """Time patches (= frames) for a window of ``n_samples``: ``fbank_frames // 16``."""
        return self.fbank_frames(n_samples) // self.patch

    def n_tokens(self, n_samples: int) -> int:
        """Tokens ``extract_features`` returns: ``8 * n_frames`` (2 s -> 96, 10 s -> 496)."""
        return self.bands * self.n_frames(n_samples)

    def frame_timestamps_ms(self, n_samples: int) -> torch.Tensor:
        """Timestamps (ms, float64) of the frames of one window: ``160 j + 80``."""
        j = torch.arange(self.n_frames(n_samples), dtype=torch.float64)
        return (j * self.hop + self.centre_offset) / self.sample_rate * 1000.0

    def pool_tokens(self, tokens: torch.Tensor, n_frames: int) -> torch.Tensor:
        """``(B, 8 * n, Dt)`` time-major tokens -> ``(B, n, D)`` frames (band pooling)."""
        batch = tokens.shape[0]
        if tokens.shape[1] != self.bands * n_frames:
            raise RuntimeError(
                f"BEATs produced {tokens.shape[1]} tokens, bookkeeping expected {self.bands * n_frames}"
            )
        grid = tokens.reshape(batch, n_frames, self.bands, tokens.shape[-1])  # token k = 8 j + b
        if self.band_pool == "mean":
            return grid.mean(dim=2)
        return grid.reshape(batch, n_frames, self.bands * tokens.shape[-1])

    def embed_window(self, wave: torch.Tensor) -> torch.Tensor:
        n_real = self.n_frames(int(wave.shape[-1]))
        if n_real == 0:
            return wave.new_zeros((wave.shape[0], 0, self.embedding_dim))
        with torch.inference_mode():
            tokens = self.beats.extract_features(wave.to(self.device), padding_mask=None)[0]
        return self.pool_tokens(tokens, n_real)


class EATWindowModel(BEATsWindowModel):
    """EAT-base (``third_party/EAT``) as a window model; same patch geometry as BEATs.

    EAT consumes a 128-mel Kaldi fbank (25 ms / 10 ms, ``eat_loader.eat_fbank``: waveform mean
    removed per window, normalised with the model card's mean/std) through a 16 x 16 patch conv
    (stride 16) and a ViT with a CLS token: ``extract_features`` returns ``1 + 8 * (frames // 16)``
    tokens, CLS first, patches time-major (token ``1 + 8 j + b`` = band ``b`` of time patch ``j``).
    Frames, hop (160 ms), receptive field (175 ms), timestamps (``160 j + 80`` ms) and band
    pooling are exactly those of :class:`BEATsWindowModel`.  ``max_length`` = 768 columns
    (122.9 s), so a whole 120 s DCASE clip fits in one pass.
    """

    def __init__(self, eat: nn.Module, band_pool: str = "mean", name: str = "eat") -> None:
        WindowModel.__init__(self)
        if band_pool not in self.BAND_POOLS:
            raise ValueError(f"band_pool must be one of {self.BAND_POOLS}, got {band_pool!r}")
        self.eat = eat.eval()
        self.band_pool = band_pool
        self.name = name
        self.sample_rate = 16000
        self.patch = 16
        self.bands = self.N_MEL // self.patch
        self.token_dim = 768
        self.embedding_dim = self.token_dim if band_pool == "mean" else self.token_dim * self.bands
        self.hop = self.FBANK_HOP * self.patch
        self.rf = self.FBANK_WINDOW + (self.patch - 1) * self.FBANK_HOP
        self.centre_offset = self.patch * self.FBANK_HOP / 2.0
        self.max_window = None

    @property
    def device(self) -> torch.device:
        return next(self.eat.parameters()).device

    def embed_window(self, wave: torch.Tensor) -> torch.Tensor:
        from third_party.EAT.eat_loader import eat_fbank  # vendored loader (repo root on sys.path)

        n_real = self.n_frames(int(wave.shape[-1]))
        if n_real == 0:
            return wave.new_zeros((wave.shape[0], 0, self.embedding_dim))
        with torch.inference_mode():
            mel = eat_fbank(wave.to(self.device))  # (B, 1, T, 128)
            tokens = self.eat.extract_features(mel)[:, 1:, :]  # drop CLS -> (B, 8 * n_real, 768)
        return self.pool_tokens(tokens, n_real)


# --------------------------------------------------------------------------- the wrapper
@dataclass(frozen=True)
class Chunk:
    """One chunk of the streaming plan (all positions in samples)."""

    index: int
    t: int  # chunk start
    e: int  # window end = min(t + C, duration)  (look-ahead limit)
    s: int  # window start = t + C - W (negative = left zero padding)
    i_min: int  # first window frame that belongs to this chunk (emit_rule); the last is n_frames - 1
    n_frames: int  # frames the window yields

    @property
    def length(self) -> int:
        return self.e - self.s

    @property
    def n_emit(self) -> int:
        return max(0, self.n_frames - self.i_min)


class StreamingWrapper(nn.Module):
    """HEAR-API model that runs ``base`` in sliding windows with bounded look-ahead.

    Args:
        base: a :class:`WindowModel` (``embed_window``, ``n_frames``, ``prepare``).
        window_s: ``W`` in seconds (``inf`` = the whole clip).
        hop_s: ``C`` in seconds (``inf`` = one chunk per clip).
        sample_rate: Hz.
        token_hop_ms: token hop of ``base`` in ms (must match ``base.hop``).
        front_rf_ms: front-end receptive field of ``base`` in ms (must match ``base.rf``).
        max_window_s: longest window ``base`` accepts (``None`` = any); ``W`` must not exceed it.
            A ``W`` within one sample of ``base.max_window`` snaps to exactly that length.
        centre_offset_ms: timestamp offset of a frame from the start of its receptive field
            (default ``base.centre_offset``, i.e. ``front_rf_ms / 2`` for WavJEPA).
        batch_windows: windows per forward pass (memory bound).
        rtf_json: optional path; the accumulated real-time factor is written there after
            every call.
        left_pad: zero-pad windows that start before the clip (default, the contract);
            ``False`` trims them at the clip start instead.
        emit_rule: which frames a chunk emits, ``"rf_end"`` (default, no frame loss) or
            ``"centre"`` (the literal DESIGN2 rule) -- see the module docstring.
    """

    EMIT_RULES = ("rf_end", "centre")

    def __init__(
        self,
        base: WindowModel,
        window_s: float,
        hop_s: float,
        sample_rate: int,
        token_hop_ms: float,
        front_rf_ms: float,
        max_window_s: Optional[float] = None,
        centre_offset_ms: Optional[float] = None,
        batch_windows: int = 32,
        rtf_json: Optional[str] = None,
        left_pad: bool = True,
        emit_rule: str = "rf_end",
    ) -> None:
        super().__init__()
        if emit_rule not in self.EMIT_RULES:
            raise ValueError(f"emit_rule must be one of {self.EMIT_RULES}, got {emit_rule!r}")
        self.emit_rule = emit_rule
        self.base = base
        self.sample_rate = int(sample_rate)
        if int(base.sample_rate) != self.sample_rate:
            raise ValueError(f"base sample rate {base.sample_rate} != {sample_rate}")
        self.hop = _ms_to_samples_exact(token_hop_ms, self.sample_rate, "token_hop_ms")
        self.rf = _ms_to_samples_exact(front_rf_ms, self.sample_rate, "front_rf_ms")
        if self.hop != base.hop or self.rf != base.rf:
            raise ValueError(
                f"token_hop_ms/front_rf_ms ({self.hop}/{self.rf} samples) disagree with the base "
                f"model ({base.hop}/{base.rf} samples)"
            )
        offset_ms = base.centre_offset_ms if centre_offset_ms is None else float(centre_offset_ms)
        # kept as 2x the sample offset so that rf/2 stays an integer
        self.centre_offset2 = int(round(2.0 * offset_ms * self.sample_rate / 1000.0))
        if abs(self.centre_offset2 - 2.0 * offset_ms * self.sample_rate / 1000.0) > 1e-6:
            raise ValueError(f"centre_offset_ms = {offset_ms} is not a half-integer number of samples")
        self.window_s = float(window_s)
        self.hop_s = float(hop_s)
        if math.isinf(self.window_s) != math.isinf(self.hop_s):
            raise ValueError("window_s and hop_s must both be finite or both be inf")
        self.max_window_s = None if max_window_s is None else float(max_window_s)
        max_window = None
        if self.max_window_s is not None:
            # the base model's exact maximum (RuntimeJEPA.unit_frames) when it has one
            max_window = base.max_window if base.max_window is not None else _samples(self.max_window_s, self.sample_rate)
            if abs(max_window / self.sample_rate - self.max_window_s) > 1.0 / self.sample_rate:
                raise ValueError(f"max_window_s = {self.max_window_s} disagrees with the base model's {max_window} samples")
        if not math.isinf(self.window_s):
            if self.hop_s <= 0 or self.window_s <= 0:
                raise ValueError("window_s and hop_s must be positive")
            if self.window_s < self.hop_s - 1e-9:
                raise ValueError(f"window_s ({window_s}) must be >= hop_s ({hop_s}) (W >= C)")
            self.window = _samples(self.window_s, self.sample_rate, snap_to=max_window)
            if abs(self.hop_s - self.window_s) < 1e-9:
                self.chunk = self.window  # C = W: identical lengths, whatever the rounding
            else:
                self.chunk = _samples(self.hop_s, self.sample_rate)
            if self.chunk < 1 or self.window < self.rf:
                raise ValueError("hop_s too small / window_s shorter than the receptive field")
            if self.chunk > self.window:
                raise ValueError("hop_s rounds to more samples than window_s")
        else:
            self.window = None
            self.chunk = None
        if max_window is not None and (self.window is None or self.window > max_window):
            raise ValueError(f"window_s = {window_s} exceeds the model's maximum window {self.max_window_s} s")
        self.batch_windows = int(batch_windows)
        if self.batch_windows < 1:
            raise ValueError("batch_windows must be >= 1")
        self.rtf_json = rtf_json
        self.left_pad = bool(left_pad)
        # HEAR API attributes
        self.embedding_size = int(base.embedding_dim)
        self.scene_embedding_size = self.embedding_size
        self.timestamp_embedding_size = self.embedding_size
        # real-time factor bookkeeping
        self.reset_rtf()
        self._plans: Dict[int, List[Chunk]] = {}

    # ----------------------------------------------------------------- reporting helpers
    def latency_ms(self) -> float:
        """Algorithmic latency: chunk length + front-end receptive field (ms); ``inf`` = whole clip."""
        if math.isinf(self.hop_s):
            return math.inf
        return self.hop_s * 1000.0 + self.rf / self.sample_rate * 1000.0

    def reset_rtf(self) -> None:
        self.total_wall_s = 0.0
        self.total_audio_s = 0.0
        self.n_calls = 0
        self.last_rtf: Optional[float] = None

    @property
    def rtf(self) -> Optional[float]:
        """Accumulated real-time factor = wall seconds / audio seconds over all calls."""
        if self.total_audio_s <= 0:
            return None
        return self.total_wall_s / self.total_audio_s

    def measure_single_window(self, warmup: int = 10, runs: int = 50) -> Optional[dict]:
        """Batch-1 latency of ONE window forward = what a real-time stream pays per chunk.

        The batched RTF (``rtf``) replays the clip with ``batch_windows`` windows per forward,
        which is an offline throughput number.  A live stream advances one chunk at a time, so
        its real-time factor is ``single-window latency / C``: with C = 10 ms and a 4 ms window
        forward that is 0.4 even though the batched RTF is < 0.01.  Median / p95 over ``runs``
        forwards of a random window of the configured length after ``warmup`` forwards, with
        ``cuda.synchronize`` around each.  ``None`` for the whole-clip configuration.
        """
        if self.window is None:
            return None
        if getattr(self, "_single_window", None) is not None:
            return self._single_window
        gen = torch.Generator(device="cpu").manual_seed(0)
        wave = torch.randn(1, self.window, generator=gen).to(self.base.device)
        times: List[float] = []
        with torch.inference_mode():
            for _ in range(warmup):
                self.base.embed_window(wave)
            self._sync()
            for _ in range(runs):
                self._sync()
                t0 = time.perf_counter()
                self.base.embed_window(wave)
                self._sync()
                times.append(time.perf_counter() - t0)
        times.sort()
        med = times[len(times) // 2]
        p95 = times[min(len(times) - 1, int(round(0.95 * (len(times) - 1))))]
        chunk_s = self.chunk / self.sample_rate
        self._single_window = {
            "single_window_ms": med * 1e3,
            "single_window_p95_ms": p95 * 1e3,
            "windows_per_audio_second": 1.0 / chunk_s,
            "streaming_rtf": med / chunk_s,
            "warmup": warmup,
            "runs": runs,
        }
        return self._single_window

    def rtf_report(self) -> dict:
        device = self.base.device
        single = self.measure_single_window()
        report = {
            "model": self.base.name,
            "W": self.window_s,
            "C": self.hop_s,
            "window_samples": self.window,
            "chunk_samples": self.chunk,
            "latency_ms": self.latency_ms(),
            "token_hop_ms": self.hop / self.sample_rate * 1000.0,
            "front_rf_ms": self.rf / self.sample_rate * 1000.0,
            "centre_offset_ms": self.centre_offset2 / 2.0 / self.sample_rate * 1000.0,
            "emit_rule": self.emit_rule,
            "pad_mode": getattr(self.base, "pad_mode", None),
            "band_pool": getattr(self.base, "band_pool", None),
            "embedding_dim": self.embedding_size,
            "batch_windows": self.batch_windows,
            "device": device.type,
            "device_name": torch.cuda.get_device_name(device) if _is_cuda(device) else _cpu_name(),
            "threads": torch.get_num_threads(),
            "calls": self.n_calls,
            "audio_seconds": self.total_audio_s,
            "wall_seconds": self.total_wall_s,
            "rtf": self.rtf,
            "rtf_note": f"offline replay, {self.batch_windows} windows per forward",
            "last_call_rtf": self.last_rtf,
            "left_pad": self.left_pad,
        }
        if single is not None:
            report.update(single)
        # JSON has no inf
        for key in ("W", "C", "latency_ms"):
            if isinstance(report[key], float) and math.isinf(report[key]):
                report[key] = "inf"
        return report

    def describe(self) -> str:
        w = "inf" if math.isinf(self.window_s) else f"{self.window_s:g} s = {self.window} samples"
        c = "inf" if math.isinf(self.hop_s) else f"{self.hop_s:g} s = {self.chunk} samples"
        extra = ""
        if getattr(self.base, "pad_mode", None) is not None:
            extra += f", pad_mode={self.base.pad_mode}"
        if getattr(self.base, "band_pool", None) is not None:
            extra += f", band_pool={self.base.band_pool}"
        return (
            f"StreamingWrapper(base={self.base.name}, W={w}, C={c}, token hop "
            f"{self.hop / self.sample_rate * 1000:g} ms, front RF {self.rf / self.sample_rate * 1000:g} ms, "
            f"centre offset {self.centre_offset2 / 2 / self.sample_rate * 1000:g} ms, "
            f"latency {self.latency_ms():g} ms, D={self.embedding_size}, emit_rule={self.emit_rule}{extra}, "
            f"batch_windows={self.batch_windows}, left_pad={self.left_pad})"
        )

    # ----------------------------------------------------------------- planning
    def plan(self, n_samples: int) -> List[Chunk]:
        """The chunk/window schedule for a clip of ``n_samples`` samples (cached)."""
        cached = self._plans.get(n_samples)
        if cached is not None:
            return cached
        chunks: List[Chunk] = []
        if self.window is None:  # whole clip in one window
            chunks.append(Chunk(0, 0, n_samples, 0, 0, self.base.n_frames(n_samples)))
        else:
            k = 0
            while k * self.chunk < n_samples:
                t = k * self.chunk
                e = min(t + self.chunk, n_samples)  # look-ahead limit (clipped at the clip end)
                s = t + self.chunk - self.window  # the contract's wave[t + C - W : t + C]
                if not self.left_pad:
                    s = max(s, 0)
                n = self.base.n_frames(e - s)
                if self.emit_rule == "rf_end":
                    # first frame i whose receptive field end s + i*hop + rf is > t ...
                    i_min = max(0, _ceil_div(t - s - self.rf + 1, self.hop))
                    if s < 0:  # ... and whose receptive field starts inside the stream (not in the left padding)
                        i_min = max(i_min, _ceil_div(-s, self.hop))
                    if n > 0:  # every window frame ends inside the window, hence <= t + C
                        assert s + (n - 1) * self.hop + self.rf <= e, "a receptive field beyond the window"
                else:
                    # first frame i with 2*(s + i*hop) + centre_offset2 >= 2*t
                    i_min = max(0, _ceil_div(2 * t - 2 * s - self.centre_offset2, 2 * self.hop))
                    if n > 0:
                        last_centre2 = 2 * (s + (n - 1) * self.hop) + self.centre_offset2
                        assert last_centre2 < 2 * (t + self.chunk), "a frame centre beyond the chunk"
                chunks.append(Chunk(k, t, e, s, i_min, n))
                k += 1
        self._plans[n_samples] = chunks
        return chunks

    def frame_centres_ms(self, n_samples: int) -> torch.Tensor:
        """Timestamps (ms, float64, 1-D) of the frames emitted for a clip of ``n_samples``."""
        centres = []
        for ch in self.plan(n_samples):
            for i in range(ch.i_min, ch.n_frames):
                centres.append((2 * (ch.s + i * self.hop) + self.centre_offset2) / 2.0)
        ts = torch.tensor(centres, dtype=torch.float64)
        return ts / self.sample_rate * 1000.0

    def n_emitted(self, n_samples: int) -> int:
        """Number of frames emitted for a clip of ``n_samples`` samples."""
        return sum(ch.n_emit for ch in self.plan(n_samples))

    # ----------------------------------------------------------------- inference
    def _window_batch(self, wave: torch.Tensor, chunks: Sequence[Chunk]) -> torch.Tensor:
        """Gather the (equal-length) windows of ``chunks`` for every clip row -> (B*n, Lw)."""
        batch, n_samples = wave.shape
        length = chunks[0].length
        out = wave.new_zeros((len(chunks), batch, length))
        for j, ch in enumerate(chunks):
            lo, hi = max(ch.s, 0), ch.e
            out[j, :, lo - ch.s : hi - ch.s] = wave[:, lo:hi]
        return out.reshape(len(chunks) * batch, length)

    def _embed(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio[:, 0, :]
        if audio.ndim != 2:
            raise ValueError(f"audio must be (B, L) or (L,), got {tuple(audio.shape)}")
        batch, n_samples = audio.shape
        wave = self.base.prepare(audio.float())
        chunks = self.plan(n_samples)
        emitting = [ch for ch in chunks if ch.n_emit > 0]
        total = sum(ch.n_emit for ch in emitting)
        out = wave.new_empty((batch, total, self.embedding_size))
        offsets: Dict[int, int] = {}
        pos = 0
        for ch in emitting:
            offsets[ch.index] = pos
            pos += ch.n_emit
        # group by window length (equal-length windows are batched), keep time order inside
        by_length: Dict[int, List[Chunk]] = {}
        for ch in emitting:
            by_length.setdefault(ch.length, []).append(ch)
        per_forward = max(1, self.batch_windows // batch)
        for length, group in by_length.items():
            for start in range(0, len(group), per_forward):
                sub = group[start : start + per_forward]
                windows = self._window_batch(wave, sub)
                emb = self.base.embed_window(windows)  # (n*B, Tw, D)
                if emb.shape[1] != sub[0].n_frames:
                    raise RuntimeError(
                        f"base returned {emb.shape[1]} frames for a {length}-sample window, "
                        f"bookkeeping expected {sub[0].n_frames}"
                    )
                emb = emb.reshape(len(sub), batch, emb.shape[1], self.embedding_size)
                for j, ch in enumerate(sub):
                    o = offsets[ch.index]
                    out[:, o : o + ch.n_emit] = emb[j, :, ch.i_min : ch.n_frames]
        ts = self.frame_centres_ms(n_samples).to(torch.float32).unsqueeze(0).repeat(batch, 1)
        assert ts.shape[1] == out.shape[1], (ts.shape, out.shape)
        return out, ts

    def _sync(self) -> None:
        if _is_cuda(self.base.device):
            torch.cuda.synchronize(self.base.device)

    def get_timestamp_embeddings(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """HEAR timestamp API: ``(B, L) -> (embeddings (B, T, D) on the model device,
        timestamps (B, T) float32 ms on the CPU)``; also updates the RTF accounting."""
        self._sync()
        t0 = time.perf_counter()
        with torch.inference_mode():
            out, ts = self._embed(audio)
        self._sync()
        wall = time.perf_counter() - t0
        seconds = float(audio.shape[-1]) / self.sample_rate * (audio.shape[0] if audio.ndim > 1 else 1)
        self.total_wall_s += wall
        self.total_audio_s += seconds
        self.n_calls += 1
        self.last_rtf = wall / seconds if seconds > 0 else None
        if self.rtf_json:
            _atomic_json_dump(self.rtf_json, self.rtf_report())
        return out, ts

    def scene(self, audio: torch.Tensor) -> torch.Tensor:
        """Clip-level embedding = mean over the emitted frames ``(B, D)``."""
        out, _ = self.get_timestamp_embeddings(audio)
        return out.mean(dim=1)

    def get_scene_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        return self.scene(audio)

    def forward(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.get_timestamp_embeddings(audio)


# --------------------------------------------------------------------------- test / smoke utility
def runtime_from_jepa(model, sample_rate: int = 16000, in_channels: int = 1) -> RuntimeJEPA:
    """Build a ``RuntimeJEPA`` around an existing ``JEPA`` instance (no checkpoint).

    ``RuntimeJEPA.__init__`` always builds the production-size transformer and needs a
    checkpoint; tests use this to wrap a tiny random model with the *real* runtime methods
    (``get_timestamp_embeddings`` etc.).  The attributes are set exactly as the constructor
    sets them (``unit_frames`` = ``JEPA.target_length`` = ``int(process_seconds * sr)``).
    """
    runtime = RuntimeJEPA.__new__(RuntimeJEPA)
    nn.Module.__init__(runtime)
    runtime.sample_rate = int(sample_rate)
    runtime.model = model.eval()
    runtime.embedding_size = model.encoder_embedding_dim
    runtime.scene_embedding_size = runtime.embedding_size
    runtime.timestamp_embedding_size = runtime.embedding_size
    runtime.unit_frames = int(model.target_length)
    runtime.output_steps = model.extract_audio.total_patches(runtime.unit_frames)
    runtime.feature_extractor = FeatureExtractor(in_channels=in_channels)
    return runtime


def cpu_safe_runtime(runtime: RuntimeJEPA) -> RuntimeJEPA:
    """Make ``RuntimeJEPA.get_timestamp_embeddings`` usable without CUDA.

    ``hear_api.feature_helper.FeatureExtractor.forward`` ends with an unconditional
    ``.cuda()``; on a CPU-only machine this replaces it (for this instance only) by the
    same computation on the runtime's device.  No-op when CUDA is available.
    """
    if torch.cuda.is_available():
        return runtime
    fe = runtime.feature_extractor
    device = next(runtime.model.parameters()).device
    fe.forward = lambda x, _fe=fe, _dev=device: _fe._wav2feature(x).to(_dev)  # type: ignore[assignment]
    return runtime
