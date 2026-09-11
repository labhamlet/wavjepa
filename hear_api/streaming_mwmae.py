"""MW-MAE (Yadav et al., ICLR 2024; ``third_party/MWMAE``, JAX/Flax) as a streaming window model.

Kept out of ``hear_api/streaming.py`` so that module stays importable without JAX. Run inside the
``mwmae-eval`` conda environment (``third_party/MWMAE/mwmae_env.yml``: jax 0.4.30 + CUDA 12, flax,
torch 2.5.1, the HEAR eval-kit dependencies).

Geometry (``configs/pretraining/mwmae_base_200_4x16_precomputed``): torch log-mel front end
(n_fft = win = 400 samples = 25 ms, hop 160 = 10 ms, 80 mels, ``center=True``), per-window
mean/std normalisation (the original HEAR wrapper normalises per clip; per window is the
streaming analogue), 200-frame units (2 s) through the encoder with 4 x 16 patches, i.e. one
frame per 4 fbank frames = **40 ms**, 5 mel bands whose 768-d tokens are CONCATENATED
(``forward_features``: ``b (t f) d -> b t (f d)``) = 3840-d frames. A partial unit is
reflect-padded to 200 frames exactly like the original wrapper, and only the columns whose 4
frames are real are returned.

Receptive-field bookkeeping for :class:`hear_api.streaming.StreamingWrapper`: fbank frame k is
centred on sample 160 k (``center=True``), so column j spans samples
``[640 j - 200, 640 j + 680)``; the wrapper assumes ``[i*hop, i*hop + rf)``, so we declare
``hop = 640``, ``rf = 880`` (55 ms: 25 ms window + 3 hops) and the true centre
``centre_offset = 240`` samples (15 ms). The 200 samples before the column start come from
``center`` padding (reflection of the window's own samples), never from the future, so the
declared receptive field is conservative: a column is emitted at most one chunk later than
strictly necessary and the reported latency C + 55 ms is an upper bound.
"""
from __future__ import annotations

import math
import os
import sys
from functools import partial
from typing import Optional

import numpy as np
import torch
import torchaudio

from hear_api.streaming import WindowModel

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MWMAE_DIR = os.path.join(_REPO, "third_party", "MWMAE")
DEFAULT_WEIGHTS_DIR = "/projects/0/prjs1338/models/MWMAE/mwmae_base_200_4x16_8x128_default_bfloat16_run1"
DEFAULT_CONFIG = "configs.pretraining.mwmae_base_200_4x16_precomputed"


def _jax_forward(batch, state, model):
    variables = {"params": state.get_all_params, "batch_stats": state.batch_stats, "buffers": state.buffers}
    return model.apply(variables, batch, train=False, mutable=False, method=model.forward_features)


class MWMAEWindowModel(WindowModel):
    """MW-MAE encoder as a window model (frames = 40 ms columns; 3840-d concatenated or 768-d averaged bands)."""

    FRAME_HOP = 160
    WIN = 400
    N_MELS = 80
    PATCH_T = 4
    PATCH_F = 16

    def __init__(self, weights_dir: str = DEFAULT_WEIGHTS_DIR, config_module: str = DEFAULT_CONFIG,
                 name: str = "mwmae", band_pool: str = "concat", framing: str = "unit",
                 unit_pad: str = "reflect", future_pad: int = 0, norm: str = "clip") -> None:
        """``band_pool``: ``concat`` = the official HEAR wrapper's 5 x 768 = 3840-d frames; ``mean`` = the
        5 band tokens of a column averaged, 768-d (the same width as WavJEPA / BEATs frames).

        ``framing``: how the 40 ms columns of a window are computed and which are emitted.
          * ``unit`` (default, the paper): the authors' pipeline (``hear_api/runtime.py`` of
            mwmae-jax-official) — torchaudio ``center=True`` log-mel (frame k centred on sample 160 k),
            200-frame units, a partial unit reflect-padded — with one rule for emission: a column is
            emitted only once its full 880-sample receptive field (25 ms window + 3 hops = 55 ms, samples
            ``[640 j - 200, 640 j + 680)``) has really arrived, i.e. never from reflected samples. The
            wrapper's bookkeeping declares it as ``[640 j, 640 j + 880)`` (``rf`` = 55 ms, conservative by
            the 200 past samples). The last column of a 200-frame unit, which torchaudio computes from the
            reflected window end, is therefore NOT emitted from that window (49 of 50); only the whole-clip
            configuration loses its final column. A new column exists every 40 ms, so the chunk must be
            >= 40 ms; latency C + 55 ms.
          * ``all``: every column the authors' wrapper computes, including the half-reflected last one
            (``rf`` declared 480 samples = 30 ms: the span of the 4 frame centres). Exactly the official
            whole-clip output, but in streaming it emits 12.5 ms of synthetic future; kept for the record.
          * ``end``: ``center=False`` frames and a window of ``880 + 640 k`` samples so that the last real
            column ends at the window end (any chunk size); kept for the record, not used in the paper.
        ``unit_pad``: ``reflect`` (official) or ``zero``: how a partial unit is completed (and how the
        future is padded for ``future_pad``).
        ``future_pad`` (samples, ``end`` framing only): synthetic future appended before the log-mel; kept
        for the record.
        ``norm``: ``clip`` (default, official: log-mel mean / std of the WHOLE clip, taken once per clip in
        :meth:`prepare` and applied to every window) or ``window`` (statistics of each window; the causal
        alternative)."""
        super().__init__()
        if band_pool not in ("concat", "mean"):
            raise ValueError(f"band_pool must be 'concat' or 'mean', got {band_pool!r}")
        if framing not in ("unit", "all", "end"):
            raise ValueError(f"framing must be 'unit', 'all' or 'end', got {framing!r}")
        if norm not in ("clip", "window"):
            raise ValueError(f"norm must be 'clip' or 'window', got {norm!r}")
        self.norm = norm
        self._clip_stats = None  # (mean, std) of the current clip batch, set by prepare() when norm == "clip"
        if unit_pad not in ("reflect", "zero"):
            raise ValueError(f"unit_pad must be 'reflect' or 'zero', got {unit_pad!r}")
        if future_pad and framing != "end":
            raise ValueError("future_pad needs framing='end'")
        if future_pad < 0 or future_pad > 640 or future_pad % 160:
            raise ValueError(f"future_pad must be a multiple of 160 in [0, 640] samples, got {future_pad}")
        self.band_pool, self.framing, self.unit_pad, self.future_pad = band_pool, framing, unit_pad, int(future_pad)
        if MWMAE_DIR not in sys.path:
            sys.path.insert(0, MWMAE_DIR)
        import jax  # noqa: WPS433 (only available in the mwmae-eval env)
        from importlib import import_module

        from src.trainer import MAETrainer  # vendored MW-MAE code

        config = import_module(config_module).get_config()
        self.trainer = MAETrainer(config, weights_dir, True, seed=0, inference=True)
        self.model = self.trainer.model
        self.forward_jit = jax.jit(partial(_jax_forward, state=self.trainer.state, model=self.model))
        img_size = tuple(self.model.img_size)
        patch = tuple(self.model.patch_size)
        if patch != (self.PATCH_T, self.PATCH_F) or img_size[1] != self.N_MELS:
            raise ValueError(f"unexpected MW-MAE geometry: img_size={img_size} patch={patch}")
        self.unit_frames = int(img_size[0])  # 200
        self.bands = self.N_MELS // self.PATCH_F  # 5
        self.token_dim = int(self.model.embed_dim)  # 768
        self.name = name
        self.sample_rate = 16000
        self.embedding_dim = self.bands * self.token_dim if band_pool == "concat" else self.token_dim  # 3840 / 768
        self.hop = self.FRAME_HOP * self.PATCH_T  # 640 samples = 40 ms
        if framing == "all":
            self.rf = (self.PATCH_T - 1) * self.FRAME_HOP  # 480 samples = 30 ms: span of the 4 frame centres (trailing half window reflected)
        else:
            self.rf = self.WIN + (self.PATCH_T - 1) * self.FRAME_HOP - self.future_pad  # 880 samples = 55 ms (minus the padded future)
        # true column centre: unit framing -> 240 samples (15 ms: frames centred at 640 j + 160 m);
        # end framing -> 440 samples (27.5 ms = rf / 2: frames [640 j + 160 m, +400))
        self.centre_offset = (self.PATCH_T - 1) * self.FRAME_HOP / 2.0 if framing in ("unit", "all") else self.rf / 2.0
        self.max_window = None  # any length; processed in 200-frame units
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=self.WIN, win_length=self.WIN, hop_length=self.FRAME_HOP,
            f_min=50.0, f_max=8000.0, n_mels=self.N_MELS, power=2.0, center=(framing in ("unit", "all")),
        )
        self.weights_dir = weights_dir

    @property
    def device(self) -> torch.device:
        """JAX owns the accelerator: report it (the wrapper synchronises/timestamps on it and the RTF
        report names it); the torch log-mel front end follows ``.to()`` and runs wherever its buffers are."""
        import jax

        if jax.default_backend() == "gpu" and torch.cuda.is_available():
            return torch.device("cuda", torch.cuda.current_device())
        return torch.device("cpu")

    def fbank_frames(self, n_samples: int) -> int:
        """torchaudio MelSpectrogram frames: ``1 + L // 160`` (``center=True``, unit framing) or
        ``1 + (L - 400) // 160`` (``center=False``, end framing)."""
        if self.framing in ("unit", "all"):
            return 1 + n_samples // self.FRAME_HOP
        return 1 + max(n_samples - self.WIN, 0) // self.FRAME_HOP if n_samples >= self.WIN else 0

    def n_frames(self, n_samples: int) -> int:
        """Columns emitted for a window of ``n_samples``: ``all`` framing = every column the authors'
        wrapper computes (``fbank_frames // 4``); unit / end framing = the WindowModel default
        ``(L - rf) // hop + 1`` (columns whose declared support lies inside the window)."""
        if self.framing == "all":
            return self.fbank_frames(n_samples) // self.PATCH_T
        return super().n_frames(n_samples)

    def prepare(self, audio: torch.Tensor) -> torch.Tensor:
        """Once per clip: with ``norm == "clip"`` take the log-mel mean / std of the whole clip (the official
        wrapper's normalisation) for use on every window of that clip."""
        self._clip_stats = None
        if self.norm == "clip":
            x = self._raw_logmel(audio)  # (B, 80, T)
            self._clip_stats = (x.mean(dim=(1, 2), keepdim=True), x.std(dim=(1, 2), keepdim=True), int(audio.shape[0]))
        return audio

    def _raw_logmel(self, wave: torch.Tensor) -> torch.Tensor:
        mel_device = next(self.melspec.buffers()).device
        with torch.no_grad():
            x = self.melspec(wave.float().to(mel_device))  # (B, 80, T)
            return (x + torch.finfo(torch.float32).eps).log()

    # n_frames: the WindowModel default, (L - rf) // hop + 1 = columns whose declared 880-sample
    # support lies inside the window. The unit's last column (which torchaudio computes from
    # reflected padding) is deliberately NOT emitted from that window: the next window, which
    # holds the real audio, emits it. Only the whole-clip configuration loses its final column.

    def _logmel(self, wave: torch.Tensor) -> np.ndarray:
        """(B, L) -> normalised log-mel ``(B, T, 80, 1)`` float32 numpy: clip statistics (official) when
        :meth:`prepare` has seen the clip batch the windows belong to, else the window's own statistics."""
        with torch.no_grad():
            x = self._raw_logmel(wave)  # (B, 80, T)
            stats = self._clip_stats
            if stats is not None and wave.shape[0] % stats[2] == 0:  # windows are ordered [window j, clip b]
                reps = wave.shape[0] // stats[2]
                mean, std = stats[0].to(x.device).repeat(reps, 1, 1), stats[1].to(x.device).repeat(reps, 1, 1)
            else:
                mean, std = x.mean(dim=(1, 2), keepdim=True), x.std(dim=(1, 2), keepdim=True)
            x = (x - mean) / (std + 1e-8)
            x = x.permute(0, 2, 1).contiguous()  # (B, T, 80)
        return x.cpu().numpy()[..., None].astype(np.float32)

    def embed_window(self, wave: torch.Tensor) -> torch.Tensor:
        import jax.numpy as jnp

        n_samples = int(wave.shape[-1])
        n_real = self.n_frames(n_samples)
        batch = wave.shape[0]
        if n_real == 0:
            return wave.new_zeros((batch, 0, self.embedding_dim))
        if self.future_pad > 0:  # extend the window with synthetic future (waveform domain) -> the last column
            w = wave.float().cpu().numpy()  # needs only 880 - future_pad real samples
            if self.unit_pad == "reflect" and w.shape[-1] >= 2:
                w = np.pad(w, ((0, 0), (0, self.future_pad)), mode="reflect")
            else:
                w = np.pad(w, ((0, 0), (0, self.future_pad)), mode="constant")
            wave = torch.from_numpy(np.ascontiguousarray(w))
        x = jnp.asarray(self._logmel(wave))  # (B, T, 80, 1)
        if self.framing == "end":  # columns anchored at the window start: drop the trailing partial column
            x = x[:, : n_real * self.PATCH_T]
        cur = x.shape[1]
        pad = (-cur) % self.unit_frames
        if pad > 0:  # complete the unit: reflect (as the original HEAR wrapper) or zeros (normalised space)
            widths = [(0, 0), (0, pad), (0, 0), (0, 0)]
            x = jnp.pad(x, widths, mode="reflect") if self.unit_pad == "reflect" else jnp.pad(x, widths, mode="constant")
        outs = []
        for i in range(x.shape[1] // self.unit_frames):
            outs.append(self.forward_jit(x[:, i * self.unit_frames:(i + 1) * self.unit_frames]))
        y = jnp.concatenate(outs, axis=1).astype(jnp.float32)  # (B, 50 * units, 3840) = b t (f d)
        if self.band_pool == "mean":
            y = y.reshape(y.shape[0], y.shape[1], self.bands, self.token_dim).mean(axis=2)  # (B, T, 768)
        y = np.asarray(y.block_until_ready())
        if y.shape[1] < n_real:
            raise RuntimeError(f"MW-MAE produced {y.shape[1]} columns, bookkeeping expected >= {n_real}")
        return torch.from_numpy(np.ascontiguousarray(y[:, :n_real])).to(wave.device)
