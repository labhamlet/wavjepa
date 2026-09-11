"""AudioMAE, SSAST and ATST-Clip as streaming window models (frames = time patches, band tokens averaged).

Code and pipelines follow the HEAR wrappers of github.com/labhamlet/awsome-audio-foundation-models (vendored under
``third_party/{AudioMAE,SSAST,ATST}``), which in turn follow the models' own feature code, with two exceptions that
are stated where they apply: (1) the AST-family log-mel normalisation is the original authors' ``(x - mean) / (2 std)``
(the awsome wrappers divide by ``std ** 2``); (2) the models' own token grid is used for timestamps instead of the
wrappers' one vector per unit.

Geometry (the ``StreamingWrapper`` contract: frame ``i`` of a window covers ``[s + i*hop, s + i*hop + rf)``):

* **AudioMAE** (ViT-B, ``pretrained.pth``): Kaldi fbank 128 mel, 25 ms window / 10 ms hop, audio mean-subtracted,
  the official recipes' dataset statistics and input lengths (AudioSet 1024 frames for general audio / DCASE, ESC-50
  512 frames), zero-padded before normalisation as the AST dataloader does, 16x16 patches stride 16
  -> 64 (or 32) time patches x 8 band patches per unit; time patch ``j`` = fbank frames ``16 j .. 16 j + 15`` = samples
  ``[2560 j, 2560 j + 2800)``: hop 160 ms, rf 175 ms, centre 80 ms (exactly BEATs' geometry). Tokens = the encoder
  blocks' outputs followed by the wrapper's ``fc_norm`` (a fresh LayerNorm, as in the repo wrapper, applied per token
  instead of after pooling); the 8 band tokens of a time patch are averaged (768-d) or concatenated.
* **SSAST** (Base-Patch-400): same fbank, the official fine-tuning construction (16x16 patches at stride 10, i.e.
  overlapping, 1024 frames with AudioSet statistics for general audio, 512 frames with ESC-50 statistics for ESC-50;
  ``run_esc_patch.sh``) -> 12 band x 101 (or 50) time patches, one frame per 100 ms, tokens after the final
  LayerNorm as ``ft_avgtok`` does, 2 class tokens dropped; the patch grid is stored band-major
  (the model transposes to ``(B, 1, F, T)``) and is un-transposed here.
* **ATST-Clip** (base): torchaudio MelSpectrogram 64 mel, n_fft = win = 1024 (64 ms), hop 160, 60-7800 Hz,
  AmplitudeToDB(power, top_db 80) per window, MinMax(-79.6482, 50.6842) -> [-1, 1]; patches 64 mel x 4 frames ->
  one 768-d token per 40 ms straight from the model (no band axis), CLS dropped; units of 6 s (the pre-training
  segment), shorter windows handled by the model's own length masking (no padding). A token's four frames are centred
  at ``640 j + {0, 160, 320, 480}`` with a 1024-sample window each, support ``[640 j - 512, 640 j + 992)`` = 1504
  samples = 94 ms; declared as ``[640 j, 640 j + 1504)`` (rf 94 ms, conservative by the 512 past samples, so a token
  is emitted only once its whole support is real audio), centre offset 240 samples.
"""
from __future__ import annotations

import os
import sys
from typing import List

import torch
import torch.nn as nn
import torchaudio

from hear_api.streaming import WindowModel

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIOMAE_DIR = os.path.join(_REPO, "third_party", "AudioMAE")
SSAST_DIR = os.path.join(_REPO, "third_party", "SSAST")
ATST_DIR = os.path.join(_REPO, "third_party", "ATST", "audiossl")
DEFAULT_AUDIOMAE = "/projects/0/prjs1338/models/AudioMAE/pretrained.pth"
DEFAULT_SSAST = "/projects/0/prjs1338/models/SSAST/SSAST-Base-Patch-400.pth"
DEFAULT_ATST = "/projects/0/prjs1338/models/ATST/base.ckpt"
# fbank statistics and input lengths of the official AST / SSAST / AudioMAE recipes (main_finetune_esc.py, run_esc_patch.sh)
AST_STATS = {"audioset": (-4.2677393, 4.5689974), "esc50": (-6.6268077, 5.358466), "speechcommands": (-6.845978, 5.5654526)}
AST_TARGET_LENGTH = {"audioset": 1024, "esc50": 512, "speechcommands": 128}
AST_NORM_MEAN, AST_NORM_STD = AST_STATS["audioset"]


def _kaldi_fbank(wave_1d: torch.Tensor, n_mels: int = 128) -> torch.Tensor:
    """The AST-family fbank of one waveform ``(L,)`` -> ``(T, n_mels)``: mean-subtracted audio, 25 ms hanning
    window, 10 ms shift, htk-compatible, no dither (``awsome`` ``_wav2fbank``)."""
    audio = (wave_1d - wave_1d.mean()).unsqueeze(0)
    return torchaudio.compliance.kaldi.fbank(
        audio, htk_compat=True, sample_frequency=16000, use_energy=False, window_type="hanning",
        num_mel_bins=n_mels, dither=0.0, frame_shift=10,
    )


class FbankPatchWindowModel(WindowModel):
    """Shared machinery of the AST family (AudioMAE, SSAST): fbank -> fixed-length units -> 16x16 patch tokens."""

    FBANK_HOP = 160
    FBANK_WINDOW = 400
    N_MEL = 128
    PATCH = 16

    def __init__(self, unit_frames: int, band_pool: str = "mean", name: str = "ast", stats: str = "audioset",
                 norm: str = "authors", tstride: int = 16, f_patches: int = 8) -> None:
        """``stats``: which official statistics normalise the fbank (``audioset`` for general audio / DCASE, ``esc50``
        for ESC-50, as the official fine-tuning recipes do); ``unit_frames``: the model input length (1024 for the
        AudioSet recipes, 512 for ESC-50); ``tstride``: time stride of the 16-frame patches (16 = non-overlapping,
        AudioMAE and SSAST pre-training; 10 = SSAST's fine-tuning recipes); ``f_patches``: band patches per time step."""
        super().__init__()
        if band_pool not in ("mean", "concat"):
            raise ValueError(f"band_pool must be 'mean' or 'concat', got {band_pool!r}")
        if norm not in ("authors", "awsome"):
            raise ValueError("norm must be 'authors' ((x-mean)/(2 std)) or 'awsome' ((x-mean)/std**2)")
        if stats not in AST_STATS:
            raise ValueError(f"stats must be one of {sorted(AST_STATS)}, got {stats!r}")
        self.unit_frames = int(unit_frames)
        self.band_pool = band_pool
        self.name = name
        self.stats = stats
        self.norm_mean, self.norm_std = AST_STATS[stats]
        self.norm = norm
        self.tstride = int(tstride)
        self.sample_rate = 16000
        self.bands = int(f_patches)
        self.token_dim = 768
        self.embedding_dim = self.token_dim if band_pool == "mean" else self.bands * self.token_dim
        self.hop = self.FBANK_HOP * self.tstride  # 2560 samples = 160 ms (stride 16) or 1600 = 100 ms (stride 10)
        self.rf = self.FBANK_WINDOW + (self.PATCH - 1) * self.FBANK_HOP  # 2800 samples = 175 ms
        self.centre_offset = self.PATCH * self.FBANK_HOP / 2.0  # 1280 samples = 80 ms
        self.max_window = None
        self._device_probe = nn.Parameter(torch.zeros(1), requires_grad=False)  # follows .to()

    @property
    def device(self) -> torch.device:
        return self._device_probe.device

    def fbank_frames(self, n_samples: int) -> int:
        if n_samples < self.FBANK_WINDOW:
            return 0
        return 1 + (n_samples - self.FBANK_WINDOW) // self.FBANK_HOP

    def n_frames(self, n_samples: int) -> int:
        """Time patches whose 16 fbank frames are all real: ``(fbank_frames - 16) // tstride + 1`` (BEATs' rule
        for stride 16)."""
        frames = self.fbank_frames(n_samples)
        return 0 if frames < self.PATCH else (frames - self.PATCH) // self.tstride + 1

    @property
    def t_patches(self) -> int:
        """Time patches the model produces per unit (``(unit_frames - 16) // tstride + 1``)."""
        return (self.unit_frames - self.PATCH) // self.tstride + 1

    def _units(self, wave: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` -> normalised fbank units ``(B, U, unit_frames, 128)``: zero-padded to whole units BEFORE
        normalisation (the AST dataloader's ``ZeroPad2d``), then ``(x - mean) / (2 std)``."""
        feats: List[torch.Tensor] = []
        for row in wave:
            fb = _kaldi_fbank(row.float().to(self.device), self.N_MEL)  # (T, 128)
            pad = (-fb.shape[0]) % self.unit_frames
            if fb.shape[0] == 0:
                pad = self.unit_frames
            if pad:
                fb = torch.nn.functional.pad(fb, (0, 0, 0, pad))
            feats.append(fb)
        x = torch.stack(feats)  # (B, U*unit_frames, 128)
        denom = self.norm_std * 2 if self.norm == "authors" else self.norm_std ** 2
        x = (x - self.norm_mean) / denom
        return x.reshape(x.shape[0], -1, self.unit_frames, self.N_MEL)

    def unit_tokens(self, units: torch.Tensor) -> torch.Tensor:
        """``(N, 1, unit_frames, 128)`` -> ``(N, t_patches, f_patches, 768)`` (time patch, band patch, token)."""
        raise NotImplementedError

    def embed_window(self, wave: torch.Tensor) -> torch.Tensor:
        n_real = self.n_frames(int(wave.shape[-1]))
        batch = wave.shape[0]
        if n_real == 0:
            return wave.new_zeros((batch, 0, self.embedding_dim))
        with torch.inference_mode():
            x = self._units(wave)  # (B, U, unit_frames, 128)
            n_units = x.shape[1]
            tok = self.unit_tokens(x.reshape(batch * n_units, 1, self.unit_frames, self.N_MEL))
            tok = tok.reshape(batch, n_units * self.t_patches, self.bands, self.token_dim)
            if tok.shape[1] < n_real:
                raise RuntimeError(f"{self.name}: {tok.shape[1]} time patches computed, bookkeeping expected >= {n_real}")
            tok = tok[:, :n_real]
            out = tok.mean(dim=2) if self.band_pool == "mean" else tok.reshape(batch, n_real, -1)
        return out.to(wave.device)


class _AudioMAEPatchEmbed(nn.Module):
    """The awsome wrapper's ``PatchEmbed_new`` (third_party/AudioMAE/hear_api/runtime.py): a non-overlapping 16x16
    conv patch embedding on ``(N, 1, T, F)`` whose tokens are time-major (``patch_hw = (T/16, F/16)``). Copied here
    because that module's package name (``hear_api``) clashes with this repository's."""

    def __init__(self, img_size, patch_size, in_chans, embed_dim, stride) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        with torch.no_grad():
            h, w = self.proj(torch.zeros(1, in_chans, img_size[0], img_size[1])).shape[2:]
        self.patch_hw = (int(h), int(w))
        self.num_patches = int(h) * int(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)  # (N, T/16 * F/16, D), time-major


class AudioMAEWindowModel(FbankPatchWindowModel):
    """AudioMAE ViT-B encoder (``third_party/AudioMAE``): 1024-frame units, 64 x 8 patches, per-token ``fc_norm``."""

    def __init__(self, weights: str = DEFAULT_AUDIOMAE, band_pool: str = "mean", norm: str = "authors",
                 stats: str = "audioset", unit_frames: int = 1024) -> None:
        """``unit_frames`` 1024 (AudioSet recipe) or 512 (the official ESC-50 recipe: ``img_size = (512, 128)``); the
        fixed sin-cos positional table is the pre-training one (64 x 8, time-major) restricted to the first
        ``unit_frames // 16`` time rows, which is exactly the sin-cos table of the smaller grid."""
        super().__init__(unit_frames, band_pool=band_pool, name="audiomae", norm=norm, stats=stats, tstride=16, f_patches=8)
        if AUDIOMAE_DIR not in sys.path:
            sys.path.insert(0, AUDIOMAE_DIR)
        import models_vit  # vendored

        m = models_vit.__dict__["vit_base_patch16"](num_classes=527, drop_path_rate=0.1, global_pool=True,
                                                    mask_2d=False, use_custom_patch=False)
        m.patch_embed = _AudioMAEPatchEmbed(img_size=(self.unit_frames, self.N_MEL), patch_size=(16, 16), in_chans=1,
                                            embed_dim=768, stride=16)
        m.pos_embed = nn.Parameter(torch.zeros(1, m.patch_embed.num_patches + 1, 768), requires_grad=False)
        ck = dict(torch.load(weights, map_location="cpu", weights_only=False)["model"])
        pe = ck["pos_embed"]  # (1, 1 + 64*8, 768), time-major
        if pe.shape[1] != m.pos_embed.shape[1]:
            ck["pos_embed"] = torch.cat([pe[:, :1], pe[:, 1:1 + m.patch_embed.num_patches]], dim=1)
        msg = m.load_state_dict(ck, strict=False)
        missing = [k for k in msg.missing_keys if not k.startswith(("fc_norm", "head"))]
        if missing:
            raise RuntimeError(f"AudioMAE: unexpected missing keys {missing[:5]}")
        self.model = m.eval()
        self.weights = weights

    def clip_embedding(self, wave: torch.Tensor) -> torch.Tensor:
        """The repo wrapper's CLIP-LEVEL encoder: ``forward_features`` per 1024-frame unit (mean over the patch
        tokens, THEN ``fc_norm``), averaged over the clip's units -> ``(B, 768)``. Used for clip-level tasks."""
        with torch.inference_mode():
            x = self._units(wave)  # (B, U, 1024, 128)
            b, u = x.shape[:2]
            y = self.model.forward_features(x.reshape(b * u, 1, self.unit_frames, self.N_MEL))  # (B*U, 768)
            out = y.reshape(b, u, -1).mean(dim=1)
        return out.to(wave.device)

    def unit_tokens(self, units: torch.Tensor) -> torch.Tensor:
        m = self.model
        x = m.patch_embed(units)  # (N, t*8, 768), time-major (patch_hw = (t, 8))
        x = x + m.pos_embed[:, 1:, :]
        cls = (m.cls_token + m.pos_embed[:, :1, :]).expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = m.pos_drop(x)
        for blk in m.blocks:
            x = blk(x)
        x = m.fc_norm(x[:, 1:, :])  # the wrapper's LayerNorm, per token
        return x.reshape(x.shape[0], self.t_patches, self.bands, self.token_dim)


class SSASTWindowModel(FbankPatchWindowModel):
    """SSAST-Base-Patch-400 (``third_party/SSAST/src``, needs timm 0.4.5): 512-frame units, 32 x 8 patches."""

    def __init__(self, weights: str = DEFAULT_SSAST, band_pool: str = "mean", norm: str = "authors",
                 stats: str = "audioset", unit_frames: int = 1024, stride: int = 16) -> None:
        """The official construction ``ASTModel(fshape=tshape=16, fstride=tstride=stride, input_tdim=unit_frames,
        pretrain_stage=False, load_pretrained_mdl_path=weights)`` at the pre-training geometry: 1024-frame (10.24 s)
        inputs and stride 16 (the third-party wrapper used 512 frames, which makes the model centre-crop its
        positional table; the official fine-tuning recipes use stride 10). The model itself re-derives the patch
        projection and adapts its positional table for other unit lengths."""
        src = os.path.join(SSAST_DIR, "src")
        if src not in sys.path:
            sys.path.insert(0, src)
        from models.ast_models import ASTModel  # vendored

        model = ASTModel(fshape=16, tshape=16, fstride=stride, tstride=stride, input_fdim=self.N_MEL,
                         input_tdim=unit_frames, pretrain_stage=False, load_pretrained_mdl_path=weights).eval()
        f_dim, t_dim = model.get_shape(stride, stride, self.N_MEL, unit_frames, 16, 16)
        super().__init__(unit_frames, band_pool=band_pool, name="ssast", norm=norm, stats=stats, tstride=stride, f_patches=f_dim)
        assert t_dim == self.t_patches, (t_dim, self.t_patches)
        self.model = model
        self.f_dim, self.t_dim = f_dim, t_dim
        self.weights = weights

    def clip_embedding(self, wave: torch.Tensor) -> torch.Tensor:
        """The authors' clip encoder (``ft_avgtok`` without the head): mean over ALL patch tokens of each padded unit
        after the final LayerNorm, averaged over the clip's units -> ``(B, 768)``."""
        with torch.inference_mode():
            x = self._units(wave)  # (B, U, 512, 128)
            b, u = x.shape[:2]
            tok = self.unit_tokens(x.reshape(b * u, 1, self.unit_frames, self.N_MEL))  # (B*U, t, f, 768)
            out = tok.reshape(b, u, -1, self.token_dim).mean(dim=(1, 2))
        return out.to(wave.device)

    def unit_tokens(self, units: torch.Tensor) -> torch.Tensor:
        # get_audio_representation without its final mean over tokens
        v = self.model.v
        x = units.transpose(2, 3)  # (N, 1, 128, 512): the model wants (B, C, F, T)
        x = v.patch_embed(x)  # (N, 8*32, 768), band-major (f * 32 + t)
        n = x.shape[0]
        if self.model.cls_token_num == 2:
            x = torch.cat((v.cls_token.expand(n, -1, -1), v.dist_token.expand(n, -1, -1), x), dim=1)
        else:
            x = torch.cat((v.cls_token.expand(n, -1, -1), x), dim=1)
        x = x + v.pos_embed
        x = v.pos_drop(x)
        for blk in v.blocks:
            x = blk(x)
        x = v.norm(x)[:, self.model.cls_token_num:, :]
        return x.reshape(n, self.f_dim, self.t_dim, self.token_dim).permute(0, 2, 1, 3)  # (N, t, f, 768)


class ATSTWindowModel(WindowModel):
    """ATST-Clip base (``third_party/ATST/audiossl``): 64-mel log-mel, 64 x 4 patches = one token per 40 ms."""

    HOP = 160
    WIN = 1024
    N_MEL = 64
    PATCH_T = 4
    UNIT_S = 6.0  # pre-training segment length (anchor_len), the wrapper's chunk

    def __init__(self, weights: str = DEFAULT_ATST) -> None:
        super().__init__()
        if ATST_DIR not in sys.path:
            sys.path.insert(0, ATST_DIR)
        from audiossl.models.atst.audio_transformer import AST_base  # vendored

        m = AST_base()
        state = torch.load(weights, map_location="cpu", weights_only=False)["teacher"]
        state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items()}
        msg = m.load_state_dict(state, strict=False)
        if msg.missing_keys:
            raise RuntimeError(f"ATST: missing keys {msg.missing_keys[:5]}")
        self.model = m.eval()
        self.weights = weights
        self.name = "atst"
        self.sample_rate = 16000
        self.embedding_dim = int(m.embed_dim)
        self.hop = self.HOP * self.PATCH_T  # 640 samples = 40 ms
        self.rf = self.WIN + (self.PATCH_T - 1) * self.HOP  # 1504 samples = 94 ms
        self.centre_offset = (self.PATCH_T - 1) * self.HOP / 2.0  # 240 samples: centre of the 4 frame centres
        self.max_window = None
        self.melspec = torchaudio.transforms.MelSpectrogram(
            16000, f_min=60, f_max=7800, hop_length=self.HOP, win_length=self.WIN, n_fft=self.WIN, n_mels=self.N_MEL,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        self.mm_min, self.mm_max = -79.6482, 50.6842

    @property
    def device(self) -> torch.device:
        return next(self.melspec.buffers()).device

    CHUNK_FRAMES = 601  # the wrapper's chunk (6 s + 1 frame); the model's positional table holds 250 patches (10 s)
    CLIP_BLOCKS = 12

    def clip_embedding(self, wave: torch.Tensor) -> torch.Tensor:
        """The authors' CLIP-LEVEL encoder (ATST-Clip's HEAR wrapper): for each of the last 12 blocks the CLS token
        and the length-masked mean of the patch tokens, concatenated -> ``(B, 2 x 12 x 768 = 18432)``, computed on
        601-frame chunks by the model's own ``get_intermediate_layers_chunks``. Used for clip-level tasks (ESC-50).
        One difference from the awsome wrapper, which caps ``length`` at 501 frames for clips over 5 s: the true
        frame count is passed (identical for ESC-50's 5 s clips)."""
        with torch.inference_mode():
            x = self._logmel(wave)  # (B, 1, 64, T)
            length = torch.full((wave.shape[0],), x.shape[-1], dtype=torch.long, device=x.device)
            out = self.model.get_intermediate_layers_chunks(x, length, self.CLIP_BLOCKS, self.CHUNK_FRAMES, avgpool=True)
        return out.to(wave.device)

    def fbank_frames(self, n_samples: int) -> int:
        return 1 + n_samples // self.HOP  # center=True

    def n_frames(self, n_samples: int) -> int:
        """Tokens emitted for a window: within one 601-frame chunk the WindowModel rule (a token once its full
        declared support is inside the window); a longer window (the whole-clip row) is cut into 601-frame chunks
        exactly as the authors' wrapper does (``get_intermediate_layers_chunks``), 150 tokens per full chunk and
        ``rem // 4`` for the last one — one frame is lost per chunk boundary, so those timestamps drift by 10 ms
        per chunk (whole-clip row only)."""
        if n_samples < self.rf:
            return 0
        frames = self.fbank_frames(n_samples)
        if frames <= self.CHUNK_FRAMES:
            return min(super().n_frames(n_samples), frames // self.PATCH_T)
        full, rem = divmod(frames, self.CHUNK_FRAMES)
        return full * (self.CHUNK_FRAMES // self.PATCH_T) + rem // self.PATCH_T

    def _tokens(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, 1, 64, T<=601)`` log-mel -> ``(B, T // 4, 768)`` last-layer tokens (CLS dropped)."""
        m = self.model
        length = torch.full((x.shape[0],), x.shape[-1], dtype=torch.long, device=x.device)
        t, _pos, _mel, _h, _w, patch_length = m.prepare_tokens(x, None, length, mask=False)
        for blk in m.blocks:
            t = blk(t, patch_length + 1)
        return m.norm(t)[:, 1:, :]

    def _logmel(self, wave: torch.Tensor) -> torch.Tensor:
        """``(B, L)`` -> ``(B, 1, 64, T)`` in [-1, 1]: mel power, dB with top_db per window, MinMax."""
        x = self.melspec(wave.float().to(self.device))  # (B, 64, T)
        x = torch.stack([self.to_db(row) for row in x])  # top_db clamps relative to each window's maximum
        x = (x - self.mm_min) / (self.mm_max - self.mm_min) * 2.0 - 1.0
        return x.unsqueeze(1)

    def embed_window(self, wave: torch.Tensor) -> torch.Tensor:
        n_real = self.n_frames(int(wave.shape[-1]))
        batch = wave.shape[0]
        if n_real == 0:
            return wave.new_zeros((batch, 0, self.embedding_dim))
        with torch.inference_mode():
            x = self._logmel(wave)
            frames = x.shape[-1]
            if frames <= self.CHUNK_FRAMES:
                t = self._tokens(x)
            else:  # the authors' chunking of long inputs
                parts = [self._tokens(x[..., a:a + self.CHUNK_FRAMES]) for a in range(0, frames, self.CHUNK_FRAMES)
                         if frames - a >= self.PATCH_T]
                t = torch.cat(parts, dim=1)
            if t.shape[1] < n_real:
                raise RuntimeError(f"ATST: {t.shape[1]} tokens computed, bookkeeping expected >= {n_real}")
            out = t[:, :n_real]
        return out.to(wave.device)
