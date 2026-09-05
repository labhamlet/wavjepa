import copy
import math
import transformers
import numpy as np
import torchaudio

from typing import List, Any, Optional

import torch
from torch import nn
from einops import repeat, rearrange
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl

from wavjepa.pos_embed import get_1d_sincos_pos_embed_from_grid
from wavjepa.functions import trunc_normal_
from wavjepa.extractors.audio_extractor import Extractor
from wavjepa.types import ForwardReturn, TransformerLayerCFG, TransformerEncoderCFG

torch._dynamo.config.capture_dynamic_output_shape_ops = True
# Packed sequence lengths vary in multiples of ``pad_multiple``; give dynamo room
# to keep one graph per distinct (dynamic) shape it still decides to specialise on.
torch._dynamo.config.cache_size_limit = 64

#: Kaiser-window sinc resampling parameters shared with the legacy CPU loader path
#: (``data_modules/WebAudioDataModule.py``) so that GPU and CPU resampling match.
RESAMPLE_KWARGS: dict[str, Any] = dict(
    lowpass_filter_width=64,
    rolloff=0.9475937167399596,
    resampling_method="sinc_interp_kaiser",
    beta=14.769656459379492,
)

def collate_fn(batch : List[torch.Tensor]) -> torch.Tensor:
    return batch.flatten(start_dim = 0, end_dim = 1)


def pack_tokens(
    x: torch.Tensor, keep: torch.Tensor, pad_multiple: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gather the kept tokens of every row to the front of a shorter, padded sequence.

    Args:
        x: (B, S, E) tokens (positional embeddings must already be added).
        keep: (B, S) bool, True = keep this token.
        pad_multiple: the packed length ``L`` is rounded up to a multiple of this
            (bounded by ``S``) so that ``torch.compile(dynamic=True)`` sees few
            distinct shapes.

    Returns:
        x_packed: (B, L, E) kept tokens first, in their original order; pad slots are 0.
        idx: (B, L) int64 original position of every packed slot (in-range also for
            pad slots, so they can be used with ``gather``/``scatter``).
        pad: (B, L) bool True = pad slot (use as ``src_key_padding_mask``).

    ``L = min(S, ceil(max_b keep_b.sum() / pad_multiple) * pad_multiple)``. Iterating
    ``(b, l)`` over ``~pad`` in row-major order visits the kept tokens in the same
    order as ``x[keep]`` (row-major over ``(b, s)``), which the decoder scatter and
    the loss rely on.
    """
    B, S, E = x.shape
    n_keep = keep.sum(dim=1)                                       # (B,)
    L = int(n_keep.max().item())                                   # eager: .item()
    L = min(S, int(math.ceil(L / pad_multiple)) * pad_multiple)
    # Stable sort: kept positions first, original order preserved within each group.
    order = torch.argsort(~keep, dim=1, stable=True)              # (B, S)
    idx = order[:, :L]                                             # (B, L)
    pad = torch.arange(L, device=x.device)[None, :] >= n_keep[:, None]  # (B, L)
    x_packed = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, E))
    x_packed = x_packed.masked_fill(pad.unsqueeze(-1), 0.0)
    return x_packed, idx, pad


class JEPA(pl.LightningModule):
    """
    Joint-Embedding Predictive Architecture (JEPA).

    This implementation is inspired by:
        * I-JEPA http://arxiv.org/abs/2301.08243
        * Data2vec 2.0 http://arxiv.org/abs/2212.07525

    Args:
        feature_encoder:
            Does the local feature encoding. Will be shared between teacher and student.

                * Input: dict with keys: ``**batch``
                * Output: ``local_features`` (batch_size, n_patches, emb_dim)

        mask_maker:
            Computes the training masks as indices.

                * Input: dict with keys:
                    - ``**batch``
                    - ``local_features`` (batch_size, n_patches, emb_dim)

                * output: tuple:
                    - ``idxs_context`` (batch_size, self.n_contexts_per_input, n_context_patches)
                    - ``idxs_target`` (batch_size, self.n_contexts_per_input, self.n_targets_per_context, n_target_patches)

        transformer_kwargs:
            Arguments for :class:`nn.Transformer`. The transformer will have the
            following signature:

                * Input: (batch_size, n_context_patches, emb_dim)
                * Output: (batch_size, n_target_patches, emb_dim)

        loss_fn: nn.Module
            Loss function to use between the ``preds`` (output of the transformer)
            and the ``targets``.
        ema_decay: float
            initial ema decay rate.
        ema_end_decay: float
            final ema decay rate.
        ema_anneal_end_step: int
            when to finish annealing ema decay rate.
        average_top_k_layers: int
            The targets are the average of the outputs of the last k layers of
            the teacher encoder. This parameter specifies the number of layers to
            use for the average.
        use_packing: bool
            Run the student encoder/decoder only on the visible tokens (gathered
            into a shorter padded sequence) instead of the full sequence with a
            key-padding mask. Same loss, fewer FLOPs. ``False`` = original path.
        pad_multiple: int
            Packed sequence lengths are rounded up to a multiple of this.
        masker: nn.Module | None
            Vectorised mask generator with a ``sample(batch_size, n_times,
            in_channels, device, generator)`` method (see ``wavjepa/masking.py``).
            Only needed for the native dict batch path of
            :meth:`on_after_batch_transfer`; ``None`` for inference / the legacy
            tuple path.
    """
    teacher_encoder: nn.Module
    def __init__(
        self,
        feature_extractor: Extractor,
        transformer_encoder_layers_cfg : TransformerLayerCFG,
        transformer_encoder_cfg : TransformerEncoderCFG,
        transformer_decoder_layers_cfg : TransformerLayerCFG,
        transformer_decoder_cfg : TransformerEncoderCFG,
        decoder_embedding_dim : int = 512,
        loss_fn: nn.Module = nn.MSELoss(reduction='none'),
        lr: float = 0.0002,
        adam_betas: tuple[float, float] = (0.9, 0.98),
        adam_eps: float = 1e-06,
        adam_weight_decay: float = 0.01,
        ema_decay: float = 0.999,
        ema_end_decay: float = 0.99999,
        ema_anneal_end_step: int = 100000,
        average_top_k_layers: int = 12,
        resample_sr : int = 16000,
        process_audio_seconds: float = 2.00,
        nr_samples_per_audio = 16,
        use_gradient_checkpointing: bool = False,
        compile_modules : bool = False,
        size : str = "base",
        use_packing: bool = True,
        pad_multiple: int = 32,
        masker: nn.Module | None = None,
        **kwargs : dict[str, Any],
    ):
        super().__init__(**kwargs)
        self.sr = resample_sr
        self.nr_samples_per_audio = nr_samples_per_audio
        self.ema_end_step = ema_anneal_end_step
        self.target_length = int(resample_sr * process_audio_seconds)
        self.total_patches = feature_extractor.total_patches(self.target_length)
        self.use_compiled_forward = compile_modules
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_packing = use_packing
        self.pad_multiple = int(pad_multiple)
        self.save_hyperparameters(
            ignore=["feature_encoder", "feature_extractor", "loss_fn", "masker"]
        )
        self.extract_audio = feature_extractor
        self.feature_norms : nn.Module = nn.LayerNorm(self.extract_audio.embedding_dim)
        self.loss_fn = loss_fn
        # Mask generator for the native batch path (no parameters/buffers, so the
        # state_dict keys are unchanged). ``None`` keeps inference / legacy usage.
        self.masker = masker
        # ``n_times`` handed to the masker is divided by the extractor's channel count.
        self._masker_in_channels = int(getattr(feature_extractor, "in_channels", 1) or 1)
        # Native-rate resampling: one ``torchaudio.transforms.Resample`` per source
        # sr, built lazily on the module's device. Kept in a plain dict on purpose so
        # they are NOT registered as submodules (state_dict keys stay the same).
        self._resamplers: dict[int, torchaudio.transforms.Resample] = {}
        # Per-rank RNG for crops / permutation / masks (created lazily on the device).
        self._generator: torch.Generator | None = None


        # If size is large, then alter the encoder parameters to mimic VIT-Large. Should results in ~300m parameters.
        if size == "large":
            transformer_encoder_layers_cfg["nhead"] = 16
            transformer_encoder_layers_cfg["d_model"] = 1024
            transformer_encoder_layers_cfg["dim_feedforward"] = 1024 * 4
            transformer_encoder_cfg["num_layers"] = 24


        self.n_encoder_heads = transformer_encoder_layers_cfg["nhead"]
        self.encoder_embedding_dim = transformer_encoder_layers_cfg["d_model"]
        self.n_decoder_heads = transformer_decoder_layers_cfg["nhead"]
        self.decoder_embedding_dim = transformer_decoder_layers_cfg["d_model"]

        encoder_layer = nn.TransformerEncoderLayer(**transformer_encoder_layers_cfg)
        self.encoder = nn.TransformerEncoder(encoder_layer, norm = nn.LayerNorm(self.encoder_embedding_dim), **transformer_encoder_cfg)
        self.post_extraction_mapper : Optional[nn.Module] = nn.Linear(feature_extractor.embedding_dim, self.encoder_embedding_dim) if feature_extractor.embedding_dim != self.encoder_embedding_dim else None
        decoder_layer = nn.TransformerEncoderLayer(**transformer_decoder_layers_cfg)
        self.decoder = nn.TransformerEncoder(decoder_layer, norm = nn.LayerNorm(self.decoder_embedding_dim), **transformer_decoder_cfg)
        self.decoder_to_encoder_mapper = nn.Linear(self.decoder_embedding_dim, self.encoder_embedding_dim, bias=True)
        self.encoder_to_decoder_mapper = nn.Linear(self.encoder_embedding_dim, self.decoder_embedding_dim)

        # For the autocast add batch dimensions.
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.decoder_embedding_dim, requires_grad=True)
        )
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.pos_encoding_encoder = self._get_pos_embed_params(self.encoder_embedding_dim)
        self.pos_encoding_decoder = self._get_pos_embed_params(self.decoder_embedding_dim)

        self.apply(self._init_weights)
        self._init_teacher()
        if compile_modules:
            self._compile_operations()
            self.collate_fn = torch.compile(collate_fn)
        else:
            self.collate_fn = collate_fn

    def _init_weights(self, m : nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: # type: ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _get_pos_embed_params(self, embedding_dim):
        """Calculates the pos embedding embedding parameters and returns them."""
        # Update positional embedding
        pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.total_patches,
                embedding_dim,
            ),
            requires_grad=False,
        )
        positions = np.arange(self.total_patches, dtype=np.float64)
        pos_embed_data = get_1d_sincos_pos_embed_from_grid(
            embedding_dim,
            positions,
        )
        pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float().unsqueeze(0))
        return pos_embed

    def _init_teacher(self):
        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_encoder.requires_grad_(False)

    def _get_ema_decay(self):
        if self.global_step >= self.ema_end_step:
            return self.hparams.ema_end_decay
        r = self.hparams.ema_end_decay - self.hparams.ema_decay
        pct_remaining = 1 - self.global_step / self.ema_end_step
        return self.hparams.ema_end_decay - r * pct_remaining

    @torch.no_grad()
    def _step_teacher(self):
        """EMA update ``teacher = r * teacher + (1 - r) * student`` with fused foreach ops."""
        r = self._get_ema_decay()
        teacher_params = [p for p in self.teacher_encoder.parameters()]
        student_params = [p.detach() for p in self.encoder.parameters()]
        torch._foreach_mul_(teacher_params, r)
        torch._foreach_add_(teacher_params, student_params, alpha=1 - r)

    def _compile_operations(self):
        """
        Use torch.compile on the extractor, encoder and decoder blocks for faster forward.

        Packed path: the encoder/decoder *core* calls see sequence lengths that vary
        in multiples of ``pad_multiple``, so they are compiled with ``dynamic=True``;
        the pack/gather code (``.item()``, boolean indexing) and the packed loss stay
        eager. The teacher (fixed full-sequence shape) keeps its fullgraph compile.
        """
        try:
            torch._dynamo.config.cache_size_limit = 64
            if self.use_packing:
                self._encoder_core = torch.compile(self._encoder_core, dynamic=True)
                self._decoder_core = torch.compile(self._decoder_core, dynamic=True)
            else:
                self.encoder_forward = torch.compile(self.encoder_forward, fullgraph=True)
                self.decoder_forward = torch.compile(self.decoder_forward, fullgraph=True)
                self.masked_loss = torch.compile(self.masked_loss)
            self._forward_teacher = torch.compile(self._forward_teacher, fullgraph=True)
            self.extract_audio = torch.compile(self.extract_audio)

        except Exception as e:
            print(f"Warning: Could not compile operations: {e}")
            self.use_compiled_forward = False

    def configure_optimizers(self):
        trainables = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainables,
            lr=self.hparams.lr,
            betas=self.hparams.adam_betas,
            eps=self.hparams.adam_eps,
            weight_decay=self.hparams.adam_weight_decay,
        )
        cosine_annealing = transformers.get_cosine_schedule_with_warmup(optimizer,
                                 num_warmup_steps=100000, num_training_steps=self.trainer.max_steps)

        return {"optimizer": optimizer,
                'lr_scheduler' : {"scheduler": cosine_annealing, "interval": "step"}}

    def _make_targets(self, layer_outputs : List[torch.Tensor]):
        """
        Predicting targets which are the average of multiple layers is more robust than
        predicting only the top most layer (K = 1) for most modalities.
        Args:
            layer_outputs: average_top_k_layers * (batch_size, n_patches, emb_dim)

        Returns:
            array of shape (batch_size, n_patches, emb_dim)
        """

        # They have for audioset -> instance_norm_target_layer: true
        # They have for audioset -> layer_norm_targets : true
        # So this is the way following the data2vec2 paper for audio.
        stacked_outputs = torch.stack(layer_outputs, )  # [num_layers, batch, seq_len, features]
        transposed = stacked_outputs.transpose(2, 3)   # [num_layers, batch, features, seq_len]

        # Apply instance norm to all layers simultaneously
        normalized = F.instance_norm(transposed)       # [num_layers, batch, features, seq_len]
        normalized = normalized.transpose(2, 3)        # [num_layers, batch, seq_len, features]

        # Compute mean across layers
        y = normalized.mean(dim=0)                     # [batch, seq_len, features]
        return y

    @torch.no_grad()
    def _forward_teacher(self, x : torch.Tensor) -> torch.Tensor:
        layer_outputs = []
        for i, bl in enumerate(self.teacher_encoder.layers): # type: ignore
            x : torch.Tensor = bl(x)
            if (
                len(self.teacher_encoder.layers) - i
                <= self.hparams.average_top_k_layers
            ):
                layer_outputs.append(x)

        if self.hparams.average_top_k_layers > 1:
            targets = self._make_targets(layer_outputs)  # (batch_size, n_patches, emb_dim)
        else:
            targets = layer_outputs[-1]
        return targets

    def get_aug_prob(self):
        return 1 - (self.global_step / self.trainer.max_steps)

    @staticmethod
    def _same_device(a: torch.device, b: torch.device) -> bool:
        """``cuda`` and ``cuda:0`` count as the same device (a generator always carries an index)."""
        return a.type == b.type and (a.index is None or b.index is None or a.index == b.index)

    def _get_generator(self, device: torch.device | str | None = None) -> torch.Generator:
        """Per-rank ``torch.Generator`` on ``device`` (default: the module's device), lazily created.

        Seeded with ``torch.initial_seed() + global_rank`` so DDP ranks draw
        different crops / permutations / masks even though ``seed_everything``
        gives every rank the same main-process seed. Re-created (and re-seeded)
        only if the module moved to another device after the first use.
        """
        device = torch.device(self.device if device is None else device)
        gen = self._generator
        if gen is None or not self._same_device(torch.device(gen.device), device):
            gen = torch.Generator(device=device)
            gen.manual_seed(int(torch.initial_seed() + int(self.global_rank)) % (2**63 - 1))
            self._generator = gen
        return gen

    def _get_resampler(self, orig_sr: int, device: torch.device) -> torchaudio.transforms.Resample:
        """Cached ``Resample(orig_sr -> self.sr)`` (kaiser params of the legacy path) on ``device``."""
        key = int(orig_sr)
        resampler = self._resamplers.get(key)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(
                key, self.sr, dtype=torch.float32, **RESAMPLE_KWARGS
            )
            self._resamplers[key] = resampler
        if resampler.kernel.device != device:
            resampler.to(device)
        return resampler

    @torch.no_grad()
    def _prepare_wave16k(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Native-rate dict batch -> ``(wave (B, sr*10) float32, valid (B,) int64)``.

        1. Rows are grouped by native ``sr``; each group is cropped to ``sr*10``
           samples, resampled to ``self.sr`` (skipped when ``sr == self.sr``) with the
           cached kaiser ``Resample`` and cropped/zero-padded to ``self.sr*10``.
           Runs outside autocast, in float32 and with cuDNN TF32 disabled so that the
           result matches the CPU ``Resample`` of the legacy loader.
        2. Every row is RMS-normalised to -14 dBFS like
           ``dataset_functions.normalize_audio`` but using only its ``valid`` samples
           (``round(length / sr * self.sr)``), i.e. as if normalisation ran before
           zero padding. Rows with ``rms == 0`` are left unchanged.
        """
        audio: torch.Tensor = batch["audio"]
        sr: torch.Tensor = batch["sr"]
        length: torch.Tensor = batch["length"]
        if audio.ndim == 3:  # (B, 1, L) -> (B, L)
            audio = audio[:, 0, :]
        device = audio.device
        B = audio.shape[0]
        out_len = int(self.sr * 10)
        sr = sr.to(device=device, dtype=torch.int64)
        length = length.to(device=device, dtype=torch.int64)

        wave = torch.zeros(B, out_len, dtype=torch.float32, device=device)
        cudnn = torch.backends.cudnn
        with torch.autocast(device_type=device.type, enabled=False), cudnn.flags(
            enabled=cudnn.enabled,
            benchmark=False,
            benchmark_limit=cudnn.benchmark_limit,
            deterministic=cudnn.deterministic,
            allow_tf32=False,
        ):
            for s in torch.unique(sr).tolist():
                s = int(s)
                rows = torch.nonzero(sr == s, as_tuple=True)[0]
                x = audio[rows, : s * 10].to(torch.float32)
                if s != self.sr:
                    x = self._get_resampler(s, device)(x)
                n = x.shape[-1]
                if n >= out_len:
                    wave[rows] = x[:, :out_len]
                else:
                    wave[rows, :n] = x

            # valid samples per row at self.sr (round like the DESIGN contract)
            valid = torch.round(length.to(torch.float64) * self.sr / sr.to(torch.float64)).to(torch.int64)
            valid = valid.clamp_(min=0, max=out_len)
            valid_mask = torch.arange(out_len, device=device)[None, :] < valid[:, None]   # (B, out_len)

            # RMS to -14 dBFS over the valid samples only (== normalize_audio before padding)
            sq_sum = (wave * wave * valid_mask).sum(dim=1)
            rms = torch.sqrt(sq_sum / valid.clamp(min=1).to(torch.float32))              # (B,)
            safe_rms = torch.where(rms > 0, rms, torch.ones_like(rms))
            current_dBFS = 20 * torch.log10(safe_rms)
            gain_dB = -14.0 - current_dBFS
            gain_linear = 10 ** (gain_dB / 20)
            gain_linear = torch.where(rms > 0, gain_linear, torch.ones_like(gain_linear))
            wave = wave * gain_linear[:, None]
        return wave, valid

    def _crop_and_mask(self, audio_batch: torch.Tensor):
        """Native path steps 3-5: random crops, standardise, bf16, GPU masks, shuffle.

        Args:
            audio_batch: (B, C, L) float waveform at ``self.sr``.
        Returns:
            ``(audio (B*n, C, target_length) bf16, ctx (B*n, S), tgt (B*n, N, S),
            ctx_tgt (B*n, N, S))`` with every row shuffled by one permutation.
        """
        if self.masker is None or not hasattr(self.masker, "sample"):
            raise RuntimeError(
                "JEPA.on_after_batch_transfer got a native dict batch but no vectorised "
                "`masker` (with a `.sample` method) was passed to the constructor."
            )
        device = audio_batch.device
        gen = self._get_generator(device)
        B, C, L_full = audio_batch.shape
        n = self.nr_samples_per_audio
        rand_starts = torch.randint(
            0, L_full - self.target_length + 1, (B, n), device=device, generator=gen
        )
        indices = rand_starts.unsqueeze(-1) + torch.arange(self.target_length, device=device)
        indices_expanded = indices.unsqueeze(2).expand(-1, -1, C, -1)
        expanded = audio_batch.unsqueeze(1).expand(-1, n, -1, -1)
        crops = torch.gather(expanded, 3, indices_expanded)                    # (B, n, C, T)

        mean = crops.mean(dim=(-2, -1), keepdim=True)
        std = crops.std(dim=(-2, -1), keepdim=True)
        crops = (crops - mean) / (std + 1e-5)

        flattened = self.collate_fn(crops.to(torch.bfloat16))                  # (B*n, C, T)
        total = flattened.shape[0]

        ctx_masks, target_indices, ctx_and_target_masks = self.masker.sample(
            total, self.total_patches, in_channels=self._masker_in_channels,
            device=device, generator=gen,
        )
        perm = torch.randperm(total, device=device, generator=gen)
        return (
            flattened[perm],
            ctx_masks[perm].to(torch.bool),
            target_indices[perm].to(torch.bool),
            ctx_and_target_masks[perm].to(torch.bool),
        )

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Runs on GPU. Splits batch by SR, resamples, recombines.

        * ``dict`` batch (native loader, ``{"audio", "sr", "length"}``): GPU resample
          + RMS normalisation (:meth:`_prepare_wave16k`), then the 8-crop logic and
          the vectorised masker on the device (:meth:`_crop_and_mask`).
        * ``tuple`` batch (legacy loader): the original code path, unchanged.
        """
        if isinstance(batch, dict):
            wave, _valid = self._prepare_wave16k(batch)
            return self._crop_and_mask(wave.unsqueeze(1))

        (
            audio_batch,
            ctx_masks,
            target_indices,
            ctx_and_target_masks,
        ) = batch

        if audio_batch.ndim != 3:
            audio_batch = audio_batch.unsqueeze(1)

        B, C, L_full = audio_batch.shape
        # Generate all random start indices at once
        rand_starts = torch.randint(
            0, L_full - self.target_length + 1,
            (B, self.nr_samples_per_audio),
            device=self.device
        )

        # Create indices for gathering
        # Shape: (B, nr_samples, target_length)
        indices = rand_starts.unsqueeze(-1) + torch.arange(self.target_length, device=self.device)
        indices_expanded = indices.unsqueeze(2).expand(-1, -1, C, -1)

        clean_scene_expanded = audio_batch.unsqueeze(1).expand(-1, self.nr_samples_per_audio, -1, -1)

        return_clean_audios = torch.gather(clean_scene_expanded, 3, indices_expanded)

        mean = return_clean_audios.mean(dim=(-2, -1), keepdim=True)
        std = return_clean_audios.std(dim=(-2, -1), keepdim=True)
        normalized_clean_audios = (return_clean_audios - mean) / (std + 1e-5) # Add epsilon for stability

        # Cast to bfloat16 and flatten batch and samples dimensions
        flattened_clean = self.collate_fn(normalized_clean_audios.to(torch.bfloat16))

        # Shuffle the samples
        idx = torch.randperm(flattened_clean.size(0))

        return flattened_clean[idx, ...], self.collate_fn(ctx_masks), self.collate_fn(target_indices), self.collate_fn(ctx_and_target_masks)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> ForwardReturn:
        audio_input, ctx_masks, target_indices, ctx_and_target_masks = batch
        out = self(audio_input,ctx_masks, target_indices, ctx_and_target_masks)

        # Enhanced logging
        log_data = {
            "train/loss": out['loss'],
            "ema" : self._get_ema_decay(),
        }

        self.log_dict(log_data, prog_bar=True, sync_dist=True)

        # EMA of the student -> teacher once per OPTIMIZER step. With gradient accumulation the
        # student only changes every `accumulate_grad_batches` micro-batches, so updating on every
        # micro-batch would just apply the decay several times per step; this keeps the original
        # cadence (update on the last micro-batch, i.e. with the pre-step student weights).
        accumulate = 1
        if self._trainer is not None:
            accumulate = int(getattr(self.trainer, "accumulate_grad_batches", 1) or 1)
        if (batch_idx + 1) % accumulate == 0:
            with torch.amp.autocast('cuda', enabled=False):  # Force FP32 computation for stability
                self._step_teacher()

        return out

    def masked_loss(self, pred, target, target_indices):
        """
        Calculates the masked loss using broadcasting to avoid memory-heavy repeats.

        pred:   Tensor of shape [(B * N), T, D]
        target: Tensor of shape [B, T, D]
        mask:   Tensor of shape [B, N, T]
        """
        B, N, _ = target_indices.shape
        D = pred.shape[-1]

        pred_reshaped = pred.view(B, N, -1, D)

        # This makes it broadcastable to the shape of pred_reshaped [B, N, T, D] during the loss

        target = repeat(target, "B T D -> B N T D", N = N)
        loss = self.loss_fn(pred_reshaped, target)  # -> Shape: [B, N, T, D]

        loss_per_timestep = loss.mean(dim=-1)  # -> Shape: [B, N, T]

        # No rearrange is needed for the mask.
        masked_loss_tensor = loss_per_timestep * target_indices  # -> Shape: [B, N, T]

        # Calculate the final mean loss over only the masked elements.
        total_loss = masked_loss_tensor.sum()
        indices_count = target_indices.sum()

        return total_loss / (indices_count + 1e-8)

    def masked_loss_packed(
        self,
        preds_packed: torch.Tensor,
        targets: torch.Tensor,
        target_indices: torch.Tensor,
        idx2: torch.Tensor,
        pad2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Loss of the packed decoder output; equals :meth:`masked_loss` on the unpacked
        predictions because every target position is decoder-visible (hence packed)
        and the loss only counts target positions.

        preds_packed:   [(B * N), L2, D] packed decoder predictions
        targets:        [B, S, D] teacher targets (full sequence)
        target_indices: [B, N, S] bool, True = target position
        idx2 / pad2:    [(B * N), L2] original position / pad flag of each packed slot
        """
        B, N, S = target_indices.shape
        D = targets.shape[-1]
        targets_rep = repeat(targets, "B S D -> (B N) S D", N=N)
        gathered_targets = torch.gather(targets_rep, 1, idx2.unsqueeze(-1).expand(-1, -1, D))
        target_mask = rearrange(target_indices, "B N S -> (B N) S")
        tmask = torch.gather(target_mask, 1, idx2) & ~pad2                     # (B*N, L2)

        loss = self.loss_fn(preds_packed, gathered_targets)                    # (B*N, L2, D)
        loss_per_timestep = loss.mean(dim=-1)                                  # (B*N, L2)
        masked_loss_tensor = loss_per_timestep * tmask
        return masked_loss_tensor.sum() / (tmask.sum() + 1e-8)


    def forward(self, audio : torch.Tensor, ctx_masks, target_indices, ctx_and_target_masks) -> ForwardReturn:
        """
        Args:
            batch: torch.Tensor
                audio data

        Returns:
            output: dict with keys:

                * loss: scalar
                * local_features: (batch_size, n_patches, emb_dim)
                    Output of ``feature_encoder``. Shared between teacher and student.
                * contextual_features: (batch_size, n_patches, emb_dim) if compute_loss=False, (batch_size, n_contexts_per_input, n_context_patches, emb_dim) if compute_loss=True.
                    Outpyt of the student ``tramnsformer.encoder``.
                * preds: (batch_size, n_contexts_per_input, n_targets_per_context, n_target_patches, emb_dim)
                    Output of the student transformer (encoder+decoder).
                    With ``use_packing`` this is the packed ``(B*N, L2, emb_dim)`` tensor
                    and ``preds_idx`` / ``preds_pad`` ``(B*N, L2)`` give the original
                    position / pad flag of every slot.
                * targets: (batch_size, n_patches, emb_dim)
                    Average of the outputs of the last k layers of the
                    teacher ``transformer.encoder``.
                * idxs_context: (batch_size, n_contexts_per_input, n_context_patches)
                    Indices of the unmasked patches to use to compute the contextual features.
                * idxs_target: (batch_size, n_contexts_per_input, n_targets_per_context, n_target_patches)
                    Indices of the patches that must be predicted by the student.
        """
        # # Compute the local representations from the waveform
        # This extract audio can be also channel based, if it is channel based the channel are flatten to the sequencel length
        local_features = self.extract_audio(audio)
        local_features = self.feature_norms(local_features)
        if self.post_extraction_mapper is not None:
            local_features = self.post_extraction_mapper(local_features)

        local_features = local_features + self.pos_encoding_encoder

        if self.use_packing:
            # Student encoder on the context tokens only -> (n_ctx_total, E_enc),
            # row-major (b, position) order like ``contextual_features[~ctx_masks]``.
            contextual_features = self.encoder_forward_packed(local_features, ctx_masks)
            contextual_features = self.encoder_to_decoder_mapper(contextual_features)

            preds, preds_idx, preds_pad = self.decoder_forward_packed(
                contextual_features,
                ctx_masks,
                nr_targets=target_indices.shape[1],
                ctx_and_target_masks=ctx_and_target_masks,
            )

            # Compute the training targets using the teacher (full sequence).
            x_targets = local_features.detach()
            targets = self._forward_teacher(x_targets)

            loss = self.masked_loss_packed(preds, targets, target_indices, preds_idx, preds_pad)

            out = ForwardReturn(
                local_features=local_features,
                contextual_features=contextual_features,
                loss=loss,
                preds=preds,
                targets=targets,
            )
            out["preds_idx"] = preds_idx
            out["preds_pad"] = preds_pad
            return out

        contextual_features = self.encoder_forward(local_features, src_key_padding_mask=ctx_masks)
        # Accumulate contextual features on the batch dimensions
        contextual_features = contextual_features[~ctx_masks]
        contextual_features = self.encoder_to_decoder_mapper(contextual_features)

        preds = self.decoder_forward(contextual_features,
                                     ctx_masks,
                                     nr_targets = target_indices.shape[1],
                                     src_key_padding_mask=ctx_and_target_masks)

        # Compute the training targets using the teacher.
        x_targets = local_features.detach()
        targets = self._forward_teacher(x_targets)

        loss = self.masked_loss(preds, targets, target_indices)

        return ForwardReturn(
            local_features=local_features,
            contextual_features=contextual_features,
            loss=loss,
            preds=preds,
            targets=targets,
        )


    def decoder_forward(self, contextual_features: torch.Tensor, ctx_mask: torch.BoolTensor, nr_targets : int, src_key_padding_mask : Optional[torch.BoolTensor] = None) -> torch.Tensor:
        B = ctx_mask.shape[0]
        # Prepare the mask tokens.
        tgt = self.mask_token.repeat(B, self.total_patches, 1).type_as(contextual_features) # (B, seq_len, decoder_dim)
        #Get the context tokens.
        tgt[~ctx_mask, :] = contextual_features.reshape((-1, self.decoder_embedding_dim))
        tgt = tgt.reshape((B, -1, self.decoder_embedding_dim))
        # Add positional encoding to the decoder
        tgt = tgt + self.pos_encoding_decoder

        # Repeat the context for every target, and absorb into batch dimension
        tgt = repeat(tgt, 'B Seq Emb -> B T Seq Emb', T = nr_targets)
        tgt = rearrange(tgt, 'B T Seq Emb -> (B T) Seq Emb')
        src_key_padding_mask = rearrange(src_key_padding_mask, 'B T Seq1 -> (B T) Seq1')

        #Decoder only attends to context tokens and target mask tokens.
        tgt = self.decoder(tgt, src_key_padding_mask = src_key_padding_mask)
        preds = self.decoder_to_encoder_mapper(tgt)
        return preds


    #TODO use flex attention
    def encoder_forward(self,
    x_contexts: torch.Tensor,
    src_key_padding_mask : Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:

        if self.use_gradient_checkpointing and self.training:
            contextual_features = checkpoint(self.encoder, x_contexts, use_reentrant=False)
        else:
            contextual_features = self.encoder(x_contexts, src_key_padding_mask = src_key_padding_mask)

        return contextual_features

    # ------------------------------------------------------------------ #
    # Packed path (DESIGN.md section C)
    # ------------------------------------------------------------------ #
    def _encoder_core(self, x_packed: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """Student encoder on a packed batch; compiled with ``dynamic=True`` when enabled."""
        if self.use_gradient_checkpointing and self.training:
            return checkpoint(self.encoder, x_packed, None, pad, use_reentrant=False)
        return self.encoder(x_packed, src_key_padding_mask=pad)

    def _decoder_core(self, tgt_packed: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """Decoder on a packed batch; compiled with ``dynamic=True`` when enabled."""
        return self.decoder(tgt_packed, src_key_padding_mask=pad)

    def encoder_forward_packed(self, local_features: torch.Tensor, ctx_masks: torch.Tensor) -> torch.Tensor:
        """
        Run the student encoder on the context tokens only.

        Args:
            local_features: (B, S, E) tokens with positional embeddings added.
            ctx_masks: (B, S) bool, True = NOT context.
        Returns:
            (n_ctx_total, E) encoder outputs of the context tokens in row-major
            ``(b, position)`` order -- identical ordering to
            ``self.encoder(...)[~ctx_masks]`` of the unpacked path.
        """
        x_packed, _idx, pad = pack_tokens(local_features, ~ctx_masks, self.pad_multiple)
        out = self._encoder_core(x_packed, pad)
        return out[~pad]

    def decoder_forward_packed(
        self,
        contextual_features: torch.Tensor,
        ctx_mask: torch.Tensor,
        nr_targets: int,
        ctx_and_target_masks: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decoder on the decoder-visible tokens (context + target mask tokens) only.

        Args:
            contextual_features: (n_ctx_total, E_dec) context tokens already mapped to
                the decoder width (same layout as :meth:`decoder_forward`).
            ctx_mask: (B, S) bool, True = NOT context.
            nr_targets: T, target groups per row.
            ctx_and_target_masks: (B, T, S) bool, True = NOT visible to the decoder.
        Returns:
            ``(preds_packed (B*T, L2, E_enc), idx2 (B*T, L2) int64, pad2 (B*T, L2) bool)``.
        """
        B = ctx_mask.shape[0]
        # Prepare the mask tokens and blend in the context (exactly as decoder_forward).
        tgt = self.mask_token.repeat(B, self.total_patches, 1).type_as(contextual_features)
        tgt[~ctx_mask, :] = contextual_features.reshape((-1, self.decoder_embedding_dim))
        tgt = tgt.reshape((B, -1, self.decoder_embedding_dim))
        tgt = tgt + self.pos_encoding_decoder

        # Repeat the context for every target group and absorb into the batch dim.
        tgt = repeat(tgt, 'B Seq Emb -> B T Seq Emb', T=nr_targets)
        tgt = rearrange(tgt, 'B T Seq Emb -> (B T) Seq Emb')
        keep = ~rearrange(ctx_and_target_masks, 'B T Seq1 -> (B T) Seq1')

        tgt_packed, idx2, pad2 = pack_tokens(tgt, keep, self.pad_multiple)
        out = self._decoder_core(tgt_packed, pad2)
        preds = self.decoder_to_encoder_mapper(out)
        return preds, idx2, pad2

    @torch.inference_mode()
    def get_audio_representation(self, audio : torch.Tensor, padding_mask : torch.tensor):
        # Get the audio representatin of waveform x.
        self.eval()
        local_features = self.extract_audio(audio)
        local_features = self.feature_norms(local_features)
        if self.post_extraction_mapper:
            local_features = self.post_extraction_mapper(local_features)
        local_features = local_features + self.pos_encoding_encoder
        # Encoder and decoder forward
        contextual_features = self.encoder_forward(local_features, src_key_padding_mask = padding_mask)
        return contextual_features
