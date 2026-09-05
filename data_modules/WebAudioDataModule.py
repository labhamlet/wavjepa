from functools import partial

import pytorch_lightning as pl
import torch
import torchaudio
import webdataset as wds
from torch.utils.data import DataLoader
from webdataset import RandomMix

from .dataset_functions import (
    NATIVE_AUDIO_KEYS,
    collate_native,
    decode_audio_bytes,
    pre_process,
    prepare_native_sample,
)


class WebAudioDataModule(pl.LightningDataModule):
    """WebDataset-backed audio data module with two loader pipelines.

    * **native** (default, ``legacy_pipeline=False``): the worker only decodes
      (``soundfile`` from bytes, channel 0), pads/crops to ``TARGET_SECONDS`` at
      the clip's *native* sample rate and collates a dict
      ``{"audio": (B, L_max) float32, "sr": (B,) int64, "length": (B,) int64}``.
      Resampling, RMS normalisation, cropping and masking happen on the GPU in
      the model (``JEPA.on_after_batch_transfer``).
    * **legacy** (``legacy_pipeline=True``): the original pipeline, kept for A/B
      benchmarking. ``wds.torch_audio`` decode, CPU resample to ``sr`` (with a
      cached ``Resample`` per source rate instead of one per sample), RMS
      normalise, pad to 10 s and run ``masker`` ``nr_samples_per_audio`` times;
      collated to the old tuple ``(audio, ctx_mask, tgt_mask, ctx_tgt_mask)``.
    """

    TARGET_SECONDS: int = 10
    # Kept for backward compatibility; the instance attributes below are used.
    SHUFFLE: int = 1000
    NUM_WORKERS: int = 16
    PREFETCH_FACTOR: int = 2

    def __init__(
        self,
        masker,
        data_dirs: list[str] | str,
        mixing_weights: list[int] | None = None,
        batch_size: int = 96,
        nr_samples_per_audio: int = 16,
        nr_time_points: int = 100,
        cache_size: int = 1000,
        in_channels: int = 1,
        sr: int = 16000,
        legacy_pipeline: bool = False,
        num_workers: int = 16,
        prefetch_factor: int = 4,
        shuffle_buffer: int = 1000,
        **kwargs,
    ):
        """Initialize the data module.

        Args:
            masker: mask generator used by the legacy pipeline (may be ``None``
                for the native pipeline, where masks are made in the model).
            data_dirs: shard pattern(s) (brace-expanded by webdataset).
            mixing_weights: per-dataset sampling weights for ``RandomMix`` when
                ``data_dirs`` is a list; ``None`` for a single dataset.
            batch_size: number of clips per batch.
            nr_samples_per_audio: crops per clip (legacy pipeline masks only).
            nr_time_points: tokens per crop (legacy pipeline masks only).
            in_channels: channels seen by the masker (legacy pipeline only).
            sr: target sample rate of the legacy pipeline's CPU resample.
            legacy_pipeline: select the old tuple-producing pipeline.
            num_workers / prefetch_factor / shuffle_buffer: DataLoader and
                webdataset shuffle settings.
        """
        super().__init__()
        self.data_dirs = data_dirs
        self.mixing_weights = mixing_weights
        self.batch_size = batch_size
        self.nr_samples_per_audio = nr_samples_per_audio
        self.cache_size = cache_size
        self.nr_time_points = nr_time_points

        self.masker = masker
        self.sr = sr

        self.in_channels = in_channels

        self.legacy_pipeline = legacy_pipeline
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.shuffle_buffer = shuffle_buffer

        # Legacy path: one Resample transform per (source sr, dtype), built lazily
        # inside each worker instead of once per sample.
        self._resamplers: dict[tuple[int, torch.dtype], torchaudio.transforms.Resample] = {}

    # ------------------------------------------------------------------ legacy
    def _get_resampler(self, audio_sr: int, dtype: torch.dtype) -> torchaudio.transforms.Resample:
        """Return the cached ``Resample(audio_sr -> self.sr)`` for this dtype."""
        key = (int(audio_sr), dtype)
        resampler = self._resamplers.get(key)
        if resampler is None:
            resampler = torchaudio.transforms.Resample(
                audio_sr,
                self.sr,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                dtype=dtype,
                beta=14.769656459379492,
            )
            self._resamplers[key] = resampler
        return resampler

    def _retrieve_sample(self, sample):
        """Legacy per-sample worker: decode -> CPU resample -> normalise/pad -> masks."""

        audio, audio_sr = sample[0]
        audio = audio[0, :] if audio.ndim > 1 else audio
        if audio_sr != self.sr:
            audio = self._get_resampler(audio_sr, audio.dtype)(audio)
        audio = pre_process(audio, self.sr)

        context_mask, target_indices, ctx_and_target_masks = self.masker(
            batch_size=self.nr_samples_per_audio,
            n_times=self.nr_time_points,
            in_channels=self.in_channels
        )

        return (
            audio,
            context_mask,
            target_indices,
            ctx_and_target_masks,
        )

    # ------------------------------------------------------------------ builders
    def _shard_source(self, data_path):
        """The shared ``WebDataset`` source (resampled, node/worker split)."""
        return wds.WebDataset(
            data_path,
            resampled=True,
            nodesplitter=wds.shardlists.split_by_node,
            workersplitter=wds.shardlists.split_by_worker,
            shardshuffle=False,
        )

    def _legacy_pipeline(self, data_path):
        return (
            self._shard_source(data_path)
            .repeat()
            .shuffle(self.shuffle_buffer)
            .decode(wds.torch_audio, handler=wds.warn_and_continue)
            .to_tuple("flac")
            .map(self._retrieve_sample)
            .batched(self.batch_size)
        )

    def _native_pipeline(self, data_path):
        prepare = partial(
            _prepare_native_tuple, target_seconds=self.TARGET_SECONDS
        )
        return (
            self._shard_source(data_path)
            .repeat()
            .shuffle(self.shuffle_buffer)
            .decode(decode_audio_bytes, handler=wds.warn_and_continue)
            .to_tuple(NATIVE_AUDIO_KEYS, handler=wds.warn_and_continue)
            .map(prepare, handler=wds.warn_and_continue)
            .batched(self.batch_size, collation_fn=collate_native, partial=False)
        )

    def _build_pipeline(self, data_path):
        if self.legacy_pipeline:
            return self._legacy_pipeline(data_path)
        return self._native_pipeline(data_path)

    def make_web_dataset_mixed(self, shuffle: int | None = None):
        """Create a ``RandomMix`` of one pipeline per entry of ``data_dirs``."""
        if shuffle is not None:
            self.shuffle_buffer = shuffle
        print(f"Mixed dataset with: {self.mixing_weights}")
        datasets = [self._build_pipeline(data_path) for data_path in self.data_dirs]
        return RandomMix(datasets, self.mixing_weights)

    def make_web_dataset(self, shuffle: int | None = None):
        """Create the WebDataset pipeline for audio processing."""
        if shuffle is not None:
            self.shuffle_buffer = shuffle
        return self._build_pipeline(self.data_dirs)

    def setup(self, stage: str):
        """Set up datasets for training."""
        if stage == "fit":
            if self.mixing_weights is None:
                self.audio_train = self.make_web_dataset()
            else:
                self.audio_train = self.make_web_dataset_mixed()

    def train_dataloader(self):
        """Return the training DataLoader."""
        multi = self.num_workers > 0
        loader = DataLoader(
            self.audio_train,
            batch_size=None,
            pin_memory=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor if multi else None,
            persistent_workers=multi,
        )
        return loader


def _prepare_native_tuple(sample, target_seconds: int):
    """``to_tuple`` output ``((audio, sr),)`` -> native sample dict (module-level so it pickles)."""
    audio, sr = sample[0]
    return prepare_native_sample(audio, sr, target_seconds)
