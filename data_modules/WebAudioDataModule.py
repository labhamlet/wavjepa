from torch.utils.data import DataLoader
import pytorch_lightning as pl
import webdataset as wds
from webdataset import RandomMix
import torch
import random
from typing import List
import torch.nn.functional as F


def pad_or_randomly_select(
    audio : torch.Tensor, target_length: int
    ) -> torch.Tensor:
    audio_length = audio.shape[-1]
    padding = target_length - audio_length
    if padding > 0:
        audio = F.pad(audio, (0, padding), "constant", 0)
    elif padding < 0:  # select a random 10 seconds.
        rand_index = torch.randint(0, audio_length - target_length, (1,))
        audio = audio[rand_index : rand_index + target_length]
    else:
        audio = audio
    assert audio.shape[-1] == target_length
    return audio


class WebAudioDataModule(pl.LightningDataModule):
    AUDIO_SR: int = 48000
    TARGET_SECONDS: int = 10
    SHUFFLE: int = 1000
    NUM_WORKERS: int = 16
    PREFETCH_FACTOR: int = 2

    def __init__(
        self,
        masker,
        data_dirs: List[str],
        mixing_weights: List[int],
        batch_size: int = 96,
        nr_samples_per_audio: int = 16,
        nr_time_points: int = 100,
        cache_size: int = 1000,
        in_channels: int = 1,
        **kwargs,
    ):
        """Initialize the data module with shared noise data."""
        super().__init__()
        self.data_dirs = data_dirs
        self.mixing_weights = mixing_weights
        self.batch_size = batch_size
        self.nr_samples_per_audio = nr_samples_per_audio
        self.cache_size = cache_size
        self.nr_time_points = nr_time_points

        self.masker = masker

        self.audio_target_length = self.AUDIO_SR * self.TARGET_SECONDS

        self.in_channels = in_channels

    def _retrieve_sample(self, sample):
        """Retrieves audio, noise, and RIR samples from disk
        Normalization is done later in the CoRA module.
        """

        audio, audio_sr = sample[0]

        if audio.ndim == 2 and (audio.shape[0] < audio.shape[1]):
            audio = audio[0]  # Take the left channel if it is stereo audio
        elif audio.ndim == 2 and (audio.shape[0] >= audio.shape[1]):
            audio = audio.T
            audio = audio[0]  # Take the left channel if it is stereo audio
        else:
            audio = audio
            
        audio = pad_or_randomly_select(audio, self.audio_target_length)

        context_mask, target_indices, ctx_and_target_masks = self.masker(
            batch_size=self.nr_samples_per_audio,
            n_times=self.nr_time_points,
            in_channels=self.in_channels,
        )

        return (
            audio,
            audio_sr,
            context_mask,
            target_indices,
            ctx_and_target_masks,
        )

    def make_web_dataset(self, shuffle: int):
        """Create a WebDataset pipeline for audio processing."""
        datasets = [] 
        for data_path in self.data_dirs:
            dataset = (
                wds.WebDataset(
                    data_path,
                    resampled=True,
                    nodesplitter=wds.shardlists.split_by_node,
                    workersplitter=wds.shardlists.split_by_worker,
                    shardshuffle=False,
                )
                .repeat()
                .shuffle(shuffle)
                .decode(wds.torch_audio, handler=wds.warn_and_continue)
                .to_tuple("flac")
                .map(self._retrieve_sample)
                .batched(self.batch_size)
            )
            datasets.append(dataset)

        mix = RandomMix(datasets, self.mixing_weights)
        return mix

    def setup(self, stage: str):
        """Set up datasets for training."""
        if stage == "fit":
            self.audio_train = self.make_web_dataset(
                shuffle=self.SHUFFLE
            )

    def train_dataloader(self):
        """Return the training DataLoader."""
        loader = DataLoader(
            self.audio_train,
            batch_size=None,
            pin_memory=True,
            num_workers=self.NUM_WORKERS,
            prefetch_factor=self.PREFETCH_FACTOR,
        )
        return loader
