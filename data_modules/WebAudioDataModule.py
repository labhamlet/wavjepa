from torch.utils.data import DataLoader
import pytorch_lightning as pl
import webdataset as wds
from webdataset import RandomMix
import torch
import random
from typing import List
import torch.nn.functional as F
import torchaudio.transforms as T


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


def normalize_audio(audio_data, target_dBFS=-14.0):
    rms = torch.sqrt(torch.mean(audio_data**2))  # Calculate the RMS of the audio
    if rms == 0:  # Avoid division by zero in case of a completely silent audio
        return audio_data
    current_dBFS = 20 * torch.log10(rms)  # Convert RMS to dBFS
    gain_dB = target_dBFS - current_dBFS  # Calculate the required gain in dB
    gain_linear = 10 ** (gain_dB / 20)  # Convert gain from dB to linear scale
    normalized_audio = audio_data * gain_linear  # Apply the gain to the audio data
    return normalized_audio

def pre_process(audio, audio_sr, resample_sr):
    waveform = audio[0, :] if audio.ndim > 1 else audio
    waveform = T.Resample(audio,
        waveform,
        resample_sr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,dtype=audio.dtype) if audio_sr != resample_sr else waveform
    # Normalize the audio using RMSE
    waveform = normalize_audio(waveform, -14.0)
    waveform = waveform.reshape(1, -1)
    # Make sure audio is 10 seconds
    padding = resample_sr * 10 - waveform.shape[1]
    if padding > 0:
        waveform = F.pad(waveform, (0, padding), "constant", 0)
    elif padding < 0:
        waveform = waveform[:, : resample_sr * 10]
    return waveform[0]


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
        audio = pre_process(audio, audio_sr, 16000)

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
