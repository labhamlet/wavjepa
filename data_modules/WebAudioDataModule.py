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

def mask_to_indices(mask: torch.BoolTensor, check: bool = False) -> torch.Tensor:
    """
    Returns the indices of the true elements.

    Args:
        mask: (*batch_dims, masked_dim)
            Boolean mask. For every element, the number of true values in the masked dimension
            should be the same. i.e. ``mask.sum(dim=-1)`` should be constant.
        check: bool
            If ``True``, check that the output is correct. Slower.

    Returns:
        indices: (*batch_dims, n_unmasked)
            Indices of the true elements in the mask.
    """

    n_true = mask.sum(dim=-1).unique()
    assert n_true.size(0) == 1
    n_true = int(n_true.item())
    batch_dims = list(mask.shape[:-1])
    if check:
        out = mask.nonzero(as_tuple=False)
        out = out.view(*batch_dims, n_true, len(batch_dims) + 1)
        for i, d in enumerate(batch_dims):
            for j in range(0, d):
                assert (out[..., i].select(i, j) == j).all()
        out = out[..., -1]
    else:
        *_, out = mask.nonzero(as_tuple=True)
        out = out.view(*batch_dims, n_true)
    return out.to(mask.device)


# using cluster for frame masking hurts the performance, so just use the naive random sampling
def gen_maskid_frame_tgt(masked_patches: torch.Tensor, mask_size: int = 100):
    indices: List[int] = masked_patches.nonzero().flatten().tolist()  # type: ignore
    mask_id: List[int] = random.sample(indices, mask_size)
    return torch.tensor(mask_id)


# using cluster for frame masking hurts the performance, so just use the naive random sampling
def gen_maskid_frame(sequence_len: int = 512, mask_size: int = 100) -> torch.Tensor:
    mask_id = random.sample(range(0, sequence_len), mask_size)
    return torch.tensor(mask_id)


def to_torch(sample):
    return torch.from_numpy(sample[0])



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

        self.noise_target_length = self.NOISE_SR * self.TARGET_SECONDS
        self.audio_target_length = self.AUDIO_SR * self.TARGET_SECONDS

        self.in_channels = in_channels

    def _retrieve_sample(self, sample):
        """Retrieves audio, noise, and RIR samples from disk
        Normalization is done later in the CoRA module.
        """

        audio, audio_sr = sample[0]
        audio = audio[0]  # Take the left channel if it is stereo audio
        audio_length = audio.shape[-1]

        #Pad or randomly select depending on the target_length.

        #We will get more than 10 seconds of the librispeech audio, or audioset 44.1kHz audio here
        #But we deal with this later in the WavJEPA on_after_batch_transfer.
        audio = pad_or_randomly_select(audio, self.audio_target_length)

        context_mask, target_indices, ctx_and_target_masks = self.masker(
            batch_size=self.nr_samples_per_audio,
            n_times=self.nr_time_points,
            in_channels=self.in_channels,
        )

        return (
            audio,
            audio_sr,
            audio_length,
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
