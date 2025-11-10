import multiprocessing as mp
import queue
import random

import pytorch_lightning as pl
import torch
import webdataset as wds
from torch.utils.data import DataLoader

from .dataset_functions import pre_process_audio
from .scene_module import generate_scenes


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
    indices: list[int] = masked_patches.nonzero().flatten().tolist()  # type: ignore
    mask_id: list[int] = random.sample(indices, mask_size)
    return torch.tensor(mask_id)


# using cluster for frame masking hurts the performance, so just use the naive random sampling
def gen_maskid_frame(sequence_len: int = 512, mask_size: int = 100) -> torch.Tensor:
    mask_id = random.sample(range(0, sequence_len), mask_size)
    return torch.tensor(mask_id)


def to_torch(sample):
    return torch.from_numpy(sample[0])


class NoiseDataManager:
    """Manages RIR data loading with multiprocessing in the main process."""

    def __init__(
        self, noise_data_dir: str, buffer_size: int = 500, num_workers: int = 1
    ):
        self.noise_data_dir = noise_data_dir
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.manager = mp.Manager()
        self.noise_queue = self.manager.Queue(maxsize=buffer_size)
        self.stop_event = self.manager.Event()
        self.processes = []
        self.started = False

    def _worker(self):
        """Worker process to load RIR data."""

        def to_torch(sample):
            return torch.from_numpy(sample[0]).float()

        shuffle_buffer = 100
        dataset = (
            wds.WebDataset(self.noise_data_dir, resampled=True, shardshuffle=False)
            .repeat()
            .shuffle(shuffle_buffer)
            .decode("pil")
            .to_tuple("npy")
            .map(to_torch)
        )

        loader = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=self.num_workers,
                prefetch_factor=4,
                batch_size=None,
            )
        )
        print("Noise Loader is set", flush=True)
        while not self.stop_event.is_set():
            try:
                rirs = next(loader)
                self.noise_queue.put(rirs, timeout=1.0)
            except queue.Full:
                continue

    def start(self):
        """Start the Noise loading process."""
        if not self.started:
            self.process = mp.Process(target=self._worker, daemon=False)
            self.process.start()
            self.started = True
        return self

    def __next__(self, timeout: float = 1.0):
        """Get Noise data from the queue."""
        try:
            return self.noise_queue.get(timeout=timeout)
        except queue.Empty:
            # Return some default RIR data if queue is empty
            return self.__next__()

    def stop(self):
        """Stop the Noise loading process."""
        if self.started:
            self.stop_event.set()
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                self.process.terminate()
            self.started = False

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.stop()


class RIRDataManager:
    """Manages RIR data loading with multiprocessing in the main process."""

    def __init__(self, rir_data_dir: str, buffer_size: int = 500, num_workers: int = 4):
        self.rir_data_dir = rir_data_dir
        self.buffer_size = buffer_size
        self.num_workers = num_workers
        self.manager = mp.Manager()
        self.rir_queue = self.manager.Queue(maxsize=buffer_size)
        self.stop_event = self.manager.Event()
        self.processes = []
        self.started = False

    def _worker(self):
        """Worker process to load RIR data."""

        def to_torch(sample):
            return torch.from_numpy(sample[0]).float()

        shuffle_buffer = 100
        dataset = (
            wds.WebDataset(self.rir_data_dir, resampled=True, shardshuffle=False)
            .repeat()
            .shuffle(shuffle_buffer)
            .decode("pil")
            .to_tuple("npy")
            .map(to_torch)
        )

        loader = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=self.num_workers,
                prefetch_factor=4,
                batch_size=None,
            )
        )
        print("RIR Loader is set", flush=True)
        while not self.stop_event.is_set():
            try:
                rirs = next(loader)
                self.rir_queue.put(rirs, timeout=1.0)
            except queue.Full:
                continue

    def start(self):
        """Start the RIR loading process."""
        if not self.started:
            self.process = mp.Process(target=self._worker, daemon=False)
            self.process.start()
            self.started = True
        return self

    def __next__(self, timeout: float = 1.0):
        """Get RIR data from the queue."""
        try:
            return self.rir_queue.get(timeout=timeout)
        except queue.Empty:
            # Return some default RIR data if queue is empty
            return self.__next__()

    def stop(self):
        """Stop the RIR loading process."""
        if self.started:
            self.stop_event.set()
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                self.process.terminate()
            self.started = False

    def __del__(self):
        """Ensure cleanup on deletion."""
        self.stop()


class WebAudioDataModuleLMDB(pl.LightningDataModule):
    def __init__(
        self,
        base_data_dir: str,
        val_data_dir: str,
        rir_data_dir: str,
        base_noise_dir: str,
        batch_size: int = 32,
        ambisonic: bool = False,
        with_noise: bool = False,
        with_rir: bool = False,
        masker=None,
        nr_samples_per_audio: int = 16,
        nr_time_points: int = 100,
        cache_size: int = 1000,
        **kwargs,
    ):
        """Initialize the data module with shared noise data."""
        super().__init__()
        self.datapath = base_data_dir
        self.val_path = val_data_dir
        self.noise_dir = base_noise_dir
        self.batch_size = batch_size
        self.sr = 32000
        self.masker = masker
        self.nr_samples_per_audio = nr_samples_per_audio
        self.cache_size = cache_size
        self.nr_time_points = nr_time_points

        self.with_noise = with_noise
        self.with_rir = with_rir
        if self.with_rir:
            self.in_channels = 4 if ambisonic else 2
        else:
            self.in_channels = 1

        self.with_noise = with_noise
        self.with_rir = with_rir

        if self.with_noise:
            self.noise_loader = NoiseDataManager(base_noise_dir).start()

        if self.with_rir:
            self.rir_loader = RIRDataManager(rir_data_dir).start()

    def _augment_sample(self, sample):
        """Augment sample with noise and RIR data."""

        audio, audio_sr = sample[0]
        audio = pre_process_audio(audio, audio_sr, self.sr)
        # Initialize all variables
        noise = None
        noise_rirs = None
        snr = None
        source_rir = None

        # If with the rir, load the rir.
        if self.with_rir:
            rirs = next(self.rir_loader)
            source_rir = rirs[0]

        # If with the noise, load the noise and fade it w.r.t audio's duration.
        if self.with_noise:
            noise = next(self.noise_loader)
            noise = generate_scenes.fade_noise(noise, audio, self.sr)

            # If with the noise and the RIR, get the noise RIRs. These always correspond to 1:
            if self.with_rir:
                noise_rirs = rirs[1:]

            # If audio is bigger than noise, then place the noise in a random location of the audio
            if audio.shape[-1] > noise.shape[-1]:
                start_idx_noise = torch.randint(
                    0, audio.shape[-1] - noise.shape[-1], (1,)
                )
                new_agg_noise = torch.zeros_like(audio)
                new_agg_noise[start_idx_noise : start_idx_noise + noise.shape[-1]] = (
                    noise
                )
                noise = new_agg_noise
            snr_val = torch.distributions.uniform.Uniform(5, 40).sample()
            snr = torch.FloatTensor([2]).fill_(snr_val)

        context_mask, target_indices, ctx_and_target_masks = self.masker(
            batch_size=self.nr_samples_per_audio,
            n_times=self.nr_time_points,
            in_channels=self.in_channels,
        )

        return (
            audio,
            noise,
            source_rir,
            noise_rirs,
            snr,
            context_mask,
            target_indices,
            ctx_and_target_masks,
        )

    def make_web_dataset(
        self, path: str, split_scene: str, split_noise: str, shuffle: int
    ):
        """Create a WebDataset pipeline for audio processing."""
        dataset = (
            wds.WebDataset(
                path,
                resampled=True,
                nodesplitter=wds.shardlists.split_by_node,
                workersplitter=wds.shardlists.split_by_worker,
                shardshuffle=False,
            )
            .repeat()
            .shuffle(shuffle)
            .decode(wds.torch_audio, handler=wds.warn_and_continue)
            .to_tuple("flac")
            .map(self._augment_sample)
            .batched(self.batch_size)
        )
        return dataset

    def setup(self, stage: str):
        """Set up datasets for training."""
        if stage == "fit":
            self.audio_train = self.make_web_dataset(
                self.datapath, "train", "tr", shuffle=1000
            )

    def train_dataloader(self):
        """Return the training DataLoader."""
        loader = DataLoader(
            self.audio_train,
            batch_size=None,
            pin_memory=True,
            num_workers=16,
            prefetch_factor=2,
        )
        return loader
