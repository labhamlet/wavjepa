import os
from functools import partial

import pytorch_lightning as pl
import webdataset as wds

from .dataset_functions import _getitem_audioset
from .iterators import NoiseIterator, SceneIterator

os.environ["WDS_VERBOSE_CACHE"] = "1"
os.environ["GOPEN_VERBOSE"] = "0"


class WebAudioDataModuleBatched(pl.LightningDataModule):
    """
    A PyTorch Lightning DataModule for audio data processing with WebDataset support.
    This dataloader is not responsible for RIR convolutions as we can not use WebDataset and CUDA at the same time without the "spawn"
    This dataloader is only responsible for reading the audio, noise, and scene files and returning them with the specified SR.
    We assume that resampling does not bottleneck the dataloader if it is done on CPU, and thus we do not need to port in on GPU.

    This class handles loading and preprocessing of audio data for training deep learning
    models. It supports room impulse response (RIR) augmentation, noise mixing, and
    configurable preprocessing for audio spectrograms.

    Parameters
    ----------
    base_data_dir : str
        Directory path to the base training data in WebDataset format.
    val_data_dir : str
        Directory path to the validation data in WebDataset format.
    base_rir_dir : str
        Base directory containing room impulse response (RIR) data.
    rir_data_dir : str
        Specific directory path within base_rir_dir for RIR data access.
    base_noise_dir : str
        Directory containing noise samples for audio augmentation.
    load_scenes_with_noise : bool
        Whether to load acoustic scenes with added noise.
    num_mel_bins : int
        Number of mel frequency bins for spectrogram transformation.
    target_length : int
        Target length of processed audio segments.
    input_length : int
        Input length of raw audio segments.
    batch_size : int, optional
        Batch size for dataloaders, default is 32.
    nr_samples_per_audio : int, optional
        Number of samples to extract from each audio file, default is 32.
    sr : int, optional
        Sample rate for audio processing, default is 32000.
    ambisonic : bool, optional
        Whether to use ambisonic audio format, default is False.

    Attributes
    ----------
    datapath : str
        Path to training data directory.
    val_path : str
        Path to validation data directory.
    rir_dir : str
        Path to RIR base directory.
    noise_dir : str
        Path to noise samples directory.
    melbins : int
        Number of mel frequency bins.
    target_length : int
        Target length for processed audio.
    input_length : int
        Input length for raw audio.
    batch_size : int
        Batch size for dataloaders.
    load_scenes_with_noise : bool
        Flag for loading scenes with noise.
    nr_samples_per_audio : int
        Number of samples per audio file.
    rir_data_dir : str
        Directory for RIR data.
    sr : int
        Audio sample rate.
    ambisonic : bool
        Flag for ambisonic audio format.
    audio_train : WebDataset
        Training dataset (available after setup).
    audio_val : WebDataset
        Validation dataset (available after setup).
    """

    def __init__(
        self,
        base_data_dir: str,
        val_data_dir: str,
        base_rir_dir: str,
        rir_data_dir: str,
        base_noise_dir: str,
        load_scenes_with_noise: bool,
        batch_size: int = 32,
        sr: int = 32000,
        ambisonic: bool = False,
        masker=None,
        nr_samples_per_audio: int = 16,
        **kwargs,
    ):
        super().__init__()
        self.datapath = base_data_dir
        self.val_path = val_data_dir
        self.rir_dir = base_rir_dir
        self.noise_dir = base_noise_dir
        self.batch_size = batch_size
        self.load_scenes_with_noise = load_scenes_with_noise
        self.rir_data_dir = rir_data_dir
        self.sr = sr
        self.ambisonic = ambisonic
        self.masker = masker
        self.nr_samples_per_audio = nr_samples_per_audio

    def make_web_dataset(
        self, path: str, split_scene: str, split_noise: str, shuffle: int
    ) -> wds.WebDataset:
        """
        Create a WebDataset pipeline for audio processing.

        This method sets up a data processing pipeline that loads audio files,
        applies room impulse responses, adds noise, and prepares batches for training.

        Parameters
        ----------
        path : str
            Path to the WebDataset shards.
        split_scene : str
            Scene split name ('train', 'val', etc.) for RIR selection.
        split_noise : str
            Noise split name ('tr', 'cv', etc.) for noise sample selection.
        shuffle : int
            Number of samples to buffer for shuffling. 0 means no shuffling.

        Returns
        -------
        WebDataset
            Configured WebDataset pipeline for audio processing.
        """
        random_scene_iter = iter(
            SceneIterator(
                scenes=os.path.join(self.rir_dir, split_scene),
                with_noise=self.load_scenes_with_noise,
                rir_data_dir=self.rir_data_dir,
                ambisonic=self.ambisonic,
                sr=self.sr,
            )
        )
        random_noise_iter = iter(
            NoiseIterator(noise_dir=os.path.join(self.noise_dir, split_noise))
        )

        batched_dataloader = partial(
            _getitem_audioset,
            random_scene_generator=random_scene_iter,
            random_noise_generator=random_noise_iter,
            resample_sr=self.sr,
            masker=self.masker,
            nr_samples_per_audio=self.nr_samples_per_audio,
        )

        # Create the WebDataset pipeline
        dataset = (
            wds.WebDataset(path, detshuffle=True)
            .shuffle(shuffle)
            .decode(wds.torch_audio, handler=wds.warn_and_continue)
            .to_tuple("flac")
            .map(
                batched_dataloader()
            )  # Need to call the function to invoke the clousere
            .repeat()
            .batched(self.batch_size)
        )
        return dataset

    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            # Change these later!
            self.audio_train = self.make_web_dataset(
                self.datapath, "test", "tr", shuffle=1000
            )
            self.audio_val = self.make_web_dataset(
                self.val_path, "test", "cv", shuffle=0
            )

    def train_dataloader(self):
        loader = wds.WebLoader(
            self.audio_train,
            batch_size=None,
            num_workers=8,
            prefetch_factor=4,  # What about a huge prefetch?
        )
        return loader
