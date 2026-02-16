from torch.utils.data import DataLoader
import pytorch_lightning as pl
import webdataset as wds
import torch 
import multiprocessing as mp 
import queue
import torchaudio 

from .dataset_functions import pre_process

from .scene_module import generate_scenes 

class NoiseDataManager:
    """Manages RIR data loading with multiprocessing in the main process."""
    
    def __init__(self, noise_data_dir: str, buffer_size: int = 500, num_workers: int = 1):
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
        dataset = (wds.WebDataset(self.noise_data_dir,
                                resampled=True,
                                shardshuffle=False)
                    .repeat()
                    .shuffle(shuffle_buffer)
                    .decode("pil")
                    .to_tuple("npy")
                    .map(to_torch))

        loader = iter(torch.utils.data.DataLoader(dataset,
                            num_workers=self.num_workers,
                            prefetch_factor=4,
                            batch_size=None))
        print("Noise Loader is set", flush = True)
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
        dataset = (wds.WebDataset(self.rir_data_dir,
                                resampled=True,
                                shardshuffle=False)
                    .repeat()
                    .shuffle(shuffle_buffer)
                    .decode("pil")
                    .to_tuple("npy")
                    .map(to_torch))

        loader = iter(torch.utils.data.DataLoader(dataset,
                            num_workers=self.num_workers,
                            prefetch_factor=4,
                            batch_size=None))
        print("RIR Loader is set", flush = True)
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



class WebAudioDataModuleDenoiser(pl.LightningDataModule):
    sr : int = 32000 
    in_channels: int = 1
    def __init__(
        self,
        data_dir: str,
        rir_dir: str, 
        noise_dir: str,
        batch_size: int = 32,
        with_noise : bool = False,
        with_rir : bool = False,
        nr_samples_per_audio: int = 16,
        nr_time_points : int = 100,
        cache_size: int = 1000,
        snr_low : float = -5.0,
        snr_high: float = 5.0,
        **kwargs
    ):
        """Initialize the data module with shared noise data."""
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.nr_samples_per_audio = nr_samples_per_audio
        self.cache_size = cache_size
        self.nr_time_points = nr_time_points
        self.snr_low = snr_low 
        self.snr_high = snr_high
        
        self.with_noise = with_noise
        self.with_rir = with_rir
            
        self.with_noise = with_noise
        self.with_rir = with_rir 

        if self.with_noise:
            self.noise_loader = NoiseDataManager(noise_dir).start()

        if self.with_rir:
            self.rir_loader = RIRDataManager(rir_dir).start()
        


    def _augment_sample(self, sample):
        """Augment sample with noise and RIR data."""
        
        audio, audio_sr = sample[0]
        audio = audio[0, :] if audio.ndim > 1 else audio
        self.resampler = torchaudio.transforms.Resample(
                        audio_sr,
                        self.sr,
                        lowpass_filter_width=64,
                        rolloff=0.9475937167399596,
                        resampling_method="sinc_interp_kaiser",
                        dtype=audio.dtype,
                        beta=14.769656459379492,
                    )

        audio = self.resampler(audio) if audio_sr != self.sr else audio
        audio = pre_process(audio, self.sr).squeeze(0)
        # Initialize all variables
        noise = None 
        snr = None 
        source_rir = None

        # If with the rir, load the rir.
        if self.with_rir:
            rirs = next(self.rir_loader)
            source_rir = rirs[0]
        
        # If with the noise, load the noise and fade it w.r.t audio's duration.
        if self.with_noise:
            #SR is always 32000
            noise = next(self.noise_loader)
            noise = generate_scenes.fade_noise(noise, audio, self.sr)

            # If audio is bigger than noise, then place the noise in a random location of the audio
            if audio.shape[-1] > noise.shape[-1]:
                start_idx_noise = torch.randint(0, audio.shape[-1] - noise.shape[-1], (1,))
                new_agg_noise = torch.zeros_like(audio)
                new_agg_noise[start_idx_noise:start_idx_noise + noise.shape[-1]] = noise
                noise = new_agg_noise
            snr = torch.distributions.uniform.Uniform(self.snr_low, self.snr_high).sample().item()
            
        return audio, noise, source_rir, snr
    def make_web_dataset(self, path: str, split_scene: str, split_noise: str, shuffle: int):
        """Create a WebDataset pipeline for audio processing."""
        dataset = (
            wds.WebDataset(
                path,
                resampled=True,
                nodesplitter=wds.shardlists.split_by_node,
                workersplitter=wds.shardlists.split_by_worker,
                shardshuffle=False
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
                self.data_dir, "train", "tr", shuffle=1000
            )

    def train_dataloader(self):
        """Return the training DataLoader."""
        loader = DataLoader(
            self.audio_train,
            batch_size=None,
            pin_memory=True,
            num_workers=16,
            prefetch_factor=2
        )
        return loader
