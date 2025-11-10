import glob
from random import randrange
from typing import List

import torch
import torchaudio


class NoiseIterator:
    """
    Iterator for WHAMR Noise dataset.
    """

    def __init__(self, noise_dir: str):
        self.noise_files: List[str] = glob.glob(f"{noise_dir}/*.wav")
        self.max_len: int = len(self.noise_files)

    def __iter__(self):
        """
        Initialize the iterator.

        Returns
        -------
        RIRIterator
            The iterator instance itself.
        """
        self.index: int = randrange(self.max_len)
        return self

    def __next__(self) -> torch.Tensor:
        """
        Retrieve the next random RIR entry from the JSON data.

        Returns
        -------
        tensor containing the Noise data.
        Raises
        ------
        StopIteration
            This iterator does not stop; it cycles through random indices indefinitely.
        """
        data, sr = torchaudio.load(self.noise_files[self.index])
        self.index: int = randrange(self.max_len)
        return data, sr
