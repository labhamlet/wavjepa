import glob
from random import randrange

import torchaudio


class NoiseIterator:
    """Iterator for WHAMR Noise dataset.
    Gets random noise samples for the specified noise_dir and iterates over them infinitely.
    Attributes
    ----------
    noise_dir : str
        The directory that contains the WHAMR! noise files.
    """

    def __init__(self, noise_dir: str):
        self.noise_files: list[str] = glob.glob(f"{noise_dir}/*.wav")
        self.max_len: int = len(self.noise_files)

    def __iter__(self):
        """Initialize the iterator.
        Returns
        -------
        NoiseIterator
            The iterator instance itself.
        """
        self.index: int = randrange(self.max_len)
        return self

    def __next__(self):
        """Retrieve the next random sound from WHAMR! data.
        This iterator does not stop; it cycles through random indices indefinitely.
        Returns
        -------
        data holding the noise instance, and the sr for the noise recording.
        """

        data, sr = torchaudio.load(self.noise_files[self.index])
        self.index: int = randrange(self.max_len)
        return data, sr
