import glob
import json
import os
import threading
from random import randrange

import numpy as np
import torch
import torch.nn.functional as F


def preprocess_rirs(reverb, sr):
    """Pad the RIRs to be the same shape.
    Arguments
    ----------
    reverb : nd.array
        The RIR itself as torch.tensor
    sr : int
        The sampling rate of the RIR
    """
    reverb_padding = sr * 2 - reverb.shape[1]
    if reverb_padding > 0:
        reverb = F.pad(reverb, (0, reverb_padding), "constant", 0)
    elif reverb_padding < 0:
        reverb = reverb[:, : sr * 2]
    return reverb


class SceneIterator:
    """
    Thread-safe iterator for RIR Scenes Dataset.
    Attributes
    ----------
    rir_data_dir : str
        The directory that contains the generated RIRs.
    scenes : str
        The directory that contains the metadata for generated scenes.
    with_noise : bool
        Indicate if we want to load the scene with noise.
    ambisonic: bool
        Indicate if we want to load ambisonic RIRs for the scene
    sr : int
        Sampling rate for the whole scene.
    """

    def __init__(
        self,
        rir_data_dir: str,
        scenes: str,
        with_noise: bool = True,
        ambisonic: bool = False,
        sr: int = 32000,
        max_noise_sources=5,
    ):
        print(scenes, flush=True)
        scenes_json: list[str] = glob.glob(f"{scenes}/*.json")
        self.scenes: list[dict[str, dict]] = []
        self.max_len = 0
        self.rir_data_dir = rir_data_dir
        for scene in scenes_json:
            with open(scene) as f:
                data = json.load(f)
                self.scenes.extend(data["sampled_regions"])
                self.max_len += len(data["sampled_regions"])
        self.with_noise = with_noise
        # Add a lock for thread safety
        self._lock = threading.RLock()
        self.ambisonic = ambisonic
        self.sr = sr
        self.max_noise_sources = max_noise_sources

    def __iter__(self):
        """
        Initialize the iterator.

        Returns
        -------
        SceneIterator
            The iterator instance itself.
        """
        # No need to set an index here - it will be set in __next__
        return self

    def __next__(self):
        """
        Retrieve the next random scene from the JSON data in a thread-safe manner.
        This iterator does not stop; it cycles through random indices indefinitely.

        Returns
        -------
        waveform, Dict[waveform], location of source

        Raises
        ------
        StopIteration
        """
        with self._lock:
            # Select a random index for each call to __next__
            index = randrange(self.max_len)
            selected_scene = self.scenes[index]
            if self.ambisonic:
                source_rir_path = selected_scene["region"]["scene"]["source"]["rir"][
                    "ambisonic_rir_path"
                ]
            else:
                source_rir_path = selected_scene["region"]["scene"]["source"]["rir"][
                    "binaural_rir_path"
                ]
            source_az = selected_scene["region"]["scene"]["source"]["azimuth"]
            source_el = selected_scene["region"]["scene"]["source"]["elevation"]
            # Load the source RIR
            source_rir_file = os.path.join(
                self.rir_data_dir, os.path.basename(source_rir_path)
            )
            source_rir = torch.tensor(np.load(source_rir_file)).float()
            source_rir = preprocess_rirs(source_rir, self.sr)

            noise_rirs = torch.zeros(
                (self.max_noise_sources, 4 if self.ambisonic else 2, self.sr * 2)
            )
            if self.with_noise:
                for i, noise in enumerate(selected_scene["region"]["scene"]["noise"]):
                    if self.ambisonic:
                        noise_rir_path = noise["rir"]["ambisonic_rir_path"]
                    else:
                        noise_rir_path = noise["rir"]["binaural_rir_path"]
                    noise_rir_file = os.path.join(
                        self.rir_data_dir, os.path.basename(noise_rir_path)
                    )
                    noise_rir = torch.tensor(np.load(noise_rir_file)).float()
                    noise_rir = preprocess_rirs(noise_rir, self.sr)
                    noise_rirs[i] = noise_rir

        return source_rir, noise_rirs, [source_az, source_el]
