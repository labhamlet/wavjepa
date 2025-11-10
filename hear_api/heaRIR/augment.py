from typing import Optional

import torch

from .iterators import SceneIterator
from .scene_module import generate_scene


class Augmenter:
    def __init__(
        self, spatial_scene_iter: Optional[SceneIterator], sr: int, snr: Optional[int]
    ):
        self.spatial_scene_iter = spatial_scene_iter
        self.sr = sr
        self.snr = snr

    def augment(
        self, audio: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Augment audio with spatial scene processing.

        This method applies room impulse responses (RIRs) to the audio and noise
        signals if a spatial scene iterator is available.

        Args:
            audio: Input audio tensor, should be normalized
            noise: Optional noise tensor, should be normalized if provided

        Returns:
            Processed audio tensor with consistent length
        """
        # Store original input length to ensure consistent output size
        input_audio_length = audio.shape[-1]
        if self.spatial_scene_iter:
            source_rir, noise_rirs, _ = next(self.spatial_scene_iter)
            source_rir = source_rir.to(audio.device)
            # Inplace processing. Send noise to device.
            noise_rirs = [noise_rir.to(audio.device) for noise_rir in noise_rirs]

            # Pad audio if RIR is longer than input audio
            if source_rir.shape[-1] > input_audio_length:
                padding_right = source_rir.shape[-1] - input_audio_length
                audio = torch.nn.functional.pad(
                    audio, (0, padding_right), value=0, mode="constant"
                )

            # Generate spatial scene with or without noise
            audio = generate_scene(
                source_rir=source_rir,
                noise_rirs=[] if noise is None else noise_rirs,
                source=audio,
                noise=noise,
                snr=self.snr,
                sr=self.sr,
            )
        # Ensure audio has channel dimension
        if audio.ndim == 1:
            audio = torch.unsqueeze(audio, 0)

        # Truncate to original input length to ensure consistent output size
        audio = audio[:, :input_audio_length]
        return audio
