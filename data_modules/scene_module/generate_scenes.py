# Generates scenes from the dataset.
# Convolves the source audio with the source RIR
# Convolves the noise audio with the noise RIRs

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio


def apply_fadein(audio: torch.Tensor, sr: int, duration: float = 0.20) -> torch.Tensor:
    """Apply fade-in to the audio source.

    Arguments
    ----------
    audio : torch.Tensor
        The audio that we want to fade-in
    sr : int 
        Sampling rate of the audio
    duration : float
        Duration of the fade-in 
    
    Returns
    --------
    torch.Tensor with faded-in audio
    """
    # convert to audio indices (samples)
    end = int(duration * sr)
    start = 0
    # compute fade in curve
    # linear fade
    fade_curve = torch.linspace(0.0, 1.0, end, device = audio.device)
    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve
    return audio


def apply_fadeout(audio: torch.Tensor, sr: int, duration: float = 0.20) -> torch.Tensor:
    """Apply fade-out to the audio source.

    Arguments
    ----------
    audio : torch.Tensor
        The audio that we want to fade-out
    sr : int 
        Sampling rate of the audio
    duration : float
        Duration of the fade-out

    Returns
    --------
    torch.Tensor with faded-out audio
    """
    # convert to audio indices (samples)
    length = int(duration * sr)
    end = audio.shape[0]
    start = end - length
    # compute fade out curve
    # linear fade
    fade_curve = torch.linspace(1.0, 0.0, length, device=audio.device)
    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve
    return audio


def load_rir(path: str) -> Optional[torch.Tensor]:
    """Loads the RIR from specified path. 

    Arguments
    ----------
    path : str
        The path that we want to load RIR from 
    
    Raises
    ---------
    AssertionError
        If the path does not exist.

    Returns
    --------
    torch.Tensor with the loaded RIR, raises exception if it can't
    """
    assert Path(path).exists(), "Path {path} does not exist"
    try:
        rir = torch.tensor(np.load(path))
        return rir
    except Exception as e:
        print(f"Error loading RIR file: {e}")


def convolve_with_rir(waveform: torch.Tensor, rir: torch.Tensor) -> torch.Tensor:
    """Convolve the waveform with the specified RIR 

    Arguments
    ---------
    waveform : torch.Tensor 
        The waveform that represent the audio 
    rir : torch.Tensor 
        The rir that we want to apply 
    
    Raises
    -------
    AssertionError
        If the audio is not mono, and has an additional dummy channel raise an error
    
    Returns
    --------
    Convolved audio with the RIR. The returned audio has the same shape as the input waveform.
    """
    assert waveform.ndim == 1, (
        "No Stero sounds are accepted, cast the sound to mono or collables the first dimension!"
    )
    if rir.ndim == 1:
        rir = rir.unsqueeze(0)

    # Because we are using earlier version of the torch audio we need to do it this way.
    x = [
        torchaudio.functional.fftconvolve(waveform, rir[i], mode="full")
        for i in range(rir.shape[0])
    ]
    convolved = torch.stack(x)
    # Always cut to the length of the input...
    if convolved.shape[0] == 1:
        # Return mono sound.
        return convolved[..., : waveform.shape[-1]]
    else:
        return convolved[..., : waveform.shape[-1]]


def fade_noise(noise_source: torch.Tensor, audio_source: torch.Tensor, sr: int):
    """Facade function to determine what kind of fade-in and fade-out we should apply to the noise
    Arguments
    ---------
    noise_source: torch.Tensor 
        The noise waveform 
    audio_source: torch.Tensor 
        The audio waveform 
    sr : int 
        The sampling rate for both audio_source and noise_source
    """

    if noise_source.shape[-1] > audio_source.shape[-1]:
        # If noise is longer than the audio, pick a random segment
        # Because we cut the noise like that, apply a fadeout!
        start_idx_noise = torch.randint(0, noise_source.shape[-1] - audio_source.shape[-1], (1,)).item()
        noise_source = noise_source[start_idx_noise:start_idx_noise + audio_source.shape[-1]]
        noise_source = apply_fadeout(noise_source, sr=sr, duration=0.2)
    # Otherwise apply fade-in and fade-out
    else:
        noise_source = apply_fadein(noise_source, sr=sr, duration=0.2)
        noise_source = apply_fadeout(noise_source, sr=sr, duration=0.2)
    return noise_source

