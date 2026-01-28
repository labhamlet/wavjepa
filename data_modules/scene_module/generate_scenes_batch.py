# Generates scenes from the dataset.
# Convolves the source audio with the source RIR
# Convolves the noise audio with the noise RIRs

from typing import List

import torch
import torchaudio
import torchaudio.functional as F


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

    assert waveform.shape[0] == rir.shape[0], "Not compatible for this operation"
    
    #Otherwise perform the convolution with vmap.
    def inner(waveform, rir):
        x = []
        for i in range(rir.shape[0]):
            x.append(torchaudio.functional.fftconvolve(waveform, rir[i], mode="full"))
        return torch.stack(x)
    
    convolve = torch.vmap(inner)
    convolved = convolve(waveform, rir)
    # Always cut to the length of the input...
    return convolved[..., : waveform.shape[-1]]


def aggregate_noise(noise_rirs, noise_source):
    """Aggregate the multiple noise sources into one waveform.
    this creates a naturalistic scene where multiple noise sources are in. 
    Arguments
    ---------
    noise_rirs : List[torch.Tensor]
        Multiple noise RIRs retrieved from the scene specification 
    noise_source : torch.Tensor 
        The noise sample from WHAMR! dataset
    
    Returns
    --------
    torch.Tensor with multiple noise sources aggregated.

    """
    in_channels = noise_rirs.shape[2]
    B, seq_len = noise_source.shape
    agg_noise = torch.zeros((B, in_channels, seq_len), device = noise_source.device)
    # Add noise sources to aggregare the noise
    # Here we are iterating over the generated sound scenes's noise RIRs
    for i in range(noise_rirs.shape[1]):
        convolved_noise = convolve_with_rir(noise_source, noise_rirs[:, i, :, :]) # B, in_channels, seq_len
        agg_noise += convolved_noise
    return agg_noise


def process_audio(source_rir : torch.Tensor, 
    noise_rirs: List[torch.Tensor], 
    audio_source: torch.Tensor, 
    noise_source: torch.Tensor):
    """Facade function for processing the audio and noise sources with their corresponding RIRs
    Arguments
    ---------
    source_rir : torch.Tensor
        The source RIR that audio_source will be convolved with
    noise_rirs : List[torch.Tensor]
        The noise RIRs that noise_source will be convolved with
    audio_source : torch.Tensor 
        The audio source from AudioSet 
    noise_source : torch.Tensor 
        The noise source from WHAMR!
    
    Raises
    -------
    AssertionError if there are no source_rirs or no noise_rirs.
    
    Returns
    --------
    The generated scene as torch.Tensor


    """
    assert source_rir is not None, "No source RIR is provided"
    assert len(noise_rirs) > 0, "No noise RIRs are provided"

    input_length = audio_source.shape[-1]
    # Noise is already faded!
    convolved_source = convolve_with_rir(audio_source, source_rir)
    agg_noise = aggregate_noise(noise_rirs, noise_source)
    # Cut the agg_noise to the length of the source audio if it is larger!
    agg_noise = agg_noise[:, :, :input_length]
    return convolved_source, agg_noise


def generate_scene(source_rir, source, noise, snr):
    # Case 1: Both source RIR and noise exist
    if source_rir[0] is not None and noise[0] is not None:
        assert noise.ndim == 2
        assert source.ndim == 2
        assert snr.ndim == 1, f"Got snr dim {snr.shape}"
        source = F.add_noise(source, noise, snr)
        # We get the first channel of the ambisonics RIRs
        convolved_source = convolve_with_rir(source, source_rir[:, [0], :])
        return convolved_source

    # Case 2: Only source RIR exists (no noise)
    elif source_rir[0] is not None and noise[0] is None:
        # Get the firt
        assert source.ndim == 2
        convolved_source = convolve_with_rir(source, source_rir[:, [0], :])
        return convolved_source

    # Case 3: Only noise exists (no source RIR)
    elif source_rir[0] is None and noise[0] is not None:
        # Add channel dim
        assert source.ndim == 2
        assert noise.ndim == 2

        if snr.ndim != 2:
            snr = snr.unsqueeze(1)
        return F.add_noise(source, noise, snr)

    # Case 4: Neither source RIR nor noise exists, return one channel audio
    else:  # source_rir[0] is None and noise[0] is None
        return source
