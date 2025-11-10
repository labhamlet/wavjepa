# Generates scenes from the dataset.
# Convolves the source audio with the source RIR
# Convolves the noise audio with the noise RIRs

from pathlib import Path

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
    fade_curve = torch.linspace(0.0, 1.0, end)
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
    fade_curve = torch.linspace(1.0, 0.0, length)
    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve
    return audio


def load_rir(path: str) -> torch.Tensor | None:
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


def add_noise(
    waveform: torch.Tensor,
    noise: torch.Tensor,
    snr: torch.Tensor,
    lengths: torch.Tensor | None = None,
) -> torch.Tensor:
    r"""Taken from torchaudio source code.

    Scales and adds noise to waveform per signal-to-noise ratio.

    Specifically, for each pair of waveform vector :math:`x \in \mathbb{R}^L` and noise vector
    :math:`n \in \mathbb{R}^L`, the function computes output :math:`y` as

    .. math::
        y = x + a n \, \text{,}

    where

    .. math::
        a = \sqrt{ \frac{ ||x||_{2}^{2} }{ ||n||_{2}^{2} } \cdot 10^{-\frac{\text{SNR}}{10}} } \, \text{,}

    with :math:`\text{SNR}` being the desired signal-to-noise ratio between :math:`x` and :math:`n`, in dB.

    Note that this function broadcasts singleton leading dimensions in its inputs in a manner that is
    consistent with the above formulae and PyTorch's broadcasting semantics.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        waveform (torch.Tensor): Input waveform, with shape `(..., L)`.
        noise (torch.Tensor): Noise, with shape `(..., L)` (same shape as ``waveform``).
        snr (torch.Tensor): Signal-to-noise ratios in dB, with shape `(...,)`.
        lengths (torch.Tensor or None, optional): Valid lengths of signals in ``waveform`` and ``noise``, with shape
            `(...,)` (leading dimensions must match those of ``waveform``). If ``None``, all elements in ``waveform``
            and ``noise`` are treated as valid. (Default: ``None``)

    Returns:
        torch.Tensor: Result of scaling and adding ``noise`` to ``waveform``, with shape `(..., L)`
        (same shape as ``waveform``).
    """

    if not (
        waveform.ndim - 1 == noise.ndim - 1 == snr.ndim
        and (lengths is None or lengths.ndim == snr.ndim)
    ):
        raise ValueError("Input leading dimensions don't match.")

    L = waveform.size(-1)

    if noise.size(-1) != L:
        raise ValueError(
            f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)})."
        )

    # compute scale
    if lengths is not None:
        mask = torch.arange(0, L, device=lengths.device).expand(
            waveform.shape
        ) < lengths.unsqueeze(-1)  # (*, L) < (*, 1) = (*, L)
        masked_waveform = waveform * mask
        masked_noise = noise * mask
    else:
        masked_waveform = waveform
        masked_noise = noise

    energy_signal = (
        torch.linalg.vector_norm(masked_waveform, ord=2, dim=-1) ** 2
    )  # (*,)
    energy_noise = torch.linalg.vector_norm(masked_noise, ord=2, dim=-1) ** 2  # (*,)
    original_snr_db = 10 * (torch.log10(energy_signal) - torch.log10(energy_noise))
    scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)

    # scale noise
    scaled_noise = scale.unsqueeze(-1) * noise  # (*, 1) * (*, L) = (*, L)

    return waveform + scaled_noise  # (*, L)


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
        # If audio is longer than the noise, just cut the noise to the audio length
        # Because we cut the noise like that, apply a fadeout!
        noise_source = noise_source[: audio_source.shape[-1]]
        noise_source = apply_fadeout(noise_source, sr=sr, duration=0.2)
    # Otherwise apply fade-in and fade-out
    else:
        noise_source = apply_fadein(noise_source, sr=sr, duration=0.2)
        noise_source = apply_fadeout(noise_source, sr=sr, duration=0.2)
    return noise_source


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
    noise = []
    for noise_rir in noise_rirs:
        convolved_audio = convolve_with_rir(noise_source, noise_rir)
        noise.append(convolved_audio)
    max_len = max([x.shape[-1] for x in noise])
    agg_noise = torch.stack(
        [
            torch.nn.functional.pad(x, (0, max_len - x.shape[-1]), "constant", value=0)
            for x in noise
        ]
    ).sum(axis=0)
    return agg_noise


def process_audio(
    source_rir: torch.Tensor,
    noise_rirs: list[torch.Tensor],
    audio_source: torch.Tensor,
    noise_source: torch.Tensor,
    sr: int,
):
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
    noise_source = fade_noise(noise_source, audio_source, sr)
    convolved_source = convolve_with_rir(audio_source, source_rir)
    agg_noise = aggregate_noise(noise_rirs, noise_source)
    # Cut the agg_noise to the length of the source audio if it is larger!
    agg_noise = agg_noise[:, :input_length]
    # If audio is bigger than noise, then pad to 0 starting from a random index and pad 0 to the end.
    # Corresponds to adding a noise at random index.
    if convolved_source.shape[1] > agg_noise.shape[1]:
        start_idx_noise = np.random.randint(0, input_length - agg_noise.shape[1])
        new_agg_noise = torch.zeros_like(convolved_source)
        new_agg_noise[:, start_idx_noise : start_idx_noise + agg_noise.shape[1]] = (
            agg_noise
        )
        return convolved_source, new_agg_noise

    return convolved_source, agg_noise


def generate_scene(source_rir, noise_rirs, source, noise, snr, sr):
    if len(noise_rirs) > 0:
        source, noise = process_audio(
            source_rir, noise_rirs, audio_source=source, noise_source=noise, sr=sr
        )
        if not torch.is_tensor(snr):
            snr = torch.tensor([snr])
        return add_noise(source, noise, snr)
    else:
        convolved_source = convolve_with_rir(source, source_rir)
        return convolved_source
