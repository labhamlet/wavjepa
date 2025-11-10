# Generates scenes from the dataset.
# Convolves the source audio with the source RIR
# Convolves the noise audio with the noise RIRs


import torch
import torchaudio


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

    # Otherwise perform the convolution with vmap.
    def inner(waveform, rir):
        x = []
        for i in range(rir.shape[0]):
            x.append(torchaudio.functional.fftconvolve(waveform, rir[i], mode="full"))
        return torch.stack(x)

    convolve = torch.vmap(inner)
    convolved = convolve(waveform, rir)
    # Always cut to the length of the input...
    return convolved[..., : waveform.shape[-1]]


def add_noise(
    waveform: torch.Tensor, noise: torch.Tensor, snr: torch.Tensor
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

    if not (waveform.ndim - 1 == noise.ndim - 1 == snr.ndim):
        raise ValueError("Input leading dimensions don't match.")

    L = waveform.size(-1)

    if noise.size(-1) != L:
        raise ValueError(
            f"Length dimensions of waveform and noise don't match (got {L} and {noise.size(-1)})."
        )

    energy_signal = torch.linalg.vector_norm(waveform, ord=2, dim=-1) ** 2  # (*,)
    energy_noise = torch.linalg.vector_norm(noise, ord=2, dim=-1) ** 2  # (*,)
    original_snr_db = 10 * (torch.log10(energy_signal) - torch.log10(energy_noise))
    scale = 10 ** ((original_snr_db - snr) / 20.0)  # (*,)

    # scale noise
    scaled_noise = scale.unsqueeze(-1) * noise  # (*, 1) * (*, L) = (*, L)

    return waveform + scaled_noise  # (*, L)


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
    agg_noise = torch.zeros((B, in_channels, seq_len), device=noise_source.device)
    # Add noise sources to aggregare the noise
    # Here we are iterating over the generated sound scenes's noise RIRs
    for i in range(noise_rirs.shape[1]):
        convolved_noise = convolve_with_rir(
            noise_source, noise_rirs[:, i, :, :]
        )  # B, in_channels, seq_len
        agg_noise += convolved_noise
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
    # Noise is already faded!
    convolved_source = convolve_with_rir(audio_source, source_rir)
    agg_noise = aggregate_noise(noise_rirs, noise_source)
    # Cut the agg_noise to the length of the source audio if it is larger!
    agg_noise = agg_noise[:, :, :input_length]
    return convolved_source, agg_noise


def generate_scene(source_rir, noise_rirs, source, noise, snr, sr):
    # Case 1: Both source RIR and noise exist
    if source_rir[0] is not None and noise[0] is not None:
        source, noise = process_audio(
            source_rir, noise_rirs, audio_source=source, noise_source=noise, sr=sr
        )
        return add_noise(source, noise, snr)

    # Case 2: Only source RIR exists (no noise)
    elif source_rir[0] is not None and noise[0] is None:
        convolved_source = convolve_with_rir(source, source_rir)
        return convolved_source

    # Case 3: Only noise exists (no source RIR)
    elif source_rir[0] is None and noise[0] is not None:
        # Need to decide: add noise to raw source or skip noise?
        snr = snr.squeeze()
        return add_noise(source, noise, snr).unsqueeze(1)  # or just return source

    # Case 4: Neither source RIR nor noise exists, return one channel audio
    else:  # source_rir[0] is None and noise[0] is None
        return source.unsqueeze(1)
