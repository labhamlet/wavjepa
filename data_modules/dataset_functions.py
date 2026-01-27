import torch
import torch.nn.functional as F
import torchaudio.transforms as T




def pad_or_truncate(feature: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Adjust the length of a feature tensor by padding or truncating.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to be adjusted. Expected shape is `(n_frames, ...)`.
    target_length : int
        The desired length of the feature along the first dimension.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(target_length, ...)`, padded or truncated as needed.

    Notes
    -----
    Padding is applied using zero-padding. Truncation is performed along the first dimension
    by slicing the tensor.
    """
    n_frames = feature.shape[1]
    padding = target_length - n_frames
    if padding > 0:
        pad = torch.nn.ZeroPad1d((0, padding))
        return pad(feature)
    elif padding < 0:
        return feature[:, :target_length]
    return feature



def pad_or_truncate_batch(feature: torch.Tensor, target_length: int) -> torch.Tensor:
    """
    Adjust the length of a feature tensor by padding or truncating.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to be adjusted. Expected shape is `(n_frames, ...)`.
    target_length : int
        The desired length of the feature along the first dimension.

    Returns
    -------
    torch.Tensor
        A tensor of shape `(target_length, ...)`, padded or truncated as needed.

    Notes
    -----
    Padding is applied using zero-padding. Truncation is performed along the first dimension
    by slicing the tensor.
    """
    n_frames = feature.shape[-1]
    padding = target_length - n_frames
    if padding > 0:
        pad = torch.nn.ZeroPad2d((0, padding))
        return pad(feature)
    elif padding < 0:
        return feature[:, :, :target_length]
    return feature


def instance_normalize(feature: torch.Tensor) -> torch.Tensor:
    """
    Normalize a feature tensor using the specified mean and standard deviation.

    Parameters
    ----------
    feature : torch.Tensor
        A tensor containing the feature to normalize.
    mean : float
        The mean value for normalization.
    std : float
        The standard deviation value for normalization.

    Returns
    -------
    torch.Tensor
        A tensor where each element is normalized as:
        `(feature - mean) / (std)`.

    Notes
    -----
    This normalization scales the data to have a mean of 0 and reduces the amplitude
    by the factor of `2 * std`.
    """
    return (feature - feature.mean()) / (feature.std() + 1e-8)


def pre_process_audio(audio, audio_sr, resample_sr):
    resampler = T.Resample(audio_sr, resample_sr, dtype=audio.dtype)
    waveform = audio[0, :] if audio.ndim > 1 else audio
    # Resample the audio with 
    waveform = (
        resampler(waveform)
        if audio_sr != resample_sr
        else waveform
    )
    # Normalize the audio using RMSE
    waveform = normalize_audio(waveform, -14.0)
    waveform = waveform.reshape(1, -1)
    # Make sure audio is 10 seconds
    padding = resample_sr * 10 - waveform.shape[1]
    if padding > 0:
        waveform = F.pad(waveform, (0, padding), "constant", 0)
    elif padding < 0:
        waveform = waveform[:, : resample_sr * 10]
    return waveform[0]


def pre_process_noise(audio, audio_sr, resample_sr):
    resampler = T.Resample(audio_sr, resample_sr, dtype=audio.dtype)
    waveform = audio[0, :] if audio.ndim > 1 else audio
    # Resample the audio
    waveform = (
        resampler(waveform)
        if audio_sr != resample_sr
        else waveform
    )
    # Normalize the audio using RMSE
    waveform = normalize_audio(waveform, -14.0)
    return waveform


def normalize_audio(audio_data, target_dBFS=-14.0):
    rms = torch.sqrt(torch.mean(audio_data**2))  # Calculate the RMS of the audio
    if rms == 0:  # Avoid division by zero in case of a completely silent audio
        return audio_data
    current_dBFS = 20 * torch.log10(rms)  # Convert RMS to dBFS
    gain_dB = target_dBFS - current_dBFS  # Calculate the required gain in dB
    gain_linear = 10 ** (gain_dB / 20)  # Convert gain from dB to linear scale
    normalized_audio = audio_data * gain_linear  # Apply the gain to the audio data
    return normalized_audio

def normalize_audio_batch(audio_data, target_dBFS=-14.0, eps=1e-9):
    """
    Vectorized normalization to a target dBFS.
    audio_data: (Batch, Time)
    """
    rms = torch.sqrt(torch.mean(audio_data**2, dim=-1, keepdim=True) + eps)
    current_dBFS = 20 * torch.log10(rms)
    gain_dB = target_dBFS - current_dBFS
    gain_linear = 10 ** (gain_dB / 20)
    normalized_audio = audio_data * gain_linear
    is_silent = rms <= eps
    normalized_audio = torch.where(
        is_silent, torch.zeros_like(normalized_audio), normalized_audio
    )
    return normalized_audio