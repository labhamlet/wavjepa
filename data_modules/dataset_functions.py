import io
from typing import Any

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Native-rate loader helpers (DESIGN.md section A). These are pure functions so
# they can run inside DataLoader workers without touching the data module.
# ---------------------------------------------------------------------------

#: Extensions handed to ``soundfile`` (libsndfile) when decoding raw shard bytes.
NATIVE_AUDIO_EXTENSIONS: tuple[str, ...] = ("flac", "wav", "ogg")
#: Sample rate assumed for ``.npy`` members (preprocessed int16 16 kHz shards).
NPY_SAMPLE_RATE: int = 16000
#: The wds ``to_tuple`` key spec used by the native pipeline (``;`` = alternatives).
NATIVE_AUDIO_KEYS: str = "flac;wav;ogg;npy"
#: Length (seconds) every clip is padded/cropped to at its native rate.
NATIVE_TARGET_SECONDS: int = 10


def _extension(key: str) -> str:
    """Return the lower-cased extension of a wds member key (``".flac"`` -> ``"flac"``)."""
    return key.rsplit(".", 1)[-1].lower()


def decode_audio_bytes(key: str, data: bytes) -> tuple[np.ndarray, int] | None:
    """Decode the raw bytes of one tar member into ``(audio float32 (L,), sr)``.

    Intended as a ``webdataset`` decoder handler: wds calls it as
    ``handler(key, data)`` with ``key`` being the member extension (``".flac"``).

    * ``flac`` / ``wav`` / ``ogg``: decoded with ``soundfile`` from a ``BytesIO``
      (no temp file), channel 0 only, at the file's native sample rate.
    * ``npy``: a ``numpy`` array; ``int16`` is scaled by ``1/32768`` (other
      integer dtypes by their full scale), floats are cast to float32. The
      sample rate is assumed to be :data:`NPY_SAMPLE_RATE`.
    * anything else: ``None`` so that wds falls through to its default handlers.
    """
    ext = _extension(key)
    if ext in NATIVE_AUDIO_EXTENSIONS:
        audio, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
        return np.ascontiguousarray(audio[:, 0]), int(sr)
    if ext == "npy":
        arr = np.load(io.BytesIO(data), allow_pickle=False)
        if arr.ndim == 2:
            # (C, L) if the first axis is the small one, else (L, C); take channel 0.
            arr = arr[0] if arr.shape[0] <= arr.shape[1] else arr[:, 0]
        arr = arr.reshape(-1)
        if arr.dtype == np.int16:
            audio = arr.astype(np.float32) / 32768.0
        elif np.issubdtype(arr.dtype, np.integer):
            audio = arr.astype(np.float32) / float(np.iinfo(arr.dtype).max + 1)
        else:
            audio = arr.astype(np.float32, copy=False)
        return np.ascontiguousarray(audio), NPY_SAMPLE_RATE
    return None


def pad_or_crop_to(x: torch.Tensor | np.ndarray, n: int) -> torch.Tensor | np.ndarray:
    """Zero-pad or crop the last axis of ``x`` to exactly ``n`` samples.

    Works for ``torch.Tensor`` and ``np.ndarray`` and returns the same type.
    Returns ``x`` itself when no change is needed.
    """
    length = int(x.shape[-1])
    if length == n:
        return x
    if length > n:
        return x[..., :n]
    if isinstance(x, torch.Tensor):
        return F.pad(x, (0, n - length))
    pad_width = [(0, 0)] * (x.ndim - 1) + [(0, n - length)]
    return np.pad(x, pad_width)


def prepare_native_sample(
    audio: np.ndarray | torch.Tensor,
    sr: int,
    target_seconds: int = NATIVE_TARGET_SECONDS,
) -> dict[str, Any]:
    """Turn one decoded clip into the per-sample dict consumed by :func:`collate_native`.

    Returns ``{"audio": float32 tensor (sr*target_seconds,), "sr": int, "length": int}``
    where ``length`` is the number of valid (non-padded) samples, i.e.
    ``min(len(audio), sr*target_seconds)``.
    """
    n = int(sr * target_seconds)
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    else:
        audio = audio.to(torch.float32)
    audio = audio.reshape(-1)
    length = min(int(audio.shape[-1]), n)
    audio = pad_or_crop_to(audio, n)
    return {"audio": audio, "sr": int(sr), "length": length}


def collate_native(samples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Collate native-rate samples into the loader -> model batch contract.

    Returns::

        {"audio":  float32 (B, L_max)   zero-padded to the longest row,
         "sr":     int64   (B,)         native sample rate per row,
         "length": int64   (B,)         valid samples per row}
    """
    batch_size = len(samples)
    l_max = max(int(s["audio"].shape[-1]) for s in samples)
    audio = torch.zeros(batch_size, l_max, dtype=torch.float32)
    for i, s in enumerate(samples):
        row = s["audio"]
        audio[i, : row.shape[-1]] = row
    sr = torch.tensor([int(s["sr"]) for s in samples], dtype=torch.int64)
    length = torch.tensor([int(s["length"]) for s in samples], dtype=torch.int64)
    return {"audio": audio, "sr": sr, "length": length}


# ---------------------------------------------------------------------------
# Legacy helpers (still used by the legacy loader path, the denoiser and the model)
# ---------------------------------------------------------------------------

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

def normalize_audio(audio_data, target_dBFS=-14.0):
    rms = torch.sqrt(torch.mean(audio_data**2))  # Calculate the RMS of the audio
    if rms == 0:  # Avoid division by zero in case of a completely silent audio
        return audio_data
    current_dBFS = 20 * torch.log10(rms)  # Convert RMS to dBFS
    gain_dB = target_dBFS - current_dBFS  # Calculate the required gain in dB
    gain_linear = 10 ** (gain_dB / 20)  # Convert gain from dB to linear scale
    normalized_audio = audio_data * gain_linear  # Apply the gain to the audio data
    return normalized_audio

def pre_process(waveform, sr):
    # Normalize the audio using RMSE
    waveform = normalize_audio(waveform, -14.0)
    #Add a channel dimension
    waveform = waveform.reshape(1, -1)
    # Make sure audio is 10 seconds
    padding = sr * 10 - waveform.shape[1]
    if padding > 0:
        waveform = F.pad(waveform, (0, padding), "constant", 0)
    elif padding < 0:
        waveform = waveform[:, : sr * 10]
    return waveform

def pre_process_noise(waveform):
    # Normalize the audio using RMSE
    waveform = normalize_audio(waveform, -14.0)
    return waveform
