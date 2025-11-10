import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as T
from einops import repeat

from .scene_module import generate_scenes


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


def _getitem_audioset(
    random_scene_generator,
    random_noise_generator,
    resample_sr,
    nr_samples_per_audio,
    masker,
):
    """
    Process a single audio sample for training with spatial audio.

    This function applies room impulse responses (RIRs), adds noise at
    a random SNR level, generates mel spectrograms, and extracts multiple random
    segments from each spectrogram for training.

    Parameters
    ----------
    sample : tuple
        A tuple containing audio data and its sample rate: (audio_data, sample_rate).
    random_scene_generator : iterator
        An iterator that provides random room scenes with source and noise RIRs.
        Expected to yield (source_rir, noise_rirs, source_rir_location) tuples.
    random_noise_generator : iterator
        An iterator that provides random noise samples.
        Expected to yield (noise_data, noise_sample_rate) tuples.
    target_length : int
        The target length (in ms) for the output spectrogram segments.
    input_length : int
        The input length (in samples) for audio processing.
    num_mel_bins : int
        Number of mel frequency bins for the spectrogram.
    resample_sr : int
        Target sample rate to resample audio if needed.
    nr_samples_per_audio : int, optional
        Number of spectrogram segments to extract from each audio sample, default is 1.

    Returns
    -------
    tuple
        A tuple containing:
        - return_fbank : torch.Tensor
            Mel spectrogram segments of shape (nr_samples_per_audio, n_channels, target_length, num_mel_bins)
        - target : torch.Tensor
            Source location coordinates repeated for each segment, shape (nr_samples_per_audio, 2)

    Raises
    ------
    AssertionError
        If target_length is greater than input_length or nr_samples_per_audio is less than 1.

    Notes
    -----
    The function applies the following processing steps:
    1. Preprocesses the input audio to the target sample rate
    2. Gets a random room scene with source and noise RIRs
    3. Gets a random noise sample
    4. Mixes the audio with noise at a random SNR (5-40 dB)
    5. Generates mel spectrograms from the mixed audio
    6. Extracts random segments from the spectrogram for training
    """

    # Save the state of the dummy variable.
    dummy = torch.zeros((nr_samples_per_audio, 100, 1))

    def generate_batch(sample):
        audio, audio_sr = sample[0]
        audio = pre_process_audio(audio, audio_sr, resample_sr)
        # Get the source and noise rirs.
        source_rir, noise_rirs, _ = next(
            random_scene_generator
        )  # Gives a random scene from data files with already processed RIRs
        noise, noise_sr = next(
            random_noise_generator
        )  # Gets a random noise from the noise dataset and loads it.\
        noise = pre_process_noise(noise, noise_sr, resample_sr)
        noise = generate_scenes.fade_noise(noise, audio, resample_sr)

        # This makes the noise start from a random part.
        # Basically pad the noise randomly with an empty signal
        if audio.shape[-1] > noise.shape[-1]:
            start_idx_noise = torch.randint(0, audio.shape[-1] - noise.shape[-1], (1,))
            new_agg_noise = torch.zeros_like(audio)
            new_agg_noise[start_idx_noise : start_idx_noise + noise.shape[-1]] = noise
            noise = new_agg_noise

        snr = torch.FloatTensor([1]).uniform_(5, 40)
        # Just repeat it.
        snr = repeat(snr, "1 -> 2")
        # Generate a masking for the input and target masks for each batch.
        # Here we need to know what is the nr_samples_per_audio.
        # Add clousere to python function to keep the state.
        context_idx, tgt_masks = masker(dummy)
        return audio, noise, source_rir, noise_rirs, snr, context_idx, tgt_masks

    return generate_batch


def _getitem_convolve(
    sample,
    random_scene_generator,
    random_noise_generator,
    resample_sr,
    target_length,
    nr_samples_per_audio=1,
):
    """
    Process a single audio sample for training with spatial audio.

    This function applies room impulse responses (RIRs), adds noise at
    a random SNR level, generates mel spectrograms, and extracts multiple random
    segments from each spectrogram for training.

    Parameters
    ----------
    sample : tuple
        A tuple containing audio data and its sample rate: (audio_data, sample_rate).
    random_scene_generator : iterator
        An iterator that provides random room scenes with source and noise RIRs.
        Expected to yield (source_rir, noise_rirs, source_rir_location) tuples.
    random_noise_generator : iterator
        An iterator that provides random noise samples.
        Expected to yield (noise_data, noise_sample_rate) tuples.
    target_length : int
        The target length (in ms) for the output spectrogram segments.
    input_length : int
        The input length (in samples) for audio processing.
    num_mel_bins : int
        Number of mel frequency bins for the spectrogram.
    resample_sr : int
        Target sample rate to resample audio if needed.
    nr_samples_per_audio : int, optional
        Number of spectrogram segments to extract from each audio sample, default is 1.

    Returns
    -------
    tuple
        A tuple containing:
        - return_fbank : torch.Tensor
            Mel spectrogram segments of shape (nr_samples_per_audio, n_channels, target_length, num_mel_bins)
        - target : torch.Tensor
            Source location coordinates repeated for each segment, shape (nr_samples_per_audio, 2)

    Raises
    ------
    AssertionError
        If target_length is greater than input_length or nr_samples_per_audio is less than 1.

    Notes
    -----
    The function applies the following processing steps:
    1. Preprocesses the input audio to the target sample rate
    2. Gets a random room scene with source and noise RIRs
    3. Gets a random noise sample
    4. Mixes the audio with noise at a random SNR (5-40 dB)
    5. Generates mel spectrograms from the mixed audio
    6. Extracts random segments from the spectrogram for training
    """

    # Get the audio, and preprocess it.
    audio, audio_sr = sample[0]
    audio = pre_process_audio(audio, audio_sr, resample_sr).float()
    # Get the source and noise rirs.
    source_rir, noise_rirs, source_rir_location = next(
        random_scene_generator
    )  # Gives a random scene from data files with already processed RIRs
    noise, noise_sr = next(
        random_noise_generator
    )  # Gets a random noise from the noise dataset and loads it.\
    noise = pre_process_noise(noise, noise_sr, resample_sr).float()
    snr = np.random.uniform(low=5, high=40)  # Random SNR between 5 and 40 :)
    # Generate the scenes with source and noise RIRs
    generated_scene = generate_scenes.generate_scene(
        source_rir=source_rir,
        noise_rirs=noise_rirs,
        source=audio,
        noise=noise,
        snr=snr,
        sr=resample_sr,
    )

    generated_scene = pad_or_truncate(generated_scene, 10 * resample_sr)
    # Just take X samples from the audio of 2 seconds and pass it onto the dataloader to shuffle them.
    return_audios = torch.zeros(
        (nr_samples_per_audio, generated_scene.shape[0], 2 * resample_sr)
    )
    target = torch.tensor([source_rir_location[0], source_rir_location[1]])
    for i in range(nr_samples_per_audio):
        start_idx = torch.randint(
            0, generated_scene.shape[1] - 2 * resample_sr, (1,)
        ).item()
        # Instance normalize over the cut audio fragments. Otherwise leakage is possible.
        return_audios[i] = instance_normalize(
            generated_scene[:, start_idx : start_idx + target_length]
        )

    return return_audios, target.repeat((nr_samples_per_audio, 1))


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
    waveform = resampler(waveform) if audio_sr != resample_sr else waveform
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
    waveform = resampler(waveform) if audio_sr != resample_sr else waveform
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
