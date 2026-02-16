from typing import Optional

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio
from arch_eval import Model, ClassificationModel

def resample(audio: torch.Tensor, resample_sr: int, original_sr = 32000) -> torch.Tensor:
    return torchaudio.functional.resample(
        audio,
        original_sr,
        resample_sr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )

def normalize_segment(audio):
    mean = audio.mean(dim=(-2, -1), keepdim=True)
    std = audio.std(dim=(-2, -1), keepdim=True)
    audio = (audio - mean) / (std + 1e-5) # Add epsilon for stability
    return audio

def normalize_audio(audio_data, target_dBFS=-14.0):
    rms = torch.sqrt(torch.mean(audio_data**2))  # Calculate the RMS of the audio
    if rms == 0:  # Avoid division by zero in case of a completely silent audio
        return audio_data
    current_dBFS = 20 * torch.log10(rms)  # Convert RMS to dBFS
    gain_dB = target_dBFS - current_dBFS  # Calculate the required gain in dB
    gain_linear = 10 ** (gain_dB / 20)  # Convert gain from dB to linear scale
    normalized_audio = audio_data * gain_linear  # Apply the gain to the audio data
    return normalized_audio

def calculate_padding_mask(pad_frames, total_frames, sr, output_steps, process_seconds, model):
    # How many 2 seconds chunks does this audio have?
    # Find it and then multiply by the output_steps.
    total_frames = int((total_frames / sr) / process_seconds)
    total_output_steps = output_steps * total_frames
    # Check the number of padding tokens that we have in the audio.
    output_sr = int(output_steps / process_seconds)
    pad_seconds = pad_frames / sr
    pad_steps = int(pad_seconds * output_sr)
    # Create the mask
    mask = torch.zeros((1, total_output_steps), dtype = torch.bool, device = model.device)
    mask[..., total_output_steps - pad_steps:] = True 
    return mask


# implement a child class of Model
class WavJEPAModelWrapper(Model):
    def __init__(self, model, device, max_length):
        super().__init__(model)
        self.model = model
        self.sr = 16000
        self.model.eval()
        self.device = device
        self.max_length = max_length
        self.unit_frames = model.target_length
        self.output_steps = model.extract_audio.total_patches(self.unit_frames)
        

    def get_embeddings(self, audio: np.ndarray, **kwargs):
        audio = audio.to(self.device)
        audio = audio.view(1,1,-1)
        # Apply loudness normalization
        audio = normalize_audio(audio)
        if audio.ndim != 3:
            raise ValueError(
                "audio input tensor must be 3D with shape (n_sounds, n_channels, num_samples)"
            )
        cur_frames = audio.shape[-1]
        pad_frames = self.unit_frames - (cur_frames % self.unit_frames)
        if pad_frames > 0:
            # Padding with constant 0s
            pad_arg = (
                0,
                pad_frames,
            )  # (channel, channel, height, height, width, width)
            audio = torch.nn.functional.pad(audio, pad_arg, mode="constant")
        padding_mask = calculate_padding_mask(pad_frames = pad_frames, 
                                              total_frames = audio.shape[-1], 
                                              sr = self.sr,
                                              output_steps = self.output_steps,
                                              process_seconds = self.model.target_length // self.sr, 
                                              model = self.model)
        embeddings = [] 
        mask_idx = 0
        masked_mean = torch.zeros(audio.shape, dtype = torch.bool)
        masked_mean[..., cur_frames:] = True
        mt = torch.masked.masked_tensor(audio, masked_mean)
        for i in range(audio.shape[-1] // self.unit_frames):
            mt = audio[..., i * self.unit_frames : (i + 1) * self.unit_frames]
            mask = padding_mask[...,mask_idx : mask_idx + self.output_steps]
            with torch.no_grad():
                embedding = self.model.get_audio_representation(
                    normalize_segment(mt),
                    mask
                )
            embedding = embedding[~mask]
            embeddings.append(embedding)
            mask_idx = mask_idx + self.output_steps
        
        embeddings = torch.cat(embeddings)
        embeddings = embeddings.mean(dim=0).squeeze()
        return embeddings

    def get_sequence_embeddings(self, audio: np.ndarray, **kwargs):
        inputs = resample(
            audio, 
            sampling_rate=self.sr, 
            max_length=self.max_length,
        )
        inputs = inputs.to(self.device)
        if inputs.ndim != 3:
            raise ValueError(
                "audio input tensor must be 2D with shape (n_sounds, n_channels, num_samples)"
            )
        cur_frames = audio.shape[-1]
        pad_frames = self.unit_frames - (cur_frames % self.unit_frames)
        if pad_frames > 0:
            # Padding with constant 0s
            pad_arg = (
                0,
                pad_frames,
            )  # (channel, channel, height, height, width, width)
            audio = torch.nn.functional.pad(audio, pad_arg, mode="constant")
        embeddings = []
        # Now get the embeddings of the model.
        for i in range(audio.shape[-1] // self.unit_frames):
            x_inp = audio[..., i * self.unit_frames : (i + 1) * self.unit_frames]
            with torch.no_grad():
                embedding = self.model.get_audio_representation(
                    x_inp
                )
            embeddings.append(embedding)

        embeddings = rearrange(torch.stack(embeddings), 's b n d -> b (n s) d') 

        return embeddings.squeeze()

    def get_classification_embedding_size(self):
        return self.model.encoder_embedding_dim

    def get_token_embedding_size(self):
        return self.model.encoder_embedding_dim

    def get_sampling_rate(self):
        return self.sr

    def get_embedding_layer(self):
        # return the size of the embedding layer
        return self.model.encoder_embedding_dim