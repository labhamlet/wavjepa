import sys

sys.path.append("..")
import torch
from einops import rearrange, repeat

from sjepa.jepa import JEPA
from sjepa.types import TransformerEncoderCFG, TransformerLayerCFG

from .feature_helper import FeatureExtractor


def normalize(audio):
    mean = audio.mean(dim=(-2, -1), keepdim=True)
    std = audio.std(dim=(-2, -1), keepdim=True)
    audio = (audio - mean) / (std + 1e-5)  # Add epsilon for stability
    return audio


def calculate_padding_mask(
    pad_frames, total_frames, sr, output_steps, process_seconds, model, B
):
    # How many 2 seconds chunks does this audio have?
    # Find it and then multiply by the output_steps.
    total_frames = int((total_frames / sr) / process_seconds)
    total_output_steps = output_steps * total_frames
    mask = torch.zeros((B, total_output_steps), dtype=torch.bool, device=model.device)

    # Check the number of padding tokens that we have in the audio.
    output_sr = int(output_steps / process_seconds)
    pad_seconds = pad_frames / sr
    pad_steps = int(pad_seconds * output_sr)
    # Create the mask
    mask[..., total_output_steps - pad_steps :] = True
    return mask, total_output_steps - pad_steps


class RuntimeNatJEPA(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        weights,
        is_spectrogram,
        process_seconds,
        extractor,
        model_size,
        sr,
        **kwargs,
    ) -> None:
        super().__init__()
        self.sample_rate = sr
        self.model = JEPA(
            feature_extractor=extractor,
            transformer_encoder_cfg=TransformerEncoderCFG.create(),
            transformer_encoder_layers_cfg=TransformerLayerCFG.create(),
            transformer_decoder_cfg=TransformerEncoderCFG.create(),
            transformer_decoder_layers_cfg=TransformerLayerCFG.create(d_model=384),
            in_channels=in_channels,
            resample_sr=self.sample_rate,
            size=model_size,
            is_spectrogram=is_spectrogram,
            process_audio_seconds=process_seconds,
        )

        new_state_dict = {}
        for key, value in weights["state_dict"].items():
            if key.startswith("extract_audio._orig_mod"):
                new_key = key.replace("extract_audio._orig_mod", "extract_audio")
                new_state_dict[new_key] = value
            elif key.startswith("encoder._orig_mod"):
                new_key = key.replace("encoder._orig_mod", "encoder")
                new_state_dict[new_key] = value
            elif key.startswith("decoder._orig_mod"):
                new_key = key.replace("decoder._orig_mod", "decoder")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        self.model.load_state_dict(new_state_dict, strict=False)
        self.embedding_size = self.model.encoder_embedding_dim
        self.scene_embedding_size = self.embedding_size
        self.timestamp_embedding_size = self.embedding_size
        self.unit_frames = int(process_seconds * self.sample_rate)
        self.output_steps = (
            self.model.extract_audio.total_patches(self.unit_frames)
            // self.model.in_channels
        )

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()
        self.feature_extractor = FeatureExtractor(in_channels=in_channels)

    def to_feature(self, batch_audio):
        return self.feature_extractor(batch_audio)

    def get_scene_embeddings(self, audio):
        embeddings, _ = self.get_timestamp_embeddings(audio)
        # This takes the mean embedding across the scene!
        embeddings = torch.mean(embeddings, dim=1)
        return embeddings

    def get_timestamp_embeddings(self, audio):
        B = audio.shape[0]
        audio = self.to_feature(audio)
        input_audio_len = audio.shape[-1]
        # Assert audio is of correct shape
        if audio.ndim != 3:
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
        padding_mask, cut_off = calculate_padding_mask(
            pad_frames=pad_frames,
            total_frames=audio.shape[-1],
            sr=self.sample_rate,
            output_steps=self.output_steps,
            process_seconds=self.model.target_length // self.sample_rate,
            model=self.model,
            B=B,
        )
        mask_idx = 0
        masked_mean = torch.zeros(audio.shape, dtype=torch.bool)
        masked_mean[..., cur_frames:] = True
        mt = torch.masked.masked_tensor(audio, masked_mean)
        # Now get the embeddings o the model.
        for i in range(audio.shape[-1] // self.unit_frames):
            mt = audio[..., i * self.unit_frames : (i + 1) * self.unit_frames]
            mask = padding_mask[..., mask_idx : mask_idx + self.output_steps]
            with torch.no_grad():
                # We do not include padding tokens in the mean and std calculation.
                mask = repeat(mask, "B E -> B (C E)", C=self.model.in_channels)
                embedding = self.model.get_audio_representation(normalize(mt), mask)
                embedding = rearrange(
                    embedding, "B (C S) E -> B C S E", C=self.model.in_channels
                )
                embedding = embedding.mean(dim=1)
            mask_idx = mask_idx + self.output_steps
            embeddings.append(embedding)

        x = torch.hstack(embeddings)
        x = x[:, :cut_off, :]
        ts = get_timestamps(self.sample_rate, B, input_audio_len, x)
        assert ts.shape[-1] == x.shape[1]
        return x, ts


def get_timestamps(sample_rate, B, input_audio_len, x):
    audio_len = input_audio_len
    sec = audio_len / sample_rate
    x_len = x.shape[1]
    step = sec / x_len * 1000  # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(B, 1)
    return ts
