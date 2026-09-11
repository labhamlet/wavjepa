"""HEAR runtime for Hugging Face wav2vec2 / HuBERT checkpoints (the AudioSet-pretrained ``ALM/*-audioset`` models).

Replicates the awsome-audio-foundation-models wrapper (``HuggingfaceModels/{HuBERT_AS,Wav2Vec2.0_AS}/hear_api/runtime.py``):
the whole clip goes through the model's own ``Wav2Vec2FeatureExtractor`` (per-clip zero-mean / unit-variance, ``do_normalize``)
and then through the transformer in ONE pass; timestamp embeddings are ``last_hidden_state`` (one 768/1024-d vector per 20 ms
frame), scene embeddings the mean over frames, timestamps ``i * clip_seconds / n_frames`` as in the wrapper.

Extension for the layer-wise probe: ``layers=[1, 4, 8, 12]`` concatenates ``hidden_states[l]`` of those transformer blocks
(``hidden_states[0]`` is the block-1 input, ``hidden_states[l]`` the output of block ``l``), each through a parameter-free
LayerNorm, exactly as ``RuntimeJEPA(layers=...)`` does for WavJEPA, so the two probes are comparable.
"""
from __future__ import annotations

import torch
from transformers import AutoFeatureExtractor, AutoModel


class RuntimeHFAudio(torch.nn.Module):
    def __init__(self, model_path: str, layers: list[int] | None = None) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_path)
        self.extractor = AutoFeatureExtractor.from_pretrained(model_path)
        self.layers = list(layers) if layers else None
        if self.layers and max(self.layers) > self.model.config.num_hidden_layers:
            raise ValueError(f"layers {self.layers} but the model has {self.model.config.num_hidden_layers} blocks")
        width = int(self.model.config.hidden_size)
        self.embedding_size = width * (len(self.layers) if self.layers else 1)
        self.scene_embedding_size = self.timestamp_embedding_size = self.embedding_size
        self.sample_rate = int(self.extractor.sampling_rate)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def _inputs(self, audio: torch.Tensor) -> torch.Tensor:
        # the eval kit hands over (n_sounds, n_samples); the HF extractor wants a list of 1-D arrays
        clips = [a.detach().cpu().float().numpy() for a in audio]
        x = self.extractor(clips, sampling_rate=self.sample_rate, return_tensors="pt", padding="longest").input_values
        return x.to(next(self.model.parameters()).device)

    @torch.no_grad()
    def get_timestamp_embeddings(self, audio: torch.Tensor):
        if audio.ndim != 2:
            raise ValueError("audio input tensor must be 2D with shape (n_sounds, num_samples)")
        x = self._inputs(audio)
        out = self.model(x, output_hidden_states=self.layers is not None)
        if self.layers is None:
            emb = out.last_hidden_state
        else:
            hs = out.hidden_states
            emb = torch.cat([torch.nn.functional.layer_norm(hs[l], hs[l].shape[-1:]) for l in self.layers], dim=-1)
        n_frames = emb.shape[1]
        step_ms = 1000.0 * audio.shape[-1] / self.sample_rate / n_frames
        ts = (step_ms * torch.arange(n_frames, dtype=torch.float32)).unsqueeze(0).repeat(emb.shape[0], 1)
        return emb, ts

    @torch.no_grad()
    def get_scene_embeddings(self, audio: torch.Tensor) -> torch.Tensor:
        emb, _ = self.get_timestamp_embeddings(audio)
        return emb.mean(dim=1)
