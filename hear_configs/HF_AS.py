"""HEAR config for the AudioSet-pretrained HuBERT / wav2vec2 checkpoints (``ALM/hubert-{base,large}-audioset``,
``ALM/wav2vec2-{base,large}-audioset``), one pass over the whole clip as in the awsome-audio-foundation-models wrapper.

``--model <path>`` is the local snapshot directory (or a hub id). ``WAVJEPA_LAYERS=1,4,8,12`` switches to the layer-wise
probe (concatenated, per-layer LayerNorm), the same knob and run-name tag the WavJEPA configs use.
"""
import os

from hear_api.runtime_hf import RuntimeHFAudio


def load_model(model_file_path: str = "", *args, **kwargs):
    path = model_file_path or os.environ.get("HF_MODEL_PATH") or "ALM/hubert-base-audioset"
    layers = os.environ.get("WAVJEPA_LAYERS")
    layers = [int(v) for v in layers.split(",")] if layers else None
    return RuntimeHFAudio(path, layers=layers)


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
