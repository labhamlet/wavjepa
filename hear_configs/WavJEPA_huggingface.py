import torch
from transformers import AutoFeatureExtractor, AutoModel

extractor = AutoFeatureExtractor.from_pretrained(
    "labhamlet/wavjepa-base", trust_remote_code=True
)


def get_timestamps(sample_rate, batch_audio, x):
    audio_len = len(batch_audio[0])
    sec = audio_len / sample_rate
    x_len = len(x[0])
    step = sec / x_len * 1000  # sec -> ms
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(len(batch_audio), 1)
    return ts


def load_model(*args, **kwargs):
    model = AutoModel.from_pretrained(
        "labhamlet/wavjepa-base", trust_remote_code=True, force_download=True
    )
    model.sample_rate = 16000
    return model


def get_scene_embeddings(audio, model):
    extracted = extractor(audio, return_tensors="pt")
    log_mel = extracted["input_values"]
    x, ts = model(log_mel)
    x = torch.mean(x, dim=1)
    return x


def get_timestamp_embeddings(audio, model):
    extracted = extractor(audio, return_tensors="pt")
    log_mel = extracted["input_values"]
    x, ts = model(log_mel, strategy="raw")
    return x, ts
