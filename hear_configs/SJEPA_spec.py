import torch

from hear_api.runtime import RuntimeJEPA
from sjepa.extractors import SpectrogramPatchExtractor


def load_model(*args, **kwargs):
    weights = None
    if len(args) != 0:
        model_path = args[0]
        weights = torch.load(
            model_path,
            weights_only=False,
            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )

    extractor = SpectrogramPatchExtractor(
        n_mels=128,
        sr=32000,
        embedding_dim=768,
        in_channels=1,
        fshape=16,
        tshape=8,
        fstride=16,
        tstride=8,
    )
    model = RuntimeJEPA(
        in_channels=1,
        process_seconds=2.01,
        weights=weights,
        is_spectrogram=True,
        extractor=extractor,
    )
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
