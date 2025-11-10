import torch
from hear_api.runtime_natjepa import RuntimeNatJEPA
from sjepa.extractors import ConvChannelFeatureExtractor


def load_model(*args, **kwargs):
    weights = None
    if len(args) != 0:
        model_path = args[0]
        weights = torch.load(
            model_path,
            weights_only=False,
            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )

    sr = kwargs.get("sr", 16000)
    sr = int(sr)
    model_size = kwargs.get("model", "base")

    extractor = ConvChannelFeatureExtractor(
        conv_layers_spec=eval("[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)]"),
        in_channels=2,
        share_weights_over_channels=False,
    )
    model = RuntimeNatJEPA(
        in_channels=2,
        process_seconds=2.01,
        weights=weights,
        sr=sr,
        model_size=model_size,
        is_spectrogram=False,
        extractor=extractor,
    )
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
