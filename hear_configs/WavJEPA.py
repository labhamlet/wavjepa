import torch

from hear_api.runtime import RuntimeJEPA
from wavjepa.extractors import ConvFeatureExtractor
import sys 
import wavjepa 

sys.modules['sjepa'] = wavjepa

SR = 16000
def load_model(*args, **kwargs):
    weights = None
    if len(args) != 0:
        model_path = args[0]
        weights = torch.load(
            model_path,
            weights_only=False,
            map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        )


    if kwargs["extractor"] == "wav2vec2":
        extractor = ConvFeatureExtractor(
            conv_layers_spec=eval("[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]"),
            in_channels=1,
        )
    else:
        extractor = ConvFeatureExtractor(
            conv_layers_spec=eval("[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)]"),
            in_channels=1,
        )    

    model = RuntimeJEPA(
        in_channels=1,
        process_seconds=float(kwargs["in_seconds"]),
        weights=weights,
        sr=SR,
        model_size="base",
        is_spectrogram=False,
        extractor=extractor,
    )
    return model


def get_scene_embeddings(audio, model):
    return model.get_scene_embeddings(audio)


def get_timestamp_embeddings(audio, model):
    return model.get_timestamp_embeddings(audio)
