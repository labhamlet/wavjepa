import sys
sys.path.append('..')

import numpy as np
import json
import torch 

from arch_eval import ESC50
from arch_eval import US8K
from arch_eval import FSD50K
from arch_eval import VIVAE

from arch_eval import FMASmall
from arch_eval import MagnaTagATune
from arch_eval import IRMAS
from arch_eval import MedleyDB

from arch_eval import RAVDESS
from arch_eval import AudioMNIST
from arch_eval import SLURP
from arch_eval import EMOVO

from configs.wavjepa_wrapper import WavJEPAModelWrapper
from wavjepa.extractors import ConvFeatureExtractor
from wavjepa.jepa import JEPA
from wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG
import wavjepa
import argparse

sys.modules['sjepa'] = wavjepa

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument('--weights', type=str)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_epochs', type=int, default=200)
parser.add_argument('--verbose', default=False, action = 'store_true')
parser.add_argument('--tsv_logging_file', type=str, default='results/hf_models.tsv')
parser.add_argument('--n_iters', type=int, default=1)
parser.add_argument('--data_config_file', type=str, default='configs/datasets_config.json')
parser.add_argument('--attentive_pooling', default=False, action = 'store_true')
parser.add_argument('--precompute_embeddings', default=False, action = 'store_true')
parser.add_argument('--enabled_datasets', type=str, nargs='+', default=["esc50", "us8k", "fsd50k", "vivae", 
                                                                        "fma_small", "magna_tag_a_tune", "irmas", "medleydb",
                                                                        "ravdess", "audio_mnist", "slurp", "emovo"])
args = parser.parse_args()

# example command:
# python3 evaluate_wavjepa_model.py --weights /gpfs/work4/0/prjs1338/saved_models_jepa_topk8_lr50k_lr_00001_bigger_targets_conv_20_10/InChannels=1/WithNoise=False/WithRIR=False/Extractor=spatial-conv-extractor/DepthWise=False/Share=None/BatchSize=64/NrSamples=8/Masking=random-cluster-masker/MaskProb=0.065/MaskLen=None/Cluster=None/TopK=8/step=500000.ckpt --device cuda --max_epochs 200 --verbose --tsv_logging_file results/wavjepa_cluster.tsv --n_iters 1 --data_config_file configs/datasets_config.json --enabled_datasets esc50 --precompute_embeddings

print("------------------------------------")
print(f"Evaluating model: {args.weights}")
print("------------------------------------")

'''
************************************************************************************************
*                                       Setting parameters                                     *
************************************************************************************************
'''

# Load model
weights=torch.load(
    args.weights,
    weights_only = False,
    map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
)

extractor = ConvFeatureExtractor(conv_layers_spec  = eval("[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)]"),
                                in_channels = 1)
model = JEPA(
    feature_extractor = extractor,
    transformer_encoder_cfg = TransformerEncoderCFG.create(),
    transformer_encoder_layers_cfg = TransformerLayerCFG.create(),
    transformer_decoder_cfg = TransformerEncoderCFG.create(),
    transformer_decoder_layers_cfg = TransformerLayerCFG.create(d_model = 384),
    resample_sr=16000,
    process_audio_seconds = 2.01)

new_state_dict = {}
for key, value in weights["state_dict"].items():
    if key.startswith('extract_audio._orig_mod'):
        new_key = key.replace('extract_audio._orig_mod', 'extract_audio')
        new_state_dict[new_key] = value
    elif key.startswith('encoder._orig_mod'):
        new_key = key.replace('encoder._orig_mod', 'encoder')
        new_state_dict[new_key] = value
    elif key.startswith('decoder._orig_mod'):
        new_key = key.replace('decoder._orig_mod', 'decoder')
        new_state_dict[new_key] = value
    else:
        new_state_dict[key] = value

model.load_state_dict(new_state_dict, strict = False)
audio_model = model.to(args.device)
model_parameters = sum(p.numel() for p in audio_model.parameters())
tsv_lines = [] 


# load datasets info
with open(args.data_config_file) as f:
    datasets_info = json.load(f)

enabled_datasets = args.enabled_datasets

for dataset_name in enabled_datasets:
    
    model = WavJEPAModelWrapper(audio_model, args.device, max_length=datasets_info[dataset_name]["max_length_seconds"]*16_000)
    
    if dataset_name == "esc50":
        evaluator = ESC50(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "us8k":
        evaluator = US8K(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "fma_small":
        evaluator = FMASmall(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "magna_tag_a_tune":
        evaluator = MagnaTagATune(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "irmas":
        evaluator = IRMAS(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "ravdess":
        evaluator = RAVDESS(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "audio_mnist":
        evaluator = AudioMNIST(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "fsd50k":
        evaluator = FSD50K(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "slurp":
        evaluator = SLURP(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "vivae":
        evaluator = VIVAE(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "medleydb":
        evaluator = MedleyDB(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    elif dataset_name == "emovo":
        evaluator = EMOVO(datasets_info[dataset_name]["path"], verbose=args.verbose, precompute_embeddings=args.precompute_embeddings)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")


    mode = "attention-pooling" if args.attentive_pooling else "linear"
    res = []
    for i in range(args.n_iters):
        if args.verbose:
            print(f"Iteration {i+1}/{args.n_iters}")
            print (f"----------------- {dataset_name} {mode} -----------------")

        res_dataset = evaluator.evaluate(
            model, 
            mode=mode, 
            device=args.device, 
            batch_size=32, 
            max_num_epochs=args.max_epochs, 
        )

        if args.verbose:
            print(f"Iteration {i+1}/{args.n_iters}")
            for metric, value in res_dataset.items():
                print(f"{metric}: {value}")

        res.append(res_dataset)

    res_mean = {}
    res_std = {}
    for metric in res[0].keys():
        res_mean[metric] = np.mean([r[metric] for r in res])
        res_std[metric] = np.std([r[metric] for r in res])

    if args.verbose:
        print(f"----------------- {dataset_name} {mode} -----------------")
        for metric, value in res_mean.items():
            print(f"{metric}: {res_mean[metric]} +/- {res_std[metric]}")

    # create a tsv line: model_tag, size, is_linear, dataset_name, mean_map_macro, std_map_macro, mean_map_weighted, std_map_weighted
    if datasets_info[dataset_name]["is_multilabel"]:
        tsv_lines.append(f"{args.model}\t{model_parameters}\tTrue\t{dataset_name}\t{res_mean['map_macro']}\t{res_std['map_macro']}\t{res_mean['map_weighted']}\t{res_std['map_weighted']}\n")
    else:
        tsv_lines.append(f"{args.model}\t{model_parameters}\tTrue\t{dataset_name}\t{res_mean['accuracy']}\t{res_std['accuracy']}\t{res_mean['f1']}\t{res_std['f1']}\n")

    with open(args.tsv_logging_file, "a") as f:
        if datasets_info[dataset_name]["is_multilabel"]:
            f.write(f"{args.model}\t{model_parameters}\tTrue\t{dataset_name}\t{res_mean['map_macro']}\t{res_std['map_macro']}\t{res_mean['map_weighted']}\t{res_std['map_weighted']}\n")
        else:
            f.write(f"{args.model}\t{model_parameters}\tTrue\t{dataset_name}\t{res_mean['accuracy']}\t{res_std['accuracy']}\t{res_mean['f1']}\t{res_std['f1']}\n")


if args.verbose:
    print("\n\nAll results:")
    for line in tsv_lines:
        print(line)

# # append tsv lines in file
# with open(args.tsv_logging_file, "a") as f:
#     for line in tsv_lines:
#         f.write(line)