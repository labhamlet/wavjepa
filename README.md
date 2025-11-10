# WavJEPA: Semantic learning unlocks robust audio foundation models for raw waveforms

Learning audio representations from raw waveforms overcomes key limitations of spectrogram-based audio representation learning, such as the long latency of spectrogram computation and the loss of phase information. Yet, while self-supervised speech representation learning from raw waveforms has been remarkably successful, these approaches have not achieved similar feats for general-purpose audio representation learning from waveforms. Here, we propose WavJEPA, a waveform-based version of the Joint-Embedding Predictive Architecture. WavJEPA leverages high-level semantic representation learning to tackle the shortcomings of representation learning at the speech unit or token level. We show that this approach substantially outperforms state-of-the-art time-domain audio foundation models across a wide variety of downstream benchmark tasks, while requiring considerably fewer computational resources. Additionally, to overcome the performance drop that time-domain models typically exhibit in noisy and reverberant real-world acoustic environments, we present WavJEPA-Nat. WavJEPA-Nat is a multi-channel extension of the WavJEPA architecture trained on simulated naturalistic scenes. We find that WavJEPA-Nat is highly robust to reverberation and noise. These results highlight the feasibility and computational efficiency of general-purpose audio representation learning from raw waveforms, showcasing the potential for low-latency, robust time-domain audio foundation models for real-world applications.

## Installation

We have two modes of installation: **Training** and **Evaluation** 

For training; install the requirements following the script
```bash
python3 -m pip install -r requirements.txt
```

For evaluation; install the requirements following the script
```bash
python3 -m pip install -r requirements_eval.txt
```

## Usage
### Getting the AudioSet data

We train WavJEPA on the unbalanced training set of AudioSet, which consists of 1.74 million 10-second sound clips scraped from YouTube (Gemmeke
et al., 2017). Download the AudioSet data we have used following the huggingface dataloading. The data download link is provided below.
https://huggingface.co/datasets/agkphysics/AudioSet
We have used data shards to train the model with WebDataLoader. These needs the path to the shards to be set. Navigate to config/data to set your paths after downloading. 

```yaml
base_data_dir: "/gpfs/work3/2/managed_datasets/hf_cache_dir/datasets--agkphysics--AudioSet/snapshots/5a2fa42a1506470d275a47ff8e1fdac5b364e6ef/data/unbal_train{000..869}.tar"
val_data_dir: "/gpfs/work3/2/managed_datasets/hf_cache_dir/datasets--agkphysics--AudioSet/snapshots/5a2fa42a1506470d275a47ff8e1fdac5b364e6ef/data/unbal_train{850..869}.tar"
```

There is no need for setting noise and rir dir for reporducing/training wavjepa. Turn the flags off, and set the clean data ratio to 1.0

Make sure to have this config.

```yaml
clean_data_ratio : 1.0
with_noise : True
with_rir : True
```

Each sound clip was resampled to 16 kHz and mean centered to enforce equal loudness across sound clips. We then randomly sampled 8 sections of 2 s from each sound clip, effectively increasing the batch size by a factor of 8 in a computationally efficient manner. Finally, each instance is instance normalized (Ulyanov et al., 2017). The waveform encoder converts each 2 s instance into an embedding w
200×768, effectively resampling the audio to 100 Hz with a stride of 10 ms and a receptive field size of 12.5 ms
### SSL pre-training 

The WavJEPA framework comprises a waveform encoder, context encoder, target encoder and a predictor. WavJEPA’s objective is to predict latent
representation of various targets blocks based on a single context block extracted from the same sound wave. As waveform encoder, we use the feature encoder of Wav2Vec 2.0, which is composed
of stacked temporal convolution layers (Baevski et al., 2020). Similar to the original I-JEPA architecture (Assran et al., 2023), a Vision Transformer (ViT) (Dosovitskiy et al., 2021) is used for the
target encoder, context encoder and predictor. For more information please check the paper: https://arxiv.org/pdf/2509.23238


### Training
The main endpoint for training is the train.py script. Here, we used PytorchLightning with Hydra configurations. 
We expose data, extractor (CNN Front-end), Masking, Training, and optimizer configs. 

After setting up the data paths, you can start training the WavJEPA model with:

```bash
python3 train.py data=audioset
```

This saves checkpoints to the directory that you specified at config/base.yaml so make sure to change the directory.
```bash
save_dir: /projects/0/prjs1338
seed: 42
```

### Inference

We provided HEAR API Inference endpoints for ease of use. However, we recommend to use the huggingface inference endpoints for feature extraction.

~~~python
import torch
from transformers import AutoModel, AutoFeatureExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained("labhamlet/wavjepa-base", trust_remote_code=True).to(device)
extractor = AutoFeatureExtractor.from_pretrained("labhamlet/wavjepa-base", trust_remote_code=True)

audio = torch.zeros([1,160000]).to(device)
extracted = extractor(audio, return_tensors="pt")
audio_feature = extracted['input_values']
result = model(audio_feature)
print(result[0].shape)
print(result[1].shape)
~~~

Later, you can extract features from our pre-trained wavjepa model and use it in the downstream tasks. 

Similarly, to use WavJEPA-Nat (Which is inherently a binaural model, but it is also very good in reverberent settings)

~~~python
import torch
from transformers import AutoModel, AutoFeatureExtractor

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained("labhamlet/wavjepa-nat-base", trust_remote_code=True).to(device)
extractor = AutoFeatureExtractor.from_pretrained("labhamlet/wavjepa-nat-base", trust_remote_code=True)

audio = torch.zeros([1,2,160000]).to(device)
extracted = extractor(audio, return_tensors="pt")
audio_feature = extracted['input_values']
result = model(audio_feature)
print(result[0].shape)
print(result[1].shape)
~~~



## Reproducing the HEAR results

Follow https://hearbenchmark.com/hear-tasks.html to get data. By default, data on HEAR's zenodo page is 48000 Hz.
We recommend downloading data directly from HEAR's GCS bucket, where you can find preprocessed 16000 Hz data.
Extract all the files to a folder $TASKS_DIR

### Extract Features

After downloading the data, you can use our huggingface endpoint to extract features on the HEAR data.

First, navigate to the 
```bash
cd listen-eval-kit
```

Here, you need to change the path in line 36 heareval/embeddings/task_embeddings.py to your root directory.
```python
sys.path.append("/home/gyuksel3/phd/gitpull/wavjepa") #Append the root directory
```

Afterwards, you can run the extraction and fine-tuning.

```bash
task_name=esc50-v2.0.0-full
tasks_dir=$TASKS_DIR

embeddings_dir=$EMBEDDINGS_DIR
score_dir=$SCORE_DIR

model_name=hear_configs.WavJEPA_huggingface

python3 -m heareval.embeddings.runner "$model_name" --tasks-dir $tasks_dir --task "$task_name" --embeddings-dir $embeddings_dir
python3 -m heareval.predictions.runner $embeddings_dir/$model_name/$task_name
```

## Results

**HEAR**

| Model | Size | DCASE | FSD50K | LC | ESC-50 | CD | VL | SC-5 | NS | BO | Mri-S | Mri-T | s(m) |
|-------|------|-------|--------|-----|--------|-----|-----|------|-----|-----|-------|-------|------|
| **WavJEPA** | B | 93.9 | 54.4 | 76.7 ± 2.4 | 86.5 ± 3.3 | 71.0 ± 0.8 | 49.8 ± 3.4 | 90.0 | 34.4 | 89.4 ± 5.4 | 97.3 ± 0.4 | 88.5 ± 0.5 | 66.0 |


```
## Cite
```bibtex
@misc{yuksel2025wavjepasemanticlearningunlocks,
      title={WavJEPA: Semantic learning unlocks robust audio foundation models for raw waveforms}, 
      author={Goksenin Yuksel and Pierre Guetschel and Michael Tangermann and Marcel van Gerven and Kiki van der Heijden},
      year={2025},
      eprint={2509.23238},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2509.23238}, 
}
```
