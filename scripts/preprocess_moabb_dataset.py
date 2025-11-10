"""
Loads, preprocesses, and saves a MOABB dataset,
subject by subject sequentially.

Useful for Lee2019 (54 subjects, ~80GB) for example.
"""

from pathlib import Path

import yaml
from braindecode.datasets import MOABBDataset
from braindecode.preprocessing import Preprocessor, preprocess
from moabb.datasets.utils import dataset_list
from tqdm import tqdm

dataset_name_map = {ds.__name__: ds for ds in dataset_list}


def process(dataset_name, subject_id, save_dir, preprocessors, offset, dataset_kwargs):
    dataset = MOABBDataset(
        dataset_name, subject_ids=[subject_id], dataset_kwargs=dataset_kwargs
    )
    preprocess(dataset, preprocessors=preprocessors, n_jobs=-1)
    dataset.save(save_dir, offset=offset, overwrite=False)
    n_files = len(dataset.datasets)
    return offset + n_files


def main(
    dataset_name,
    path,
    dataset_kwargs: dict = None,
    sfreq=128,
    l_freq=0.5,
    h_freq=40,
    suffix="",
):
    ds = dataset_name_map[dataset_name]()
    save_dir = Path(path).expanduser() / f"{ds.code.lower()}{suffix}"
    save_dir.mkdir(exist_ok=True, parents=True)
    assert save_dir.is_dir()
    subject_ids = ds.subject_list
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor("set_eeg_reference", ref_channels="average", ch_type="eeg"),
        Preprocessor("resample", sfreq=sfreq),
        Preprocessor("filter", l_freq=l_freq, h_freq=h_freq),
    ]
    offset = 0
    for subject_id in tqdm(subject_ids):
        offset = process(
            dataset_name,
            subject_id,
            offset=offset,
            save_dir=save_dir,
            preprocessors=preprocessors,
            dataset_kwargs=dataset_kwargs,
        )


def main_main(config_path: str):
    with open(config_path) as f:
        config = yaml.safe_load(f)
    args = ["sfreq", "l_freq", "h_freq", "path", "suffix"]
    kwargs = {k: config[k] for k in args}
    for dataset_config in config["datasets"]:
        print(f"Processing {dataset_config}")
        main(**{f"dataset_{k}": v for k, v in dataset_config.items()}, **kwargs)


if __name__ == "__main__":
    import fire

    fire.Fire(main_main)

    # Example command:
    # python scripts/preprocess_moabb_dataset.py scripts/config/eeg_jepa_Lee2019_preprocessing.yaml
