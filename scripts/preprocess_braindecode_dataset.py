"""
Loads, preprocesses, and saves a braindecode dataset.

Useful for TUH for example.
"""

from pathlib import Path

from braindecode.datasets.base import BaseConcatDataset
from braindecode.preprocessing import Preprocessor, preprocess


def select_by_channels(ds: BaseConcatDataset, ch_names: list[str]) -> BaseConcatDataset:
    # these are the channels we are looking for
    seta = set(ch_names)
    split_ids = []
    for i, d in enumerate(ds.datasets):
        # these are the channels of the recoding
        setb = set(d.raw.ch_names)
        # if recording contains all channels we are looking for, include it
        if seta.issubset(setb):
            split_ids.append(i)
    return ds.split(split_ids)["0"]


def process(
    dataset, save_dir, mandatory_ch_names, preprocessors, offset, n_jobs, overwrite
):
    print(f"Processing {dataset}")
    if mandatory_ch_names is not None:
        dataset = select_by_channels(dataset, mandatory_ch_names)
    for d in dataset.datasets:
        # I had a RuntimeError: info["meas_date"] seconds out of range
        if d.raw.info["meas_date"].year < 2000:
            meas_date = d.raw.info["meas_date"].replace(year=2000)
            d.raw.set_meas_date(meas_date)
    if offset == 0:  # if only one dataset, save as soon as the files are preprocessed
        preprocess(
            dataset,
            preprocessors=preprocessors,
            n_jobs=n_jobs,
            save_dir=save_dir,
            overwrite=overwrite,
        )
    else:  # if multiple datasets, save only after all files are preprocessed
        preprocess(dataset, preprocessors=preprocessors, n_jobs=n_jobs)
        dataset.save(save_dir, offset=offset, overwrite=overwrite)
    n_files = len(dataset.datasets)
    return offset + n_files


def main(
    datasets: BaseConcatDataset | list[BaseConcatDataset],
    path: str = None,
    sfreq: float = 128,
    l_freq: float = 0.5,
    h_freq: float = 40,
    # discards recordings that do not contain these channels:
    mandatory_ch_names: list[str] | None = None,
    suffix: str = "",
    n_jobs: int = -1,
    overwrite: bool = False,
):
    if not isinstance(datasets, list):
        datasets = [datasets]
    preprocessors = [
        Preprocessor("pick_types", eeg=True, meg=False, stim=False),
        Preprocessor("resample", sfreq=sfreq),
        Preprocessor("filter", l_freq=l_freq, h_freq=h_freq),
    ]
    offset = 0
    for dataset in datasets:
        save_dir = (
            Path(path).expanduser() / f"{dataset.__class__.__name__.lower()}{suffix}"
        )
        save_dir.mkdir(exist_ok=True, parents=True)
        assert save_dir.is_dir()
        offset = process(
            dataset,
            save_dir=save_dir,
            mandatory_ch_names=mandatory_ch_names,
            preprocessors=preprocessors,
            offset=offset,
            n_jobs=n_jobs,
            overwrite=overwrite,
        )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(main, fail_untyped=False)

    # Example command:
    # python scripts/preprocess_braindecode_dataset.py --config=scripts/config/eeg_jepa_TUH_preprocessing.yaml --n_jobs=-1 --datasets.n_jobs=-1

    # For testing:
    # python scripts/preprocess_braindecode_dataset.py --config=scripts/config/eeg_jepa_TUH_preprocessing.yaml --datasets.recording_ids="[1,]" --datasets.path="/Users/Pierre.Guetschel/data/tuh_eeg" --overwrite=true
