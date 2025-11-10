from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import wandb
from braindecode.datautil import load_concat_dataset
from braindecode.preprocessing import create_windows_from_events
from mne import events_from_annotations
from sjepa.braindecode_datamodule import wrapp_windows
from sjepa.eeg_jepa import EegJepa
from torch.utils.data import DataLoader
from torchinfo import summary


def main(trainer: pl.Trainer, epoch_eps=0.1, run_id="ln3q8v7q", log_wandb=False):
    # reference can be retrieved in artifacts panel
    # "VERSION" can be a version (ex: "v2") or an alias ("latest or "best")
    checkpoint_reference = (
        f"self-supervised-spd/self-supervised-masked/model-{run_id}:latest"
    )

    # download checkpoint locally (if not already cached)
    run = wandb.init(
        entity="self-supervised-spd",
        project="self-supervised-masked",
        resume="must",
        id=run_id,
    )
    artifact = run.use_artifact(checkpoint_reference, type="model")
    artifact_dir = artifact.download()

    # load checkpoint
    model = EegJepa.load_from_checkpoint(Path(artifact_dir) / "model.ckpt")
    print(summary(model))

    # dataset path
    data_path = Path("~").expanduser() / "data" / "preprocessed_datasets"
    datasets_names = [
        "lee2019-mi_128_0.5-40",
        "lee2019-ssvep_128_0.5-40",
        "lee2019-erp_128_0.5-40",
    ]
    predictions_path = Path("./predictions")
    predictions_path.mkdir(exist_ok=True)

    for n in datasets_names:
        print(f"Dataset: {n}")
        # load raw dataset
        raws_dataset = load_concat_dataset(data_path / n, n_jobs=-1, preload=False)

        # make windows
        durations, counts = np.unique(
            raws_dataset.datasets[0].raw.annotations.duration, return_counts=True
        )
        windows_length = durations[counts.argmax()] + epoch_eps  # seconds
        print(f"Windows length: {windows_length}")
        for d in raws_dataset.datasets:
            # fix unequal durations:
            d.raw.annotations.set_durations(windows_length)
            # fix overlapping events:
            mask = np.diff(events_from_annotations(d.raw)[0][:, 0]) > 0
            mask = [True] + list(mask)
            d.raw.set_annotations(d.raw.annotations[mask])
        windows_dataset = create_windows_from_events(raws_dataset)
        final_dataset = wrapp_windows(windows_dataset)

        # predictions
        dataloader = DataLoader(
            final_dataset,
            batch_size=32,
            num_workers=8,
        )
        predictions = trainer.predict(model, dataloader)

        # save predictions locally
        table_path = predictions_path / f"{n}.parquet"
        table = (
            windows_dataset.get_metadata().copy().reset_index(drop=True).reset_index()
        )
        for k in predictions[0].keys():
            table[k] = [t.numpy() for tt in predictions for t in tt[k]]
            table[f"{k}_shape"] = table[k].map(np.shape)
            table[k] = table[k].map(np.ravel)
        table.to_parquet(table_path, compression=None)

        # save prediction wandb as artifact
        if log_wandb:
            artifact = wandb.Artifact(f"predictions_{n}_{run_id}", type="predictions")
            artifact.add_file(table_path)
            run.log_artifact(artifact)

        # out = dict(
        #     metadata=windows_dataset.get_metadata(),
        #     predictions=predictions,
        # )
        # with open(predictions_path / f'{n}.pkl', 'wb') as f:
        #     pickle.dump(out, f)


def test_main():
    trainer = pl.Trainer(accelerator="cpu")
    main(trainer)


if __name__ == "__main__":
    from lightning.pytorch.cli import LightningArgumentParser

    parser = LightningArgumentParser()
    parser.add_class_arguments(pl.Trainer, "trainer")
    parser.add_argument("--epoch_eps", type=float, default=0.1)
    parser.add_argument("--run_id", type=str, default="ln3q8v7q")
    args = parser.parse_args()
    trainer = pl.Trainer(**args.trainer)
    main(trainer, epoch_eps=args.epoch_eps, run_id=args.run_id, log_wandb=True)
