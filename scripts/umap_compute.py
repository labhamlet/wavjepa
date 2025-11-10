from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import wandb
from umap import UMAP


def main(
    run_id="ln3q8v7q", max_embeddings=1000, n_subjects=54, style="run", hue="target"
):
    # paths and names
    datasets_names = [
        "lee2019-mi_128_0.5-40",
        "lee2019-ssvep_128_0.5-40",
        # 'lee2019-erp_128_0.5-40',
    ]
    n_cols = 5
    n_rows = ceil(n_subjects / n_cols)
    embedding_keys = ["local_features", "contextualized_features"]
    metadata_cols = ["subject", "session", "run", "target"]
    export_dir = Path("export")
    export_dir.mkdir(exist_ok=True)
    for n in datasets_names:
        print(f"Dataset: {n}")
        # download predictions
        checkpoint_reference = f"self-supervised-spd/self-supervised-masked/predictions_{n}_{run_id}:latest"
        # select data
        try:
            table_path = (
                Path("artifacts") / checkpoint_reference.split("/")[-1] / f"{n}.parquet"
            )
            metadata = pd.read_parquet(table_path, columns=metadata_cols + ["index"])
        except:
            api = wandb.Api()
            artifact = api.artifact(checkpoint_reference)
            artifact_dir = artifact.download()
            table_path = Path(artifact_dir) / f"{n}.parquet"
            metadata = pd.read_parquet(table_path, columns=metadata_cols + ["index"])
        # metadata = metadata.set_index('index')
        # n_combinations = len(metadata.groupby(metadata_cols))
        # print(f'Number of combinations: {n_combinations}')
        # this_max_embeddings = max_embeddings
        # filters = None
        # if len(metadata) > max_embeddings and len(metadata) - max_embeddings > n_combinations:
        #     if n_combinations > max_embeddings:
        #         this_max_embeddings = n_combinations
        #     splitter = StratifiedShuffleSplit(n_splits=1, train_size=this_max_embeddings)
        #     idx, _ = next(splitter.split(metadata, metadata))
        #     filters = [('index', 'in', metadata.index[idx].to_numpy())]

        # ALL SUBJECTS TOGETHER
        # colouring = [c for c in metadata_cols if c != 'run'] + ['test_subject']
        # fig, axes = plt.subplots(2, len(metadata_cols), figsize=(2 * len(metadata_cols), 5))
        # for i, k in enumerate(embedding_keys):
        #     # load data
        #     table = pd.read_parquet(
        #         table_path, filters=filters,
        #         columns=[k, 'index'] + metadata_cols,
        #         engine='pyarrow',
        #     )
        #     # preprocess data
        #     table['test_subject'] = table['subject'] > 40
        #     print(f'Number of embeddings: {len(table)}')
        #     # compute umap
        #     embeddings = UMAP(n_components=2).fit_transform(list(table[k]))
        #     # plot
        #     for j, c in enumerate(colouring):
        #         ax = axes[i, j]
        #         sns.scatterplot(
        #             x=embeddings[:, 0], y=embeddings[:, 1],
        #             hue=table[c].astype('category'), ax=ax,
        #             alpha=0.7,
        #             legend=(c != 'subject'),
        #         )
        #         ax.set_title(f'{k} - {c}')
        #         ax.set(xticklabels=[], yticklabels=[])  # remove the tick labels
        #         ax.tick_params(bottom=False, left=False)  # remove the ticks
        # fig.suptitle(n)
        # plt.savefig(export_dir / f'umap_{n}_merged.pdf', bbox_inches='tight')
        # plt.show(block=False)
        # SUBJECT BY SUBJECT
        for i, k in enumerate(embedding_keys):
            print(f"Embedding: {k}")
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            for j, s in enumerate(range(1, n_subjects + 1)):
                print(f"Subject: {s}")
                # load data
                table = pd.read_parquet(
                    table_path,
                    filters=[("subject", "==", s)],
                    columns=[k, "index"] + metadata_cols,
                    engine="pyarrow",
                )
                print(f"Number of embeddings: {len(table)}")
                # compute umap
                embeddings = UMAP(n_components=2).fit_transform(list(table[k]))
                # plot
                ax = axes[j // n_cols, j % n_cols] if n_rows > 1 else axes[j]
                sns.scatterplot(
                    x=embeddings[:, 0],
                    y=embeddings[:, 1],
                    hue=table[hue].astype("category"),
                    style=table[style].astype("category"),
                    ax=ax,
                    alpha=0.7,
                    legend=("auto" if j == 0 else False),
                )
                ax.set_title(f"subject {s}")
                ax.set(xticklabels=[], yticklabels=[])
                ax.tick_params(bottom=False, left=False)  # remove the ticks
            fig.suptitle(f"{n} - {k}")
            plt.savefig(
                export_dir / f"umap_{n}__{run_id}_{k}_{hue}_{style}.pdf",
                bbox_inches="tight",
            )
            plt.show(block=False)


def umap_scatter(table, k, c, legend=False, ax=None):
    # compute umap
    embeddings = UMAP(n_components=2).fit_transform(list(table[k]))
    # plot
    sns.scatterplot(
        x=embeddings[:, 0],
        y=embeddings[:, 1],
        hue=table[c].astype("category"),
        ax=ax,
        alpha=0.7,
        legend=legend,
    )


def test_main():
    main(max_embeddings=20, n_subjects=2)


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
