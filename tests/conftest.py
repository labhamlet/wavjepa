import pytest
import torch
import numpy as np
from sjepa.eeg_jepa import EegJepa


@pytest.fixture
def batch():
    return dict(
        X=torch.randn(2, 4, 100),
        ch_names=[
            ["a", "b"],
            ["b", "a"],
            ["e", "c"],
            ["d", "d"],
        ],  # 'd' and 'e' are unknown channels
        ch_locs=torch.randn(2, 4, 3),
    )


@pytest.fixture
def skorch_X_y_ch_names(batch):
    X = np.random.randn(200, 4, 100).astype("float32")
    y = np.random.randint(0, 3, size=200)
    ch_names = ["a", "b", "c", "d"]
    return X, y, ch_names


@pytest.fixture(scope="function")
def model() -> EegJepa:
    feature_encoder_kwargs = dict(
        conv_layers_spec=[
            (14, 10, 1),
            (14, 5, 2),
        ]
    )  # time 100-> 44
    pos_encoder_kwargs = dict(
        spat_dim=6,
        time_dim=8,
        ch_names=["a", "b", "c"],
        ch_locs=[[1, 2, 0], [0, 1, 0], [-1, 1, 1]],
        sfreq_features=64,
    )
    mask_maker_kwargs = dict(
        n_contexts_per_input=6,
        n_targets_per_context=4,
        chs_n_unmasked=2,
        chs_n_masked=2,
        time_n_unmasked=22,
        time_n_ctx_blk=2,
        time_width_ctx_blk=10,
        time_width_tgt_blk=5,
    )
    model = EegJepa(
        feature_encoder_kwargs=feature_encoder_kwargs,
        pos_encoder_kwargs=pos_encoder_kwargs,
        mask_maker_kwargs=mask_maker_kwargs,
        transformer_kwargs=dict(
            d_model=14,
            num_encoder_layers=3,
            num_decoder_layers=1,
            nhead=7,
        ),
        average_top_k_layers=2,
        metadata_keys=["labels"],
    )
    return model


@pytest.fixture(scope="function")
def pretrained_model(model, batch) -> EegJepa:
    # initialize lazy module:
    _ = model(batch)
    return model
