from math import prod

import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn

from .audio_extractor import Extractor


class ConvChannelFeatureExtractor(Extractor, nn.Module):
    """
    Convolutional feature encoder for the audio data.

    Computes successive 1D convolutions (with activations) over the time
    dimension of the audio signal. This encoder also uses different kernels for each time signal.
    Therefore, in_channels argument is necessary!

    Inspiration from https://github.com/facebookresearch/fairseq/blob/main/fairseq/models/wav2vec/wav2vec2.py
    and https://github.com/SPOClab-ca/BENDR/blob/main/dn3_ext.py

    Args:
        conv_layers_spec: list of tuples (dim, k, stride) where:
            * dim: number of output channels of the layer (unrelated to EEG channels);
            * k: temporal length of the layer's kernel;
            * stride: temporal stride of the layer's kernel.

        in_channels: int
            Number of audio channels.
        dropout: float
        mode: str
            Normalisation mode. Either``default`` or ``layer_norm``.
        conv_bias: bool
        depthwise: bool
            Perform depthwise convolutions rather than the full convolution.
    """

    def __init__(
        self,
        *args,
        conv_layers_spec: list[tuple[int, int, int]],
        in_channels: int = 2,
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        depthwise: bool = False,
        share_weights_over_channels: bool = False,
        **kwargs,
    ):
        assert mode in {"default", "layer_norm"}
        super().__init__()  # type: ignore

        def block(
            n_in: int,
            n_out: int,
            k: int,
            stride: int,
            is_layer_norm: bool = False,
            is_group_norm: bool = False,
            conv_bias: bool = False,
            depthwise: bool = True,
        ):
            def make_conv():
                if depthwise:
                    assert n_out % n_in == 0, (
                        f"For depthwise signals we can not have non-multipler of {n_out} and {n_in}"
                    )
                    conv = nn.Conv1d(
                        n_in, n_out, k, stride=stride, bias=conv_bias, groups=n_in
                    )
                else:
                    conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)

                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert not (is_layer_norm and is_group_norm), (
                "layer norm and group norm are exclusive"
            )

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        Rearrange("... channels time -> ... time channels"),
                        nn.LayerNorm(
                            n_out, elementwise_affine=True
                        ),  # Fixed: use n_out instead of dim
                        Rearrange("... time channels -> ... channels time"),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.GroupNorm(
                        n_out, n_out, affine=True
                    ),  # Fixed: use n_out instead of dim
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

        self.in_channels = in_channels
        self.depthwise = depthwise
        self.conv_layers_spec = conv_layers_spec
        self.cnns = nn.ModuleList()

        if share_weights_over_channels:
            in_d = 1
            conv_layers = []
            for i, cl in enumerate(conv_layers_spec):
                assert len(cl) == 3, "invalid conv definition: " + str(cl)
                (dim, k, stride) = cl
                conv_layers.append(  # type: ignore
                    block(
                        in_d,
                        dim,
                        k,
                        stride,
                        is_layer_norm=mode == "layer_norm",
                        is_group_norm=mode == "default" and i == 0,
                        conv_bias=conv_bias,
                        depthwise=self.depthwise,
                    )
                )
                in_d = dim
            cnn: nn.Module = nn.Sequential(*conv_layers)  # type: ignore
            self.embedding_dim = conv_layers_spec[-1][0]
            self.cnns.append(cnn)
        else:
            for channels in range(self.in_channels):
                in_d = 1
                conv_layers = []
                for i, cl in enumerate(conv_layers_spec):
                    assert len(cl) == 3, "invalid conv definition: " + str(cl)
                    (dim, k, stride) = cl
                    conv_layers.append(  # type: ignore
                        block(
                            in_d,
                            dim,
                            k,
                            stride,
                            is_layer_norm=mode == "layer_norm",
                            is_group_norm=mode == "default" and i == 0,
                            conv_bias=conv_bias,
                            depthwise=self.depthwise,
                        )
                    )
                    in_d = dim
                cnn: nn.Module = nn.Sequential(*conv_layers)  # type: ignore
                self.cnns.append(cnn)

        self.embedding_dim = self.conv_layers_spec[-1][0]
        self.weight_sharing = share_weights_over_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, n_chans, n_times)
                    Batched EEG signal.

        Returns:
            local_features: (batch_size, emb_dim, n_times_out)
                Local features extracted from the audio signal.
                ``emb_dim`` corresponds to the ``dim`` of the last element of
                ``conv_layers_spec``.
        """

        out = []
        for channel_index in range(self.in_channels):
            # If we are sharing weights over the channels, use one CNN for all the channel dimensions
            if self.weight_sharing:
                module = self.cnns[0]
            else:
                module = self.cnns[channel_index]
            processed = module(x[:, [channel_index], ...])
            processed = rearrange(
                processed,
                "batch_size n_channels n_time -> batch_size n_time n_channels",
            )
            out.append(processed)
        processed = torch.stack(out, dim=1)
        processed = torch.flatten(processed, start_dim=1, end_dim=2)
        return processed

    def total_patches(self, time: int) -> int:
        """Calculate the number of output time steps for a given input length."""
        x = torch.zeros((1, self.in_channels, time))
        processed = self.forward(x)
        print(processed.shape)
        return processed.shape[1]  # Return time dimension size

    @property
    def receptive_fields(self) -> list[int]:
        rf = 1
        receptive_fields = [rf]
        for _, width, stride in reversed(self.conv_layers_spec):
            rf = (rf - 1) * stride + width  # assumes no padding and no dilation
            receptive_fields.append(rf)
        return list(reversed(receptive_fields))

    def description(
        self, sfreq: int | None = None, dummy_time: int | None = None
    ) -> str:
        dims, _, strides = zip(*self.conv_layers_spec)
        receptive_fields = self.receptive_fields
        rf = receptive_fields[0]
        desc = f"Receptive field: {rf} samples"
        if sfreq is not None:
            desc += f", {rf / sfreq:.2f} seconds"

        ds_factor = prod(strides)
        desc += f" | Downsampled by {ds_factor}"
        if sfreq is not None:
            desc += f", new sfreq: {sfreq / ds_factor:.2f} Hz"
        desc += f" | Overlap of {rf - ds_factor} samples"
        if dummy_time is not None:
            n_times_out = self.total_patches(dummy_time)
            desc += f" | {n_times_out} encoded samples/trial"

        n_features = [
            f"{dim}*{rf}"
            for dim, rf in zip([self.in_channels] + list(dims), receptive_fields)
        ]
        desc += f" | #features/sample at each layer (n_channels*n_times): [{', '.join(n_features)}] = {[eval(x) for x in n_features]}"
        return desc
