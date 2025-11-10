import torch
import torchaudio
from torch import nn

from .audio_extractor import Extractor


def generate_patches(input, fstride, tstride, fshape, tshape):
    r"""Function that extract patches from tensors and stacks them.

    See :class:`~kornia.contrib.ExtractTensorPatches` for details.

    Args:
        input: tensor image where to extract the patches with shape :math:`(B, C, H, W)`.

    Returns:
        the tensor with the extracted patches with shape :math:`(B, N, C, H_{out}, W_{out})`.

    Examples:
        >>> input = torch.arange(9.).view(1, 1, 3, 3)
        >>> patches = extract_tensor_patches(input, (2, 3))
        >>> input
        tensor([[[[0., 1., 2.],
                  [3., 4., 5.],
                  [6., 7., 8.]]]])
        >>> patches[:, -1]
        tensor([[[[3., 4., 5.],
                  [6., 7., 8.]]]])

    """
    batch_size, num_channels = input.size()[:2]
    dims = range(2, input.dim())
    for dim, patch_size, stride in zip(dims, (fshape, tshape), (fstride, tstride)):
        input = input.unfold(dim, patch_size, stride)
    input = input.permute(0, *dims, 1, *(dim + len(dims) for dim in dims)).contiguous()
    return input.view(batch_size, -1, num_channels, fshape, tshape)


# get the shape of intermediate representation.
def get_shape(fstride, tstride, input_fdim, input_tdim, fshape, tshape):
    test_input = torch.randn(1, 2, input_fdim, input_tdim)
    test_proj = nn.Conv2d(
        2,
        2,
        kernel_size=(fshape, tshape),
        stride=(fstride, tstride),
    )
    test_out = test_proj(test_input)
    f_dim = test_out.shape[2]
    t_dim = test_out.shape[3]
    return f_dim, t_dim


class SpectrogramPatchExtractor(Extractor, nn.Module):
    """
    Convolutional feature encoder for EEG data.

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
        n_mels: int = 128,
        window_size: int = 1024,
        n_fft: int = 1024,
        hop_length: int = 320,
        sr: int = 32000,
        embedding_dim: int = 768,
        in_channels: int = 2,
        fshape: int = 16,
        tshape: int = 32,
        fstride: int = 16,
        tstride: int = 32,
        input_tdim: int = 200,
        trainable: bool = True,
        **kwargs,
    ):
        super().__init__()  # type: ignore
        self.fstride = fstride
        self.tstride = tstride
        self.fshape = fshape
        self.tshape = tshape
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.audio_encoder = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            win_length=window_size,
            hop_length=hop_length,
            f_min=50,
            f_max=sr // 2,
            n_mels=n_mels,
            power=2.0,
        ).float()

        self.patcher = torch.nn.Conv2d(
            self.in_channels,
            self.embedding_dim,
            kernel_size=(fshape, tshape),
            stride=(fstride, tstride),
        )

        self.input_tdim = input_tdim
        p_f_dim, p_t_dim = get_shape(
            fstride=fstride,
            tstride=tstride,
            input_fdim=n_mels,
            input_tdim=input_tdim,
            fshape=fshape,
            tshape=tshape,
        )
        self.grid_size = (p_f_dim, p_t_dim)

    def to_mel(self, x: torch.Tensor) -> torch.Tensor:
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

        mel = self.audio_encoder(x)
        spec = torch.log(mel + 1e-5)[:, :, :, : self.input_tdim]
        # B, C, F, T
        # Reshape back to separate batch and channel dimensions
        # Assuming encoder output shape is (batch_size * n_chans, out_dim, out_times)

        dims_to_normalize = [1, 2, 3]
        mean = spec.mean(dim=dims_to_normalize, keepdim=True)
        var = spec.var(dim=dims_to_normalize, keepdim=True, unbiased=False)

        # Apply normalization
        eps = 1e-5
        spec = (spec - mean) / torch.sqrt(var + eps)
        return spec

    def get_audio_segments(self, audio) -> torch.Tensor:
        """
        Segments the audio with non overlapping convolutions.
        """

        mel = self.to_mel(audio)
        patches = generate_patches(
            mel, self.fstride, self.tstride, self.fshape, self.tshape
        )
        return patches.flatten(2)

    def forward(self, audio):
        mel = self.to_mel(audio)
        patches = self.patcher(mel)
        return patches.flatten(2).transpose(1, 2)

    @property
    def patch_dim(self):
        """The radius property."""
        return self.fshape * self.tshape * self.in_channels

    def total_patches(self, time: int) -> int:
        """Calculate the number of output time steps for a given input length."""
        x = torch.zeros((1, self.in_channels, time))
        out = self.forward(x)
        return out.shape[1]  # Return time dimension size
