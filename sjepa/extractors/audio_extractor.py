from abc import ABC, abstractmethod

import torch


class Extractor(ABC):
    """Abstract base class for encoders."""

    # Just declare that implementers should have this attribute
    embedding_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the encoder."""
        pass

    @abstractmethod
    def total_patches(self, time: int) -> int:
        """Returns the total patches given the time dimension of the input."""
        pass
