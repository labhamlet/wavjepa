from torch import nn
from einops import repeat, rearrange
import torch

from wavjepa.audio_masking import compute_mask_indices

class AudioMasker(nn.Module):
    """
    Mask maker for audio data.
    This masking is motivated by two aspects of speech that differs from general purpose audio. 
    1 More context is need for speech
    2.Short lengths are needed to predict the "phonems" 
    3.Context should not be sparse
    """

    def __init__(
        self,
        target_masks_per_context : int = 4,
        target_prob : float = 0.25,
        target_length : int = 5,
        ratio_cutoff : float = 0.3,
        min_conext_len: int = 5,
        channel_based_masking : bool = False,
        **kwargs,
    ):
        super().__init__() # type: ignore
        self.target_masks_per_context =  target_masks_per_context
        self.target_prob = target_prob
        self.target_length = target_length
        self.ratio_cutoff = ratio_cutoff
        self.channel_based_masking = channel_based_masking
        self.min_context_len = min_conext_len

    def forward(self, batch_size: int , n_times: int, in_channels: int):
        """
        Args:
            batch_size, n_times
            n_times is the total time points after doing the feature extraction with convolutional layer.
        Returns:
            out: tuple of:
                * masks_context: (batch_size, n_times)
                    The patch elements to use by the student  to compute the contextualised
                    representations during training.

                * masks_target: (batch_size, n_contexts_per_input, n_times)
                    The patches that must be predicted by the student during training.
                        it is a bool tensor true for the masked elements.
        """

        # These track which positions are targets and contexts (True = selected)
        n_times = n_times // in_channels #Because our extractor flattens the channels, acutual n_times is divided by in_channels
        target_positions = torch.zeros([batch_size, self.target_masks_per_context, n_times], dtype=torch.bool)
        context_positions = torch.zeros([batch_size, n_times], dtype=torch.bool)
        for batch_idx in range(batch_size):
            target_positions_ = torch.zeros([self.target_masks_per_context, n_times], dtype=torch.bool)
            while True:
              # Non masked parts are targets
              context_positions_ = compute_mask_indices(
                shape = (1, n_times),
                padding_mask = None,
                mask_prob = 1.0,
                mask_length = 1, #Does not matter
                min_masks = n_times
              )
              for target_group in range(self.target_masks_per_context):
                  # Generate target region indices
                  _target_positions_ = compute_mask_indices(
                                shape = (1, n_times),
                                padding_mask = None,
                                mask_prob = self.target_prob,
                                mask_length = self.target_length,
                              )
                  # Mark these positions as targets
                  target_positions_[target_group] = _target_positions_
              any_target_at_position = torch.any(target_positions_, dim=0)
              context_positions_ = context_positions_ & ~any_target_at_position
              vals, counts = torch.unique_consecutive(context_positions_, return_counts=True)
              clean_segments = []
              for v, c in zip(vals, counts):
                  length = c.item()
                  # If it is Context (True) AND shorter than 5...
                  if v and length < self.min_context_len:
                      # ... replace it masked
                      clean_segments.append(torch.zeros(length, dtype=torch.bool))
                  else:
                      # Otherwise keep it as is
                      clean_segments.append(torch.full((length,), v, dtype=torch.bool))
              
              # Reassemble the cleaned context
              context_positions_ = torch.cat(clean_segments)

              ratio = torch.sum(context_positions_) / n_times
              if ratio >= self.ratio_cutoff:
                break
            target_positions[batch_idx] = target_positions_
            context_positions[batch_idx] = context_positions_


        final_context_mask = ~context_positions  # True = masked context
        combined_visible_mask = torch.logical_xor(final_context_mask.unsqueeze(1), target_positions)

        # Channel based masking repeats the mask for the other channel, and then flattens it
        # This assumes that our extractor also flattens the channels. Thus, we will mask the same time points.
        if self.channel_based_masking:
            final_context_mask = rearrange(repeat(final_context_mask, "B S -> B C S", C = in_channels),
                                          "B C S -> B (S C)")
            target_positions = rearrange(repeat(target_positions, "B N S -> B C N S", C = in_channels),
                                              "B C N S -> B N (S C)")
            combined_visible_mask = rearrange(repeat(combined_visible_mask, "B N S -> B C N S", C = in_channels),
                                              "B C N S -> B N (S C)")

        return final_context_mask, target_positions, combined_visible_mask.to(torch.bool)
    
