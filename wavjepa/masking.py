from torch import nn
from einops import repeat, rearrange
import torch

from wavjepa.audio_masking import compute_mask_indices

class AudioMasker(nn.Module):
    def __init__(
        self,
        target_masks_per_context : int = 4,
        context_prob : float = 0.65, 
        context_length : int = 10,
        target_prob : float = 0.25,
        target_length : int = 10,
        ratio_cutoff : float = 0.1,
        min_context_len: int = 1,
        channel_based_masking : bool = False,
        **kwargs,
    ):
        super().__init__()
        self.target_masks_per_context = target_masks_per_context
        self.target_prob = target_prob
        self.target_length = target_length
        self.ratio_cutoff = ratio_cutoff
        self.channel_based_masking = channel_based_masking
        self.min_context_len = min_context_len
        self.context_prob = context_prob 
        self.context_length = context_length

    def filter_small_clusters(self, mask: torch.Tensor):
        """
        Identifies contiguous blocks of True values. 
        If a block is shorter than min_context_len, it is set to False.
        """
        # unique_consecutive returns the values and the length of each run
        values, counts = torch.unique_consecutive(mask, return_counts=True)
        
        # Identify segments that are 'Context' (True) but too short
        small_context_mask = (values) & (counts < self.min_context_len)
        
        # Flip those specific segments to False
        values[small_context_mask] = False
        
        # Reconstruct the full length tensor
        return torch.repeat_interleave(values, counts)

    def forward(self, batch_size: int , n_times: int, in_channels: int):
        n_times = n_times // in_channels 
        target_positions = torch.zeros([batch_size, self.target_masks_per_context, n_times], dtype=torch.bool)
        context_positions = torch.zeros([batch_size, n_times], dtype=torch.bool)

        for batch_idx in range(batch_size):
            while True:
                #Find the non context indices.
                context = ~compute_mask_indices(
                                shape = (1, n_times),
                                padding_mask = None,
                                mask_prob = self.context_prob,
                                mask_length = self.context_length,
                              )
                target_positions_ = torch.zeros([self.target_masks_per_context, n_times], dtype=torch.bool)
                for target_group in range(self.target_masks_per_context):
                    _target_positions_ = compute_mask_indices(
                                shape = (1, n_times),
                                padding_mask = None,
                                mask_prob = self.target_prob,
                                mask_length = self.target_length,
                              )
                    target_positions_[target_group] = _target_positions_
                
                any_target_at_position = torch.any(target_positions_, dim=0)
                context_positions_ = ~any_target_at_position & context

                context_positions_ = self.filter_small_clusters(context_positions_)

                ratio = torch.sum(context_positions_) / n_times
                if ratio >= self.ratio_cutoff:
                    break
            
            target_positions[batch_idx] = target_positions_
            context_positions[batch_idx] = context_positions_

        final_context_mask = ~context_positions 
        combined_visible_mask = torch.logical_xor(final_context_mask.unsqueeze(1), target_positions)

        if self.channel_based_masking:
            final_context_mask = rearrange(repeat(final_context_mask, "B S -> B C S", C = in_channels),
                                          "B C S -> B (S C)")
            target_positions = rearrange(repeat(target_positions, "B N S -> B C N S", C = in_channels),
                                              "B C N S -> B N (S C)")
            combined_visible_mask = rearrange(repeat(combined_visible_mask, "B N S -> B C N S", C = in_channels),
                                              "B C N S -> B N (S C)")

        return final_context_mask, target_positions, combined_visible_mask.to(torch.bool)
    
class SpeechMasker(nn.Module):
    def __init__(
        self,
        target_masks_per_context : int = 4,
        target_prob : float = 0.25,
        target_length : int = 5,
        ratio_cutoff : float = 0.3,
        min_context_len: int = 5,
        channel_based_masking : bool = False,
        **kwargs,
    ):
        super().__init__()
        self.target_masks_per_context = target_masks_per_context
        self.target_prob = target_prob
        self.target_length = target_length
        self.ratio_cutoff = ratio_cutoff
        self.channel_based_masking = channel_based_masking
        self.min_context_len = min_context_len

    def filter_small_clusters(self, mask: torch.Tensor):
        """
        Identifies contiguous blocks of True values. 
        If a block is shorter than min_context_len, it is set to False.
        """
        # unique_consecutive returns the values and the length of each run
        values, counts = torch.unique_consecutive(mask, return_counts=True)
        
        # Identify segments that are 'Context' (True) but too short
        small_context_mask = (values) & (counts < self.min_context_len)
        
        # Flip those specific segments to False
        values[small_context_mask] = False
        
        # Reconstruct the full length tensor
        return torch.repeat_interleave(values, counts)

    def forward(self, batch_size: int , n_times: int, in_channels: int):
        n_times = n_times // in_channels 
        target_positions = torch.zeros([batch_size, self.target_masks_per_context, n_times], dtype=torch.bool)
        context_positions = torch.ones([batch_size, n_times], dtype=torch.bool)

        for batch_idx in range(batch_size):
            while True:
                target_positions_ = torch.zeros([self.target_masks_per_context, n_times], dtype=torch.bool)
                for target_group in range(self.target_masks_per_context):
                    _target_positions_ = compute_mask_indices(
                                shape = (1, n_times),
                                padding_mask = None,
                                mask_prob = self.target_prob,
                                mask_length = self.target_length,
                              )
                    target_positions_[target_group] = _target_positions_
                
                any_target_at_position = torch.any(target_positions_, dim=0)
                context_positions_ = ~any_target_at_position

                context_positions_ = self.filter_small_clusters(context_positions_)

                ratio = torch.sum(context_positions_) / n_times
                if ratio >= self.ratio_cutoff:
                    break
            
            target_positions[batch_idx] = target_positions_
            context_positions[batch_idx] = context_positions_

        final_context_mask = ~context_positions 
        combined_visible_mask = torch.logical_xor(final_context_mask.unsqueeze(1), target_positions)

        if self.channel_based_masking:
            final_context_mask = rearrange(repeat(final_context_mask, "B S -> B C S", C = in_channels),
                                          "B C S -> B (S C)")
            target_positions = rearrange(repeat(target_positions, "B N S -> B C N S", C = in_channels),
                                              "B C N S -> B N (S C)")
            combined_visible_mask = rearrange(repeat(combined_visible_mask, "B N S -> B C N S", C = in_channels),
                                              "B C N S -> B N (S C)")

        return final_context_mask, target_positions, combined_visible_mask.to(torch.bool)