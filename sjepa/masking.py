from random import randrange

import torch
from einops import rearrange, repeat
from torch import nn

from sjepa.audio_masking import (
    compute_mask_indices,
    gen_maskid_patch,
    gen_maskid_patch_tgt,
    gen_targetid_cluster,
    gen_targetid_random,
)


class TimeInverseBlockMasker(nn.Module):
    """
    Mask maker for EEG data.

    Uses the :func:`channels_block_masking` followed by :func:`random_masking`
    to mask the channels.
    Uses :func:`time_inverse_block_masking` to mask the time samples.

    Args:
        n_contexts_per_input: int
            Number of context masks to generate per input example.
        n_targets_per_context: int
            Number of target masks to generate per context mask.
        chs_radius_blocks: float
            Radius of the masking blocks to use for channel masking.
        chs_n_blocks_masked: int
            Number of masking blocks to generate for the channel masking.
        chs_n_unmasked: int
            Number of channels to leave unmasked.
        chs_n_masked: int
            Number of channels to use in each target.
        time_n_unmasked: int
            Number of time samples to leave unmasked.
        # time_unmasked_width: int
        #     Width of the unmasked blocks to use for time masking.
        time_n_ctx_blk: int
            Number of context blocks for the temporal masking.
        time_width_tgt_blk: int
            Width of the target blocks for the temporal masking.
        time_width_ctx_blk: int
            Width of the context blocks for the temporal masking.
        # time_exact: bool
        #     Whether to leave exactly ``ch_n_unmasked`` unmasked elements in
        #     the time dimension. If true, ``time_unmasked_width`` may be silently violated.
        return_indices: bool
            Whether to return the indices of the masked elements.
            Requires ``time_exact`` to be True.
    """

    def __init__(
        self,
        target_masks_per_context: int = 4,
        context_mask_prob: float = 0.3,
        context_mask_length: int = 10,
        target_prob: float = 0.2,
        target_length: int = 20,
        ratio_cutoff: float = 0.05,
        channel_based_masking: bool = False,
        **kwargs,
    ):
        super().__init__()  # type: ignore
        self.target_masks_per_context = target_masks_per_context
        self.context_mask_prob = context_mask_prob
        self.context_mask_length = context_mask_length
        self.target_prob = target_prob
        self.target_length = target_length
        self.ratio_cutoff = ratio_cutoff
        self.channel_based_masking = channel_based_masking

    def forward(self, batch_size: int, n_times: int, in_channels: int):
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
        n_times = (
            n_times // in_channels
        )  # Because our extractor flattens the channels, acutual n_times is divided by in_channels
        target_positions = torch.zeros(
            [batch_size, self.target_masks_per_context, n_times], dtype=torch.bool
        )
        context_positions = torch.zeros([batch_size, n_times], dtype=torch.bool)
        for batch_idx in range(batch_size):
            target_positions_ = torch.zeros(
                [self.target_masks_per_context, n_times], dtype=torch.bool
            )
            while True:
                # Non masked parts are targets
                context_positions_ = ~compute_mask_indices(
                    shape=(1, n_times),
                    padding_mask=None,
                    mask_prob=self.context_mask_prob,
                    mask_length=self.context_mask_length,
                )
                for target_group in range(self.target_masks_per_context):
                    # Generate target region indices
                    _target_positions_ = compute_mask_indices(
                        shape=(1, n_times),
                        padding_mask=None,
                        mask_prob=self.target_prob,
                        mask_length=self.target_length,
                    )
                    # Mark these positions as targets
                    target_positions_[target_group] = _target_positions_
                any_target_at_position = torch.any(target_positions_, dim=0)
                context_positions_ = context_positions_ & ~any_target_at_position
                ratio = torch.sum(context_positions_) / n_times
                if ratio >= self.ratio_cutoff:
                    break
            target_positions[batch_idx] = target_positions_
            context_positions[batch_idx] = context_positions_

        final_context_mask = ~context_positions  # True = masked context
        combined_visible_mask = torch.logical_xor(
            final_context_mask.unsqueeze(1), target_positions
        )

        # Channel based masking repeats the mask for the other channel, and then flattens it
        # This assumes that our extractor also flattens the channels. Thus, we will mask the same time points.
        if self.channel_based_masking:
            final_context_mask = rearrange(
                repeat(final_context_mask, "B S -> B C S", C=in_channels),
                "B C S -> B (S C)",
            )
            target_positions = rearrange(
                repeat(target_positions, "B N S -> B C N S", C=in_channels),
                "B C N S -> B N (S C)",
            )
            combined_visible_mask = rearrange(
                repeat(combined_visible_mask, "B N S -> B C N S", C=in_channels),
                "B C N S -> B N (S C)",
            )

        return (
            final_context_mask,
            target_positions,
            combined_visible_mask.to(torch.bool),
        )


class MultiBlockMaskMaker(nn.Module):
    """
    Mask maker for EEG data.

    Uses the :func:`channels_block_masking` followed by :func:`random_masking`
    to mask the channels.
    Uses :func:`time_inverse_block_masking` to mask the time samples.

    Args:
        n_contexts_per_input: int
            Number of context masks to generate per input example.
        n_targets_per_context: int
            Number of target masks to generate per context mask.
        chs_radius_blocks: float
            Radius of the masking blocks to use for channel masking.
        chs_n_blocks_masked: int
            Number of masking blocks to generate for the channel masking.
        chs_n_unmasked: int
            Number of channels to leave unmasked.
        chs_n_masked: int
            Number of channels to use in each target.
        time_n_unmasked: int
            Number of time samples to leave unmasked.
        # time_unmasked_width: int
        #     Width of the unmasked blocks to use for time masking.
        time_n_ctx_blk: int
            Number of context blocks for the temporal masking.
        time_width_tgt_blk: int
            Width of the target blocks for the temporal masking.
        time_width_ctx_blk: int
            Width of the context blocks for the temporal masking.
        # time_exact: bool
        #     Whether to leave exactly ``ch_n_unmasked`` unmasked elements in
        #     the time dimension. If true, ``time_unmasked_width`` may be silently violated.
        return_indices: bool
            Whether to return the indices of the masked elements.
            Requires ``time_exact`` to be True.
    """

    def __init__(
        self,
        target_masks_per_context: int = 4,
        context_cluster_d: float = 0.85,
        context_cluster_u: float = 1.0,
        target_cluster_d: float = 0.15,
        target_cluster_u: float = 0.2,
        ratio_cutoff: float = 0.05,
        channel_based_masking: bool = False,
        **kwargs,
    ):
        super().__init__()  # type: ignore
        self.target_masks_per_context = target_masks_per_context
        self.context_cluster_d = context_cluster_d
        self.context_cluster_u = context_cluster_u
        self.target_cluster_d = target_cluster_d
        self.target_cluster_u = target_cluster_u
        self.channel_based_masking = channel_based_masking
        self.ratio_cutoff = ratio_cutoff

    def forward(self, batch_size: int, n_times: int, in_channels: int):
        """
        Generate masks for JEPA (Joint-Embedding Predictive Architecture) training.

        This function creates:
        1. Context regions: visible tokens used for encoding
        2. Target regions: masked tokens to be predicted
        3. Attention masks: control what tokens can attend to what

        Args:
            batch_size: Number of samples in the batch
            n_times: Total time points after feature extraction
            in_channels: Number of input channels (unused)

        Returns:
            tuple containing:
            - context_mask: (batch_size, n_times)
                True = masked context, False = visible context
            - target_mask: (batch_size, n_target_groups, n_times)
                True = target tokens to predict, False = not targets
            - context_attention_mask: (batch_size, n_times, n_times)
                True = can attend, False = cannot attend
            - target_attention_mask: (batch_size, n_target_groups, n_times, n_times)
                True = can attend, False = cannot attend
        """

        # Step 1: Initialize tracking tensors
        # These track which positions are targets and contexts (True = selected context and target tokens)

        target_positions = torch.zeros(
            [batch_size, self.target_masks_per_context, n_times], dtype=torch.bool
        )
        context_positions = torch.zeros([batch_size, n_times], dtype=torch.bool)

        for batch_idx in range(batch_size):
            while True:
                target_positions_ = torch.zeros(
                    [self.target_masks_per_context, n_times], dtype=torch.bool
                )
                context_positions_ = torch.zeros([n_times], dtype=torch.bool)
                for target_group in range(self.target_masks_per_context):
                    # Generate target region indices
                    target_indices = gen_targetid_cluster(
                        n_times=n_times,
                        cluster_d=int(n_times * self.target_cluster_d),
                        cluster_u=int(n_times * self.target_cluster_u),
                    )

                    # Mark these positions as targets
                    target_positions_[target_group, target_indices] = True

                # Step 3b: Sample context region (visible tokens for encoding)
                context_indices = gen_targetid_cluster(
                    n_times=n_times,
                    cluster_d=int(n_times * self.context_cluster_d),
                    cluster_u=int(n_times * self.context_cluster_u),
                )
                # Visible tokens are assigned True
                context_positions_[context_indices] = True

                # Step 3c: Remove overlap between context and targets
                # If a position is already a target, it cannot be context
                any_target_at_position = torch.any(target_positions_, dim=0)
                context_positions_ = context_positions_ & ~any_target_at_position
                ratio = torch.sum(context_positions_) / n_times
                if ratio >= self.ratio_cutoff:
                    break
            target_positions[batch_idx] = target_positions_
            context_positions[batch_idx] = context_positions_

        final_context_mask = ~context_positions  # True = masked context
        combined_visible_mask = torch.logical_xor(
            final_context_mask.unsqueeze(1), target_positions
        )
        return (
            final_context_mask,
            target_positions,
            combined_visible_mask.to(torch.bool),
        )


# This is more useful for data2vec
class RandomMaskMaker(nn.Module):
    """
    Mask maker for EEG data.

    Uses the :func:`channels_block_masking` followed by :func:`random_masking`
    to mask the channels.
    Uses :func:`time_inverse_block_masking` to mask the time samples.

    Args:
        n_contexts_per_input: int
            Number of context masks to generate per input example.
        n_targets_per_context: int
            Number of target masks to generate per context mask.
        chs_radius_blocks: float
            Radius of the masking blocks to use for channel masking.
        chs_n_blocks_masked: int
            Number of masking blocks to generate for the channel masking.
        chs_n_unmasked: int
            Number of channels to leave unmasked.
        chs_n_masked: int
            Number of channels to use in each target.
        time_n_unmasked: int
            Number of time samples to leave unmasked.
        # time_unmasked_width: int
        #     Width of the unmasked blocks to use for time masking.
        time_n_ctx_blk: int
            Number of context blocks for the temporal masking.
        time_width_tgt_blk: int
            Width of the target blocks for the temporal masking.
        time_width_ctx_blk: int
            Width of the context blocks for the temporal masking.
        # time_exact: bool
        #     Whether to leave exactly ``ch_n_unmasked`` unmasked elements in
        #     the time dimension. If true, ``time_unmasked_width`` may be silently violated.
        return_indices: bool
            Whether to return the indices of the masked elements.
            Requires ``time_exact`` to be True.
    """

    def __init__(
        self,
        target_masks_per_context: int = 4,
        context_cluster_d: float = 0.1,
        context_cluster_u: float = 0.25,
        target_cluster_d: float = 0.7,
        target_cluster_u: float = 0.85,
        channel_based_masking: bool = False,
        **kwargs,
    ):
        super().__init__()  # type: ignore
        self.target_masks_per_context = target_masks_per_context
        self.context_cluster_d = context_cluster_d
        self.context_cluster_u = context_cluster_u
        self.target_cluster_d = target_cluster_d
        self.target_cluster_u = target_cluster_u
        self.channel_based_masking = channel_based_masking

    def forward(self, batch_size: int, n_times: int, in_channels: int):
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
        target_positions = torch.zeros(
            [batch_size, self.target_masks_per_context, n_times], dtype=torch.bool
        )
        context_positions = torch.zeros([batch_size, n_times], dtype=torch.bool)

        for batch_idx in range(batch_size):
            # We sample it with a really low probability. [0.75, 0.9] are masked basically
            context_indices = gen_targetid_random(
                n_times=n_times,
                cluster_d=int(n_times * self.context_cluster_d),
                cluster_u=int(n_times * self.context_cluster_u),
            )
            context_positions[batch_idx, context_indices] = True

            ratio = len(context_indices) / n_times
            for target_group in range(self.target_masks_per_context):
                # Generate target region indices
                target_indices = gen_targetid_random(
                    n_times=n_times,
                    cluster_d=int(n_times * self.target_cluster_d),
                    cluster_u=min(
                        int(n_times * self.target_cluster_u),
                        int(n_times - ratio * n_times),
                    ),
                    selected_tokens=context_positions[batch_idx],
                )

                # Mark these positions as targets
                target_positions[batch_idx, target_group, target_indices] = True

        final_context_mask = ~context_positions  # True = masked context

        combined_visible_mask = torch.logical_xor(
            final_context_mask.unsqueeze(1), target_positions
        )
        return (
            final_context_mask,
            target_positions,
            combined_visible_mask.to(torch.bool),
        )


class RandomClusterMaskMaker(nn.Module):
    """
    Mask maker for EEG data.

    Uses the :func:`channels_block_masking` followed by :func:`random_masking`
    to mask the channels.
    Uses :func:`time_inverse_block_masking` to mask the time samples.

    Args:
        n_contexts_per_input: int
            Number of context masks to generate per input example.
        n_targets_per_context: int
            Number of target masks to generate per context mask.
        chs_radius_blocks: float
            Radius of the masking blocks to use for channel masking.
        chs_n_blocks_masked: int
            Number of masking blocks to generate for the channel masking.
        chs_n_unmasked: int
            Number of channels to leave unmasked.
        chs_n_masked: int
            Number of channels to use in each target.
        time_n_unmasked: int
            Number of time samples to leave unmasked.
        # time_unmasked_width: int
        #     Width of the unmasked blocks to use for time masking.
        time_n_ctx_blk: int
            Number of context blocks for the temporal masking.
        time_width_tgt_blk: int
            Width of the target blocks for the temporal masking.
        time_width_ctx_blk: int
            Width of the context blocks for the temporal masking.
        # time_exact: bool
        #     Whether to leave exactly ``ch_n_unmasked`` unmasked elements in
        #     the time dimension. If true, ``time_unmasked_width`` may be silently violated.
        return_indices: bool
            Whether to return the indices of the masked elements.
            Requires ``time_exact`` to be True.
    """

    def __init__(
        self,
        target_masks_per_context: int = 4,
        context_cluster_d: float = 0.2,
        context_cluster_u: float = 0.3,
        target_cluster_d: float = 0.2,
        target_cluster_u: float = 0.3,
        channel_based_masking: bool = False,
        **kwargs,
    ):
        super().__init__()  # type: ignore
        self.target_masks_per_context = target_masks_per_context
        self.context_cluster_d = context_cluster_d
        self.context_cluster_u = context_cluster_u
        self.target_cluster_d = target_cluster_d
        self.target_cluster_u = target_cluster_u
        self.channel_based_masking = channel_based_masking

    def forward(self, batch_size: int, n_times: int, in_channels: int):
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
        target_positions = torch.zeros(
            [batch_size, self.target_masks_per_context, n_times], dtype=torch.bool
        )
        context_positions = torch.zeros([batch_size, n_times], dtype=torch.bool)

        for batch_idx in range(batch_size):
            target_positions_ = torch.zeros(
                [self.target_masks_per_context, n_times], dtype=torch.bool
            )
            context_positions_ = torch.zeros([n_times], dtype=torch.bool)
            total_context_idx = randrange(
                int(n_times * self.context_cluster_d),
                int(n_times * self.context_cluster_u),
            )
            context_indices = gen_maskid_patch(n_times, total_context_idx, 5)
            # These are our context positions
            context_positions_[context_indices] = True

            for target_group in range(self.target_masks_per_context):
                # Generate target region indices
                total_context_idx = randrange(
                    int(n_times * self.target_cluster_d),
                    int(n_times * self.target_cluster_u),
                )
                target_indices = gen_maskid_patch_tgt(
                    ~context_positions_, total_context_idx, 3
                )
                # Mark these positions as targets
                # These are our target positions
                target_positions_[target_group, target_indices] = True
            target_positions[batch_idx] = target_positions_
            context_positions[batch_idx] = context_positions_

        final_context_mask = ~context_positions  # True = masked context

        combined_visible_mask = torch.logical_xor(
            final_context_mask.unsqueeze(1), target_positions
        )
        return (
            final_context_mask,
            target_positions,
            combined_visible_mask.to(torch.bool),
        )
