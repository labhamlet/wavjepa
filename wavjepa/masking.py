from torch import nn
from einops import repeat, rearrange
import torch

from wavjepa.audio_masking import compute_mask_indices

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
        target_masks_per_context : int = 4,
        context_mask_prob : float = 0.3,
        context_mask_length : int = 10,
        target_prob : float = 0.2,
        target_length : int = 20,
        ratio_cutoff : float = 0.05,
        channel_based_masking : bool = False,
        **kwargs,
    ):
        super().__init__() # type: ignore
        self.target_masks_per_context =  target_masks_per_context
        self.context_mask_prob = context_mask_prob 
        self.context_mask_length = context_mask_length
        self.target_prob = target_prob
        self.target_length = target_length
        self.ratio_cutoff = ratio_cutoff
        self.channel_based_masking = channel_based_masking

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
              context_positions_ = ~compute_mask_indices(
                shape = (1, n_times),
                padding_mask = None,
                mask_prob = self.context_mask_prob,
                mask_length = self.context_mask_length,
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

    # ------------------------------------------------------------------ #
    # Vectorised path (same distribution as ``forward``, no loop over rows)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _sample_spans(
        batch_size: int,
        sz: int,
        mask_prob: float,
        mask_length: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Vectorised equivalent of ``batch_size`` independent calls of
        ``compute_mask_indices(shape=(1, sz), padding_mask=None, mask_prob, mask_length)``
        with its defaults (``mask_type="static"``, ``num_mask_ver=2``,
        ``idc_select_ver=1``, ``no_overlap=False``, ``min_masks=0``,
        ``require_same_masks=True`` and ``mask_dropout=0`` -- both no-ops for bsz=1).

        The per-row process being replicated (``p=mask_prob``, ``l=mask_length``):

        1. ``a = p * sz / float(l)`` (float64);
           ``num_mask = int(a + U[0,1))`` i.e. ``floor(a)`` or ``floor(a)+1`` with
           ``P(+1) = frac(a)``; ``max(min_masks=0, .)`` is a no-op.
        2. ``lengths = [l] * num_mask``; if ``num_mask == 0`` the reference raises
           ``ValueError("this should never happens")`` -- we raise the same.
        3. ``min_len = l``; ``if sz - min_len <= num_mask: min_len = sz - num_mask - 1``
           (note this depends on the *per-row* ``num_mask``, so ``n = sz - min_len``
           differs between rows that drew ``floor(a)`` and ``floor(a)+1``).
        4. ``starts = rng.choice(sz - min_len, num_mask, replace=False)``: a uniform
           ``num_mask``-subset of ``[0, sz - min_len)``.
        5. positions ``= {s + o : s in starts, o in [0, l)}``, ``np.unique``, keep ``< sz``.
           Only the *set* matters, so "uniform k-subset" is all we need from step 4.

        Vectorisation: draw ``k_hi = floor(a)+1`` distinct starts per row as the first
        ``k_hi`` entries of ``argsort(rand(B, n_alloc))`` (the prefix of a uniform random
        permutation of ``[0, n_b)`` is a uniform subset); columns ``>= n_b`` are pushed
        to the end of the sort with a key of 2.0 so per-row ``n_b`` is respected; the
        per-row count ``k_b`` is applied as ``valid = (j < k_b) & (pos < sz)`` and the
        valid positions are scattered into a ``(B, sz)`` bool tensor.

        Returns:
            (batch_size, sz) bool, True = inside a span.
        """
        B, l = int(batch_size), int(mask_length)
        # identical float64 expression to compute_mask_indices
        a = mask_prob * sz / float(l)
        k_lo = int(a)                          # floor for a >= 0
        k_hi = k_lo + 1 if a > k_lo else k_lo  # floor(a + u) can only be k_lo when a is integral

        u = torch.rand(B, dtype=torch.float64, device=device, generator=generator)
        k = torch.floor(a + u).to(torch.int64)  # (B,) == int(a + rng.random()) per row
        if k_lo == 0 and bool((k == 0).any()):
            raise ValueError(
                "this should never happens"  # same error as compute_mask_indices
                f" (num_mask=0 drawn for mask_prob={mask_prob}, mask_length={l}, sz={sz})"
            )

        # min_len adjustment -> number of admissible start positions per row
        n = torch.where(sz - l > k, torch.full_like(k, sz - l), k + 1)   # (B,)
        n_alloc = max(sz - l, k_hi + 1)                                  # >= n_b for every row

        keys = torch.rand(B, n_alloc, dtype=torch.float64, device=device, generator=generator)
        col = torch.arange(n_alloc, device=device)
        keys = keys.masked_fill(col[None, :] >= n[:, None], 2.0)  # out-of-range columns sort last
        starts = keys.argsort(dim=1)[:, :k_hi]                    # (B, k_hi) distinct, uniform in [0, n_b)

        pos = starts[:, :, None] + torch.arange(l, device=device)[None, None, :]   # (B, k_hi, l)
        valid = (torch.arange(k_hi, device=device)[None, :, None] < k[:, None, None]) & (pos < sz)
        idx = torch.where(valid, pos, torch.full_like(pos, sz)).reshape(B, -1)  # dummy column sz
        out = torch.zeros(B, sz + 1, dtype=torch.bool, device=device)
        out.scatter_(1, idx, torch.ones_like(idx, dtype=torch.bool))
        return out[:, :sz]

    def _draw_rows(
        self, n_rows: int, sz: int, device: torch.device, generator: torch.Generator | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One unconditioned draw of ``n_rows`` (context_positions, target_positions, ok)."""
        context_positions = ~self._sample_spans(
            n_rows, sz, self.context_mask_prob, self.context_mask_length, device, generator
        )
        if self.target_masks_per_context > 0:
            target_positions = torch.stack(
                [
                    self._sample_spans(
                        n_rows, sz, self.target_prob, self.target_length, device, generator
                    )
                    for _ in range(self.target_masks_per_context)
                ],
                dim=1,
            )  # (n_rows, N, sz)
        else:
            target_positions = torch.zeros(n_rows, 0, sz, dtype=torch.bool, device=device)
        context_positions = context_positions & ~torch.any(target_positions, dim=1)
        # same float32 division / comparison as ``forward``
        ratio = torch.sum(context_positions, dim=1) / sz
        ok = ratio >= self.ratio_cutoff
        return context_positions, target_positions, ok

    def sample(
        self,
        batch_size: int,
        n_times: int,
        in_channels: int = 1,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        max_redraw_rounds: int = 1000,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Vectorised version of :meth:`forward` with the same output distribution.

        Per row, ``forward`` does (with ``S = n_times // in_channels``)::

            while True:
                ctx = ~spans(context_mask_prob, context_mask_length)   # 1 call
                tgt = [spans(target_prob, target_length) for _ in range(N)]
                ctx &= ~any(tgt)
                if ctx.sum() / S >= ratio_cutoff: break            # else redraw everything

        where ``spans`` is :func:`compute_mask_indices` (see :meth:`_sample_spans` for
        the exact per-call recipe). Here all rows are drawn at once; rows failing the
        ``ratio_cutoff`` are redrawn together (context and all targets) until none fail,
        which is exactly rejection sampling per row.

        Args:
            batch_size: number of rows (B).
            n_times: tokens per row *before* dividing by ``in_channels``.
            in_channels: same semantics as in :meth:`forward`.
            device: device for the RNG draws and the outputs (default: the
                generator's device, else CPU).
            generator: optional ``torch.Generator`` on ``device``.
            max_redraw_rounds: guard against an unsatisfiable ``ratio_cutoff``.

        Returns:
            ``(ctx_mask (B, S) True = NOT context,
               tgt_mask (B, N, S) True = target,
               ctx_tgt_mask (B, N, S) True = NOT visible to the decoder)`` bool tensors;
            ``ctx_tgt_mask == logical_xor(ctx_mask[:, None], tgt_mask)``. With
            ``channel_based_masking`` the last dim is ``S * in_channels``, laid out
            as ``(S C)`` like :meth:`forward`.
        """
        if device is None:
            device = generator.device if generator is not None else torch.device("cpu")
        device = torch.device(device)
        if generator is not None and torch.device(generator.device).type != device.type:
            raise ValueError(f"generator device {generator.device} does not match device {device}")

        sz = n_times // in_channels
        context_positions, target_positions, ok = self._draw_rows(batch_size, sz, device, generator)

        rounds = 0
        while True:
            failing = torch.nonzero(~ok, as_tuple=True)[0]
            if failing.numel() == 0:
                break
            rounds += 1
            if rounds > max_redraw_rounds:
                raise RuntimeError(
                    f"TimeInverseBlockMasker.sample: {failing.numel()} rows still below "
                    f"ratio_cutoff={self.ratio_cutoff} after {max_redraw_rounds} redraw rounds"
                )
            ctx_new, tgt_new, ok_new = self._draw_rows(failing.numel(), sz, device, generator)
            context_positions[failing] = ctx_new
            target_positions[failing] = tgt_new
            ok[failing] = ok_new

        final_context_mask = ~context_positions  # True = masked context
        combined_visible_mask = torch.logical_xor(final_context_mask.unsqueeze(1), target_positions)

        if self.channel_based_masking:
            final_context_mask = rearrange(repeat(final_context_mask, "B S -> B C S", C = in_channels),
                                          "B C S -> B (S C)")
            target_positions = rearrange(repeat(target_positions, "B N S -> B C N S", C = in_channels),
                                              "B C N S -> B N (S C)")
            combined_visible_mask = rearrange(repeat(combined_visible_mask, "B N S -> B C N S", C = in_channels),
                                              "B C N S -> B N (S C)")

        return final_context_mask, target_positions, combined_visible_mask.to(torch.bool)
