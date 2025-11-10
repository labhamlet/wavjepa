from math import ceil

import einops
import torch


def time_inverse_block_masking(
    n_masks: int,
    n_times: int,
    unmasked_width: int,
    n_unmasked: int,
    ratio_adjuster: float = 0.1,
    max_iter: int = 1000,
    exact: bool = True,
    out=None,
):
    """
    See Baevski 2022: data2vec 2.0

    Args:
        n_masks: number of masks to create.
        n_times: length of the sequence to mask (L in paper).
        unmasked_width: minimal width for the non-masked segments (B in paper).
        n_unmasked: minimum number of non-masked elements (n_unmasked / n_times = 1 - R in paper).
        ratio_adjuster: hyperparameter to help with sampling (A in paper).
        max_iter: maximum number of sampling to do before raising an error.
        exact: if ``True``, the number of non-masked elements will be exactly ``n_unmasked``,
            even if unmasked_width has to be violated.
        out: pre-allocated tensor where the mask will be stored.
            shape (n_masks, n_times)

    Returns:
        mask: ``(n_masks, n_times)``
            True where masked (i.e. should be ignored)
    """
    n_blocks = ceil((n_times * ratio_adjuster + n_unmasked) / unmasked_width)
    if exact:
        assert n_blocks * unmasked_width - n_unmasked <= unmasked_width, (
            "cannot use exact=True with these parameters. "
            "Increase unmasked_width or decrease ratio_adjuster"
        )
    mask = out if out is not None else torch.empty((n_masks, n_times), dtype=torch.bool)
    todo = torch.ones(n_masks, dtype=torch.bool, device=mask.device)
    rand_acc = mask.new_empty(
        (n_masks, n_times + unmasked_width - 1), dtype=torch.float32
    )
    rand_acc[:, : unmasked_width - 1] = torch.inf
    rand_acc[:, -unmasked_width + 1 :] = torch.inf
    n_todo = todo.sum()
    for _ in range(max_iter):
        _ = rand_acc[:n_todo][:, unmasked_width - 1 : -unmasked_width + 1].uniform_()
        starts_order = (
            rand_acc[:n_todo].argsort(dim=1).argsort(dim=1)
        )  # of non-masked blocks
        mask[todo] = (
            (starts_order >= n_blocks)  # (n_todo, n_times + unmasked_width - 1)
            .unfold(
                dimension=1, size=n_times, step=1
            )  # (n_todo, unmasked_width, n_times)
            .all(dim=1)  # (n_todo, n_times)
        )
        diff = (n_times - mask[todo].sum(dim=1)) - n_unmasked  # want this positive
        done = diff >= 0
        if exact:
            for im, d, so in zip(
                torch.arange(len(todo))[todo][done],
                diff[done],
                starts_order[done],
            ):
                if d == 0:
                    continue
                # reduce an arbitrary non-masked block to have exactly n_unmasked:
                j0 = so.argmin()
                mask[im, j0 - d + 1 : j0 + 1] = True
        todo[todo.clone()] = ~done
        n_todo = todo.sum()
        if n_todo > 0:
            continue
        return mask.squeeze()

    params = dict(
        n_masks=n_masks,
        n_times=n_times,
        unmasked_width=unmasked_width,
        n_unmasked=n_unmasked,
        ratio_adjuster=ratio_adjuster,
    )
    raise RuntimeError(
        f"too many iterations ({max_iter}) with the following parameters: {params}"
    )


def time_block_context_block_targets(
    n_ctx: int,
    n_times: int,
    n_ctx_blk: int,
    width_ctx_blk: int,
    n_ctx_patches: int,
    n_tgt_per_ctx: int,
    width_tgt_blk: int,
    out_ctx=None,
    out_tgt=None,
    compute_target=True,
):
    """
    Block context and block targets masking strategy.

    Creates context masks each made of multiple blocks with a minimal width.
    For each context, creates multiple targets, each made of one block with
    all the same width.
    The targets are not overlapping with their context.
    If ``n_ctx_patches != n_ctx_blk * width_ctx_blk``, then the "remaining"
    patches are distributed randomly uniformly among the context blocks.

    Args:
        n_ctx: number of context masks to create.
        n_times: length of the sequence to mask (L in paper).
        n_ctx_blk: number of blocks in each context.
        width_ctx_blk: minimal width for the blocks in the contexts.
        n_ctx_patches: exact number of patches in each context.
        n_tgt_per_ctx: number of targets to create for each context.
        width_tgt_blk: width of the target blocks.
        out_ctx: pre-allocated ctx output. Shape (n_ctx, n_times)
        out_tgt: pre-allocated tgt output. Shape (n_ctx, n_tgt_per_ctx, n_times)
        compute_target: if ``False``, only compute the context mask.

    Returns:
        tuple:
            * masks_context: ``(n_ctx, n_times)``
                True at the masked positions (i.e. should be ignored when
                computing the context).
            * masks_target: ``(n_ctx, n_tgt_per_ctx, n_times)``
                True at the masked positions (i.e. should not be predicted).
                Only present if ``compute_target=True``.
    """
    n_remaining = n_ctx_patches - n_ctx_blk * width_ctx_blk
    if n_remaining < 0:
        raise ValueError(
            f"Because n_remaining = {n_ctx_patches - n_ctx_blk * width_ctx_blk=} "
            f"should not be negative"
        )
    n_unused = n_times - n_ctx_patches - n_tgt_per_ctx * width_tgt_blk
    if n_unused < 0:
        raise ValueError(
            f"Because n_unused = {n_times - n_ctx_patches - n_tgt_per_ctx * width_tgt_blk=} "
            f"should not be negative"
        )
    out_ctx = (
        torch.empty((n_ctx, n_times), dtype=torch.bool) if out_ctx is None else out_ctx
    )
    out_tgt = (
        torch.empty((n_ctx, n_tgt_per_ctx, n_times), dtype=torch.bool)
        if compute_target and out_tgt is None
        else out_tgt
    )
    widths = out_ctx.new_empty(
        (n_ctx, n_ctx_blk + n_tgt_per_ctx + n_unused), dtype=torch.int64
    )
    widths[:, :n_ctx_blk] = width_ctx_blk
    widths[:, n_ctx_blk : n_ctx_blk + n_tgt_per_ctx] = width_tgt_blk
    if n_unused:
        widths[:, -n_unused:] = 1
    if n_remaining > 0:
        distribution_loc = (
            widths.new_empty((n_ctx, n_remaining), dtype=torch.float32).uniform_()
            * n_ctx_blk
        ).long()
        for i in range(ceil(n_remaining / n_ctx_blk)):
            widths[:, :n_ctx_blk].scatter_add_(
                dim=1,
                index=distribution_loc[:, n_ctx_blk * i : n_ctx_blk * (i + 1)],
                src=widths.new_ones((n_ctx, n_ctx_blk)),
            )
    perm = torch.empty_like(widths, dtype=torch.float32).uniform_().argsort(dim=1)
    rev_perm = perm.argsort(dim=1)
    cumwidths = torch.cat(
        (widths.new_zeros((n_ctx, 1)), widths.gather(dim=1, index=perm)),
        dim=1,
    ).cumsum(dim=1)
    idx = torch.arange(n_times, device=out_ctx.device)
    # Contexts:
    out_ctx[:] = ~(
        (
            idx[None, None, :]
            >= cumwidths.gather(
                dim=1,
                index=rev_perm[:, :n_ctx_blk],
            )[:, :, None]
        )
        & (
            idx[None, None, :]
            < cumwidths.gather(
                dim=1,
                index=rev_perm[:, :n_ctx_blk] + 1,
            )[:, :, None]
        )
    ).any(dim=1)
    if not compute_target:
        return out_ctx
    # Targets:
    out_tgt[:] = ~(
        (
            idx[None, None, :]
            >= cumwidths.gather(
                dim=1,
                index=rev_perm[:, n_ctx_blk : n_ctx_blk + n_tgt_per_ctx],
            )[:, :, None]
        )
        & (
            idx[None, None, :]
            < cumwidths.gather(
                dim=1,
                index=rev_perm[:, n_ctx_blk : n_ctx_blk + n_tgt_per_ctx] + 1,
            )[:, :, None]
        )
    )
    return out_ctx, out_tgt


def get_distance_matrix(channels_positions: torch.Tensor):
    """
    Distance matrix.

    Args:
        channels_positions: ``(n_channels, 3)``

    Returns:
        output: ``(n_channels, n_channels)``
    """
    return (
        (channels_positions.unsqueeze(0) - channels_positions.unsqueeze(1)) ** 2
    ).sum(dim=2) ** 0.5


def channels_block_masking(
    channels_positions: torch.Tensor,
    n_masks: int,
    radius: float,
    n_blocks: int,
    min_unmasked=None,
    return_centers: bool = False,
    out=None,
):
    """Masks all channels in the radius of certain channels (centers) chosen randomly.

    See Bao 2021. ``True`` where masked (i.e. should be ignored).

    Returns:
        output: ``(n_masks, n_channels)``
    """
    n_channels, n_coord = channels_positions.size()
    if n_blocks == 0:  # nothing masked:
        return channels_positions.new_zeros((n_masks, n_channels), dtype=torch.bool)
    centers = (
        channels_positions.new_empty((n_masks, n_channels), dtype=torch.float32)
        .uniform_()
        .argsort(dim=1)[:, :n_blocks]
    )  # (n_masks, n_blocks)
    dist_mat = get_distance_matrix(channels_positions)  # (n_channels, n_channels)
    mask_mat = dist_mat < radius
    mask = (
        out
        if out is not None
        else channels_positions.new_empty((n_masks, n_channels), dtype=torch.bool)
    )

    mask[:] = mask_mat[centers].any(
        dim=1
    )  # (n_masks, n_blocks, n_channels) -> (n_masks, n_channels)
    if min_unmasked is not None:
        unmasked = dist_mat[centers].min(dim=1).values.argsort(dim=1)[:, -min_unmasked:]
        # Distance to the closest block center (n_masks,n_channels),
        # then indices of the min_unmasked elements the furthest away from a center
        # that must remain unmasked (n_masked, min_unmasked).

        selected = torch.zeros_like(mask, dtype=torch.bool)
        selected.scatter_(1, unmasked, True)
        mask[selected] = False
    if return_centers:
        return mask, centers
    return mask


def random_masking(
    padding_mask: torch.BoolTensor,
    n_unmasked: int | torch.LongTensor,
    strict: bool = False,
):
    """
    Returns ``True`` where masked (i.e. should be ignored).
    There are ``n_unmasked`` patches per element.
    The patches masked by ``padding_mask`` are given priority to be masked here,
    but if less than ``n_unmasked`` patches are not masked by `padding_mask``,
    then some padded patches will not be masked by this function, except if
    ``strict=True``.


    Args:
        padding_mask: mask that is ``True`` where padding was applied, i.e. ignored by the function
        n_unmasked: number of batches that should be non masked per batch (or any previous dimension)
        strict: if true, raises an error if there are not enough non-masked elements to mask.

    Returns:
        output: ``(*, n_patches)``
    """
    n_unmasked_in_padding_mask = (~padding_mask).sum(dim=-1)
    # Unsqueeze prefered to keepdim=True because nicer error print
    if (n_unmasked_in_padding_mask.unsqueeze(-1) < n_unmasked).any():
        if strict:
            msg = (
                f"[random_masking] Some padding masks have less than "
                f"{n_unmasked=} non-masked elements:\n {n_unmasked_in_padding_mask=}."
            )
            raise ValueError(msg)

    rand_acc = torch.empty_like(padding_mask, dtype=torch.float32).uniform_()
    rand_acc[padding_mask] = torch.inf
    return rand_acc.argsort(dim=-1).argsort(dim=-1) >= n_unmasked


def random_masking_2stages(
    padding_mask: torch.BoolTensor,
    strict_padding_mask: torch.BoolTensor,
    n_unmasked: int,
):
    """
    Similar to ``random_masking``, except:
    stage 1: random_masking(padding_mask, ...)
    stage 2: eventually complete with random_masking(strict_padding_mask, ..., strict=True)

    Args:
        padding_mask: mask that is ``True`` where padding was applied, i.e. ignored by the function
        strict_padding_mask: mask used where padding_mask can not satisfy the n_unmasked requirement
        n_unmasked: number of batches that should be non masked per batch (or any previous dimension)
        strict: if true, raises an error if there are not enough non-masked elements to mask.

    Returns:
        output: ``(*, n_patches)``
    """
    n_unmasked_in_padding_mask = (~padding_mask | ~strict_padding_mask).sum(dim=-1)
    # Unsqueeze prefered to keepdim=True because nicer error print
    if (n_unmasked_in_padding_mask.unsqueeze(-1) < n_unmasked).any():
        msg = (
            f"[random_masking_2stages] Some padding masks have less than "
            f"{n_unmasked=} non-masked elements:\n {n_unmasked_in_padding_mask=}."
        )
        raise ValueError(msg)
    output1 = random_masking(padding_mask, n_unmasked, strict=False)
    output1[padding_mask] = True
    n_unmasked_in_output1 = n_unmasked - (~output1).sum(dim=-1)
    padding_mask2 = strict_padding_mask | ~padding_mask
    output2 = random_masking(
        padding_mask2, n_unmasked_in_output1.unsqueeze(-1), strict=True
    )
    assert (output1 | output2).all()  # no element left unmasked by both stages
    output = output1 & output2
    assert ((~output).sum(dim=-1) == n_unmasked).all()
    return output


def mask_to_indices(mask: torch.BoolTensor, check: bool = False) -> torch.Tensor:
    """
    Returns the indices of the true elements.

    Args:
        mask: (*batch_dims, masked_dim)
            Boolean mask. For every element, the number of true values in the masked dimension
            should be the same. i.e. ``mask.sum(dim=-1)`` should be constant.
        check: bool
            If ``True``, check that the output is correct. Slower.

    Returns:
        indices: (*batch_dims, n_unmasked)
            Indices of the true elements in the mask.
    """

    n_true = mask.sum(dim=-1).unique()
    assert n_true.size(0) == 1
    n_true = int(n_true.item())
    batch_dims = list(mask.shape[:-1])
    if check:
        out = mask.nonzero(as_tuple=False)
        out = out.view(*batch_dims, n_true, len(batch_dims) + 1)
        for i, d in enumerate(batch_dims):
            for j in range(0, d):
                assert (out[..., i].select(i, j) == j).all()
        out = out[..., -1]
    else:
        *_, out = mask.nonzero(as_tuple=True)
        out = out.view(*batch_dims, n_true)
    return out.to(mask.device)


def complement_index(index, n):
    """
    Get the complement of the index in the range [0, n)

    Args:
        index: (*bdims, n0)
            The index used to select the elements from x.
            Only works if the index is 1D.
        n: int
            The size of the dimension to select from.
    Returns:
        out: (*bdims, n - n0)
    """
    assert index.max() < n, "sparse index"
    n_bdims = index.ndim - 1
    all_idx = einops.rearrange(
        torch.arange(n, device=index.device), f"n -> {'1 ' * n_bdims} n 1"
    )
    index = einops.rearrange(index, "... n0 -> ... 1 n0")
    out = (all_idx != index).all(dim=-1)
    comp_indices = mask_to_indices(out, check=True)
    assert comp_indices.shape[-1] == n - index.shape[-1], "sparse index"
    return comp_indices
