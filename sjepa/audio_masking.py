import random
from random import randint, randrange

import numpy as np
import torch

from sjepa.masking_utils import mask_to_indices


def gen_targetid_random(
    n_times: int,
    selected_tokens: torch.Tensor | None = None,
    cluster_d: int = 15,
    cluster_u: int = 20,
):
    """
    Generate random target indices for masking.

    Args:
        n_times: The total number of time steps/patches
        selected_tokens: Optional boolean tensor indicating already selected positions (True = selected)
        cluster_d: Minimum cluster size (lower bound)
        cluster_u: Maximum cluster size (upper bound)

    Returns:
        torch.Tensor: Indices of selected positions
    """
    # Sample from non-selected tokens
    if selected_tokens is not None:
        # Find positions that are NOT selected (False positions)
        available_indices = torch.nonzero(~selected_tokens, as_tuple=False).squeeze(-1)
        indices_list = available_indices.tolist()
    else:
        indices_list = list(range(n_times))

    # Ensure we don't try to sample more than available
    mask_size = randint(cluster_d, cluster_u)
    mask_size = min(mask_size, len(indices_list))

    if mask_size == 0:
        return torch.tensor([], dtype=torch.long)

    mask_id = random.sample(indices_list, mask_size)

    return torch.tensor(mask_id, dtype=torch.long)


def gen_targetid_cluster(n_times: int, cluster_d: int = 15, cluster_u: int = 20):
    """
    :p_t_dim: The patch time dimension...
    :mask_patch: Number of patches to mask
    """
    mask_id = []

    cur_clus = randint(cluster_d, cluster_u)
    start_id = randint(0, n_times - cur_clus)
    cur_mask = []
    for i in range(0, cur_clus):
        mask_cand = start_id + i
        if mask_cand < n_times:
            cur_mask.append(mask_cand)

    mask_id.extend(cur_mask)
    mask_id = list(set(mask_id))
    return torch.tensor(mask_id)


def compute_mask_indices(
    shape: tuple[int, int],
    padding_mask: torch.Tensor | None,
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
    require_same_masks: bool = True,
    mask_dropout: float = 0.0,
    add_masks: bool = False,
    seed: int | None = None,
    epoch: int | None = None,
    indices: torch.Tensor | None = None,
    idc_select_ver: int = 1,  # 2 to reproduce mask_tokens_dataset
    num_mask_ver: int = 2,  # 2 to reproduce mask_tokens_dataset
):
    """
    Computes random mask spans for a given shape

    Args:
        shape: the the shape for which to compute masks.
            should be of size 2 where first element is batch size and 2nd is timesteps
        padding_mask: optional padding mask of the same size as shape, which will prevent masking padded elements
        mask_prob: probability for each token to be chosen as start of the span to be masked. this will be multiplied by
            number of timesteps divided by length of mask span to mask approximately this percentage of all elements.
            however due to overlaps, the actual number will be smaller (unless no_overlap is True)
        mask_type: how to compute mask lengths
            static = fixed size
            uniform = sample from uniform distribution [mask_other, mask_length*2]
            normal = sample from normal distribution with mean mask_length and stdev mask_other. mask is min 1 element
            poisson = sample from possion distribution with lambda = mask length
        min_masks: minimum number of masked spans
        no_overlap: if false, will switch to an alternative recursive algorithm that prevents spans from overlapping
        min_space: only used if no_overlap is True, this is how many elements to keep unmasked between spans
        require_same_masks: if true, will randomly drop out masks until same amount of masks remains in each sample
        mask_dropout: randomly dropout this percentage of masks in each example
    """

    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    if num_mask_ver == 1:
        all_num_mask = int(
            # add a random number for probabilistic rounding
            mask_prob * all_sz / float(mask_length) + np.random.rand()
        )
        all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(bsz):
        if seed is not None and epoch is not None and indices is not None:
            seed_i = int(hash((seed, epoch, indices[i].item())) % 1e6)
        else:
            seed_i = None

        rng = np.random.default_rng(seed_i)

        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            assert sz >= 0, sz
        else:
            sz = all_sz

        if num_mask_ver == 1:
            if padding_mask is not None:
                num_mask = int(
                    # add a random number for probabilistic rounding
                    mask_prob * sz / float(mask_length) + np.random.rand()
                )
                num_mask = max(min_masks, num_mask)
            else:
                num_mask = all_num_mask
        elif num_mask_ver == 2:
            num_mask = int(
                # add a random number for probabilistic rounding
                mask_prob * sz / float(mask_length) + rng.random()
            )
            num_mask = max(min_masks, num_mask)
        else:
            raise ValueError()

        if mask_type == "static":
            lengths = np.full(num_mask, mask_length)
        elif mask_type == "uniform":
            lengths = rng.randint(mask_other, mask_length * 2 + 1, size=num_mask)
        elif mask_type == "normal":
            lengths = rng.normal(mask_length, mask_other, size=num_mask)
            lengths = [max(1, int(round(x))) for x in lengths]
        elif mask_type == "poisson":
            lengths = rng.poisson(mask_length, size=num_mask)
            lengths = [int(round(x)) for x in lengths]
        else:
            raise Exception("unknown mask selection " + mask_type)

        if sum(lengths) == 0:
            if mask_type == "static":
                raise ValueError("this should never happens")
            else:
                lengths = [min(mask_length, sz - 1)]

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = rng.randint(s, e - length)
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = np.fromiter(
                    (e - s if e - s >= length + min_space else 0 for s, e in parts),
                    np.int,
                )
                l_sum = np.sum(lens)
                if l_sum == 0:
                    break
                probs = lens / np.sum(lens)
                c = rng.choice(len(parts), p=probs)
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = np.asarray(mask_idc)
        else:
            if idc_select_ver == 1:
                min_len = min(lengths)
                if sz - min_len <= num_mask:
                    min_len = sz - num_mask - 1
                mask_idc = rng.choice(sz - min_len, num_mask, replace=False)
            elif idc_select_ver == 2:
                mask_idc = rng.choice(sz, num_mask, replace=False)
            else:
                raise ValueError()

            mask_idc = np.asarray(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idc = np.unique(mask_idc[mask_idc < sz])
        if len(mask_idc) >= sz:
            raise ValueError(
                
                    f"the entire sequence is masked. "
                    f"sz={sz}; mask_idc[mask_idc]; "
                    f"index={indices[i] if indices is not None else None}"
                
            )
        mask_idcs.append(mask_idc)

    target_len = None
    if require_same_masks:
        if add_masks:
            target_len = max([len(m) for m in mask_idcs])
        else:
            target_len = min([len(m) for m in mask_idcs])

    for i, mask_idc in enumerate(mask_idcs):
        if target_len is not None and len(mask_idc) > target_len:
            mask_idc = rng.choice(mask_idc, target_len, replace=False)

        mask[i, mask_idc] = True

        if target_len is not None and len(mask_idc) < target_len:
            unmasked = np.flatnonzero(~mask[i])
            to_mask = rng.choice(unmasked, target_len - len(mask_idc), replace=False)
            mask[i, to_mask] = True

        if mask_dropout > 0:
            masked = np.flatnonzero(mask[i])
            num_holes = np.rint(len(masked) * mask_dropout).astype(int)
            to_drop = rng.choice(masked, num_holes, replace=False)
            mask[i, to_drop] = False

    return torch.tensor(mask, dtype=torch.bool).squeeze()


def generate_masks_block_masking(
    B: int,
    sequence_len: int,
    mask_prob: float,
    mask_length: int,
) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    Generate context and target masks with random clustering factor.
    Returns the mask tensor and target mask boolean tensor where masked indices are True.
    Target and context vector do not overlap. Context returns True when it is selected.
    Arguments
    ---------
        sequence_len : int
            The length of the sequence that we are going to mask. Corresponds to L in the paper

        nr_targets_per_ctx : int
            Number of targets to generate per one context vector

        cluster_ctx : bool
            Clustering factor for the context masks. Randomly select between [3,5].
            This leads to masking blocks of the context.

        cluster_tgt : bool
            Clustering factor for the target masks. Randomly select between [3,5].
            This leads to masking blocks of the target.

        mask_patch : int
            Number of patches to mask.

        nr_target_patches : int
            Number of target patches to produce

    Returns
    --------
        [torch.Tensor, torch.Tensor] boolean tensor indicating the masked indices.

    """

    mask = torch.zeros((B, sequence_len), requires_grad=False, dtype=torch.bool)
    # Create attention matrix - start with all False
    att_matrix = torch.zeros([B, sequence_len, sequence_len], dtype=torch.bool)
    for i in range(B):
        mask[i] = compute_mask_indices(
            (1, sequence_len),
            padding_mask=None,
            mask_prob=mask_prob,
            mask_length=mask_length,
        )
        idx = mask_to_indices(mask[i])
        # Set entire rows for masked positions to True
        # True in attention_mask means that attention is not applied.
        att_matrix[i, idx, :] = True
        # Set entire columns for masked positions to True
        att_matrix[i, :, idx] = True

    return mask, att_matrix


def generate_mask_random(
    B: int,
    sequence_len: int,
    mask_patch: int,
    cluster: bool,
) -> tuple[torch.BoolTensor]:
    """
    Generate context and target masks with random clustering factor.
    Returns the mask tensor and target mask boolean tensor where masked indices are True.
    Target and context vector do not overlap. Context returns True when it is selected.
    Arguments
    ---------
        sequence_len : int
            The length of the sequence that we are going to mask. Corresponds to L in the paper

        nr_targets_per_ctx : int
            Number of targets to generate per one context vector

        cluster : bool
            Clustering factor for the context masks. Randomly select between [3,5].
            This leads to masking blocks of the context.

        mask_patch : int
            Number of patches to mask.

    Returns
    --------
        [torch.Tensor] boolean tensor indicating the masked indices.

    """

    mask = torch.zeros((B, sequence_len), requires_grad=False, dtype=torch.bool)
    att_matrix = torch.zeros([B, sequence_len, sequence_len], dtype=torch.bool)

    for i in range(B):
        if cluster:
            mask_id = gen_maskid_patch(sequence_len=sequence_len, mask_patch=mask_patch)
        else:
            mask_id = gen_maskid_frame(sequence_len=sequence_len, mask_size=mask_patch)
        mask[i, mask_id] = 1
        idx = mask_to_indices(mask[i])
        att_matrix[i, idx, :] = True
        # Set entire columns for masked positions to True
        att_matrix[i, :, idx] = True

    return mask, att_matrix


def generate_masks(
    B: int,
    sequence_len: int,
    nr_targets_per_ctx: int,
    mask_patch: int,
    cluster_ctx: bool,
    cluster_tgt: bool,
    nr_target_patches: torch.Tensor,
) -> tuple[torch.BoolTensor, torch.BoolTensor]:
    """
    Generate context and target masks with random clustering factor.
    Returns the mask tensor and target mask boolean tensor where masked indices are True.
    Target and context vector do not overlap. Context returns True when it is selected.
    Arguments
    ---------
        sequence_len : int
            The length of the sequence that we are going to mask. Corresponds to L in the paper

        nr_targets_per_ctx : int
            Number of targets to generate per one context vector

        cluster_ctx : bool
            Clustering factor for the context masks. Randomly select between [3,5].
            This leads to masking blocks of the context.

        cluster_tgt : bool
            Clustering factor for the target masks. Randomly select between [3,5].
            This leads to masking blocks of the target.

        mask_patch : int
            Number of patches to mask.

        nr_target_patches : int
            Number of target patches to produce

    Returns
    --------
        [torch.Tensor, torch.Tensor] boolean tensor indicating the masked indices.

    """

    assert nr_target_patches.shape[0] == B, (
        "Number of target patches should have the same length as B"
    )
    assert nr_target_patches.shape[1] == nr_targets_per_ctx, (
        "Inner list of number of target patches should have the same length as nr_targets_per_ctx"
    )
    mask = torch.zeros((B, sequence_len), requires_grad=False, dtype=torch.bool)
    mask_tgts = torch.zeros(
        (B, nr_targets_per_ctx, sequence_len), dtype=torch.bool, requires_grad=False
    )
    for i in range(B):
        if cluster_ctx:
            mask_id = gen_maskid_patch(sequence_len=sequence_len, mask_patch=mask_patch)
        else:
            mask_id = gen_maskid_frame(sequence_len=sequence_len, mask_size=mask_patch)
        mask[i, mask_id] = 1
        for j in range(nr_targets_per_ctx):
            _nr_target_patches = int(nr_target_patches[i, j].item())
            if cluster_tgt:
                mask_tgts_id = gen_maskid_patch_tgt(
                    masked_patches=mask[i], mask_patch=_nr_target_patches
                )
            else:
                mask_tgts_id = gen_maskid_frame_tgt(
                    masked_patches=mask[i], mask_size=_nr_target_patches
                )
            mask_tgts[i, j, mask_tgts_id] = 1
    return mask, mask_tgts


def gen_maskid_patch_tgt(
    masked_patches: torch.Tensor, mask_patch: int = 100, cluster: int = 3
):
    """
    :p_t_dim: The patch time dimension...
    :mask_patch: Number of patches to mask
    """
    mask_id: list[int] = []
    indices: torch.Tensor = masked_patches.nonzero().flatten()
    indices_list: list[int] = indices.tolist()  # type: ignore
    indices_set = set(indices_list)

    # randomize clutering factor in [3,6)
    cur_clus = randrange(cluster) + 3
    while len(list(set(mask_id))) < mask_patch:
        start_id = random.sample(indices_list, 1)[0]
        cur_mask: list[int] = []
        for i in range(-cur_clus, cur_clus):
            mask_cand = start_id + i
            if mask_cand in indices_set:
                cur_mask.append(mask_cand)
        mask_id.extend(cur_mask)
    mask_id = list(set(mask_id))[:mask_patch]
    return torch.tensor(mask_id)


# using cluster for frame masking hurts the performance, so just use the naive random sampling
def gen_maskid_frame_tgt(masked_patches: torch.Tensor, mask_size: int = 100):
    # We sample from the masked indices.
    indices: list[int] = masked_patches.nonzero().flatten().tolist()  # type: ignore
    try:
        mask_id: list[int] = random.sample(indices, k=mask_size)
    except ValueError as e:
        print(e)
        print(len(indices), mask_size, flush=True)
        raise e
    return torch.tensor(mask_id)


def gen_maskid_patch(
    sequence_len: int = 512, mask_patch: int = 100, cluster: int = 3
) -> torch.Tensor:
    """
    :p_t_dim: The patch time dimension...
    :mask_patch: Number of patches to mask
    """
    mask_id: list[int] = []

    # randomize clutering factor in [3,6)
    cur_clus = randrange(cluster) + 3
    while len(list(set(mask_id))) < mask_patch:
        start_id = randrange(sequence_len)
        cur_mask: list[int] = []
        for i in range(-cur_clus, cur_clus):
            mask_cand = start_id + i
            if mask_cand >= 0 and mask_cand < sequence_len:
                cur_mask.append(mask_cand)
        mask_id.extend(cur_mask)
    mask_id = list(set(mask_id))[:mask_patch]
    return torch.tensor(mask_id)


# using cluster for frame masking hurts the performance, so just use the naive random sampling
def gen_maskid_frame(sequence_len: int = 512, mask_size: int = 100) -> torch.Tensor:
    mask_id = random.sample(range(0, sequence_len), mask_size)
    return torch.tensor(mask_id)
