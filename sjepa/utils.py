import inspect

import torch


def has_len(v):
    try:
        len(v)
        return True
    except TypeError:
        return False


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def eq_dict(d1: dict, d2: dict):
    """Check if the dictionaries match on the keys that are present in both.

    Example usage:
    >>> ctx, tgt = time_block_context_block_targets(n_ctx=1, n_times=15, n_ctx_blk=2, width_ctx_blk=5, width_tgt_blk=4, n_ctx_patches=10, n_tgt_per_ctx=1)
    >>> print(f"ctx: {mask_repr(ctx)}\ntgt: {mask_repr(tgt[0], false_char='?')}")
    """
    return all(d1[k] == d2[k] for k in d1 if k in d2)


def mask_repr(x: torch.Tensor, true_char="#", false_char="-"):
    """String representation of a binary mask tensor."""
    assert x.ndim < 3
    while x.ndim < 2:
        x = x.unsqueeze(0)
    return "\n".join(
        ["".join([true_char if xij else false_char for xij in xi]) for xi in x]
    )


def expand_index_like(index: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    """Expands the index along the last dimension of the input tokens.

    Args:
        index:
            Index tensor with shape (batch_size, idx_length) where each entry is
            an index in [0, sequence_length).
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).

    Returns:
        Index tensor with shape (batch_size, idx_length, dim) where the original
        indices are repeated dim times along the last dimension.

    """
    dim = tokens.shape[-1]
    index = index.unsqueeze(-1).expand(-1, -1, dim)
    return index


def set_at_index(
    tokens: torch.Tensor, index: torch.Tensor, value: torch.Tensor
) -> torch.Tensor:
    """Copies all values into the input tensor at the given indices.

    Args:
        tokens:
            Tokens tensor with shape (batch_size, sequence_length, dim).
        index:
            Index tensor with shape (batch_size, index_length).
        value:
            Value tensor with shape (batch_size, index_length, dim).

    Returns:
        Tokens tensor with shape (batch_size, sequence_length, dim) containing
        the new values.

    """
    index = expand_index_like(index, tokens)
    return torch.scatter(tokens, 1, index, value)


def repeat_token(token: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Repeats a token size times.

    Args:
        token:
            Token tensor with shape (1, 1, dim).
        size:
            (batch_size, sequence_length) tuple.

    Returns:
        Tensor with shape (batch_size, sequence_length, dim) containing copies
        of the input token.

    """
    batch_size, sequence_length = size
    return token.repeat(batch_size, sequence_length, 1)
