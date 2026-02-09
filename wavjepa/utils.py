import inspect
import torch
from typing import Tuple
import torch.nn.functional as F

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


def mask_repr(x:torch.Tensor, true_char="#", false_char="-"):
    """ String representation of a binary mask tensor."""
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

    

def repeat_token(token: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
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

def fade_in(audio, sr, duration = 0.2):
    end = int(duration * sr)
    start = 0
    fade_curve = torch.linspace(0.0, 1.0, end, device = audio.device)
    audio[start:end] = audio[start:end] * fade_curve
    return audio

def fade_out(audio: torch.Tensor, sr: int, duration: float = 0.20) -> torch.Tensor:
    """Apply fade-out to the audio source.

    Arguments
    ----------
    audio : torch.Tensor
        The audio that we want to fade-out
    sr : int 
        Sampling rate of the audio
    duration : float
        Duration of the fade-out

    Returns
    --------
    torch.Tensor with faded-out audio
    """
    # convert to audio indices (samples)
    length = int(duration * sr)
    end = audio.shape[0]
    start = end - length
    # compute fade out curve
    # linear fade
    fade_curve = torch.linspace(1.0, 0.0, length, device=audio.device)
    # apply the curve
    audio[start:end] = audio[start:end] * fade_curve
    return audio

def loop(audio, sr, target_length):
    assert audio.ndim == 1, "Audio has channel dimension, collapse this before using looping"
    audio_length = audio.shape[-1]
    
    max_loops = (target_length // audio_length) + 1
    looped_audio = torch.zeros(target_length, device=audio.device, dtype=audio.dtype)
    
    # Copy first audio (no fade-in)
    looped_audio[:audio_length] = audio
    
    # Loop for additional copies
    for i in range(1, max_loops):
        current_pos = i * audio_length
        remaining = target_length - current_pos
        remaining_seconds = remaining / sr
        
        if remaining_seconds >= 0.5:
            chunk = audio.clone()
            chunk = fade_in(chunk, sr, duration=0.2)
            copy_length = min(audio_length, remaining)
            
            if copy_length < audio_length:
                chunk[:copy_length] = fade_out(chunk[:copy_length], sr, duration=0.2)
            
            looped_audio[current_pos:current_pos + copy_length] = chunk[:copy_length]
        else:
            # Gap < 0.5s, stop (already zero-padded)
            break
    
    return looped_audio

def pad_random_select_or_loop(
    audio : torch.Tensor, 
    target_length: int,
    sr : int,

    ) -> torch.Tensor:
    audio_length = audio.shape[-1]
    padding = target_length - audio_length
    needed_seconds = padding / sr 

    if needed_seconds >= 0.5:
        audio = loop(audio, sr, target_length)
    elif needed_seconds <= 0.5 and needed_seconds >= 0.0:
        audio = F.pad(audio, (0, padding), "constant", 0)
    elif needed_seconds <= 0:  # select a random 10 seconds.
        rand_index = torch.randint(0, audio_length - target_length, (1,))
        audio = audio[rand_index : rand_index + target_length]
    else:
        audio = audio
    assert audio.shape[-1] == target_length
    return audio
