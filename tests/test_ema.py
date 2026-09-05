"""
Tests for the fused (``torch._foreach_*``) EMA teacher update of ``wavjepa.jepa.JEPA``
against the original per-parameter loop (copied here verbatim as the reference).

pytest-style, but also runnable as ``python tests/test_ema.py`` (pytest is optional).
CPU only, tiny model, well under a minute.
"""
from __future__ import annotations

import copy
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402
from torch import nn  # noqa: E402

from wavjepa.extractors import ConvFeatureExtractor  # noqa: E402
from wavjepa.jepa import JEPA  # noqa: E402
from wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG  # noqa: E402

try:  # pytest is optional
    import pytest  # noqa: F401
except ImportError:  # pragma: no cover
    pytest = None

torch.set_num_threads(min(4, torch.get_num_threads()))


def _tiny_model(seed: int = 0) -> JEPA:
    torch.manual_seed(seed)
    extractor = ConvFeatureExtractor(
        conv_layers_spec=[(64, 10, 5), (64, 3, 2), (64, 3, 2)], in_channels=1, depthwise=False
    )
    model = JEPA(
        feature_extractor=extractor,
        transformer_encoder_layers_cfg=TransformerLayerCFG.create(d_model=64, nhead=4),
        transformer_encoder_cfg=TransformerEncoderCFG.create(num_layers=3),
        transformer_decoder_layers_cfg=TransformerLayerCFG.create(d_model=32, nhead=4),
        transformer_decoder_cfg=TransformerEncoderCFG.create(num_layers=2),
        decoder_embedding_dim=32,
        average_top_k_layers=2,
        process_audio_seconds=(2760 + 0.5) / 16000,
        nr_samples_per_audio=2,
        compile_modules=False,
    )
    return model.eval()


@torch.no_grad()
def _loop_ema(student: nn.Module, teacher: nn.Module, r: float) -> None:
    """The original ``JEPA._step_teacher`` body (reference)."""
    for student_p, teacher_p in zip(student.parameters(), teacher.parameters()):
        teacher_p.data.mul_(r).add_((1 - r) * student_p.detach().data)


@torch.no_grad()
def _perturb(module: nn.Module, scale: float, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        p.add_(torch.randn(p.shape, generator=g) * scale)


def _assert_same_params(a: nn.Module, b: nn.Module, rtol=1e-6, atol=1e-7):
    names_a = [n for n, _ in a.named_parameters()]
    names_b = [n for n, _ in b.named_parameters()]
    assert names_a == names_b
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert pa.shape == pb.shape and pa.dtype == pb.dtype, name
        assert torch.allclose(pa, pb, rtol=rtol, atol=atol), (
            f"{name}: max abs diff {(pa - pb).abs().max().item():.3e}"
        )


def test_foreach_ema_equals_loop_ema():
    model = _tiny_model()
    # make teacher != student so the update actually moves something
    _perturb(model.encoder, 0.1, seed=1)
    _perturb(model.teacher_encoder, 0.05, seed=2)
    ref_teacher = copy.deepcopy(model.teacher_encoder)
    student_before = copy.deepcopy(model.encoder)
    n_params = sum(1 for _ in model.encoder.parameters())
    assert n_params == sum(1 for _ in model.teacher_encoder.parameters()) > 0

    for r in (0.999, 0.99, 0.5, 0.0, 1.0):
        model._get_ema_decay = lambda r=r: r  # type: ignore[method-assign]
        model._step_teacher()
        _loop_ema(student_before, ref_teacher, r)
        _assert_same_params(model.teacher_encoder, ref_teacher)
        # the student is never modified by the EMA step
        _assert_same_params(model.encoder, student_before, rtol=0, atol=0)

    # r == 1 leaves the teacher unchanged, r == 0 copies the student
    snapshot = copy.deepcopy(model.teacher_encoder)
    model._get_ema_decay = lambda: 1.0  # type: ignore[method-assign]
    model._step_teacher()
    _assert_same_params(model.teacher_encoder, snapshot, rtol=0, atol=0)
    model._get_ema_decay = lambda: 0.0  # type: ignore[method-assign]
    model._step_teacher()
    _assert_same_params(model.teacher_encoder, model.encoder, rtol=0, atol=0)


def test_ema_many_steps_stays_close_to_loop():
    """Accumulated rounding over many steps stays within a few ulps of the loop."""
    model = _tiny_model(seed=3)
    _perturb(model.encoder, 0.2, seed=4)
    ref_teacher = copy.deepcopy(model.teacher_encoder)
    model._get_ema_decay = lambda: 0.999  # type: ignore[method-assign]
    for step in range(50):
        _perturb(model.encoder, 1e-3, seed=100 + step)   # a "training" step
        model._step_teacher()
        _loop_ema(model.encoder, ref_teacher, 0.999)
    _assert_same_params(model.teacher_encoder, ref_teacher, rtol=1e-5, atol=1e-6)


def test_ema_step_properties():
    model = _tiny_model()
    # no trainer attached -> global_step == 0 -> the initial decay
    assert abs(model._get_ema_decay() - model.hparams.ema_decay) < 1e-12
    model._step_teacher()
    for p in model.teacher_encoder.parameters():
        assert p.requires_grad is False and torch.isfinite(p).all()
    for p in model.encoder.parameters():
        assert p.requires_grad is True
    # works under an autocast-disabled region exactly like training_step calls it
    with torch.autocast(device_type="cpu", enabled=False):
        model._step_teacher()


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            failures += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {name}: {e}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
