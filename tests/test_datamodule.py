"""Tests for the native / legacy loader pipelines (DESIGN.md sections A and F).

Runs under pytest (``python -m pytest tests/test_datamodule.py -q``) and as a
plain script (``python tests/test_datamodule.py``) when pytest is unavailable.
CPU only, tiny synthetic shards, < 1 min.
"""
import gc
import io
import os
import sys
import tarfile
import tempfile
import time

import numpy as np
import soundfile as sf
import torch
import webdataset as wds

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from data_modules import WebAudioDataModule  # noqa: E402
from data_modules.dataset_functions import (  # noqa: E402
    NPY_SAMPLE_RATE,
    collate_native,
    decode_audio_bytes,
    pad_or_crop_to,
    pre_process,
    prepare_native_sample,
)
from wavjepa.masking import TimeInverseBlockMasker  # noqa: E402

SAMPLE_DIR = (
    "/gpfs/scratch1/nodespecific/int4/71563/claude-71563/-gpfs-home1-gyuksel2-wavjepa/"
    "aab9dab0-dceb-4114-a2f1-b6333cea566b/scratchpad/sample"
)
SAMPLE_FLACS = [os.path.join(SAMPLE_DIR, "a.flac"), os.path.join(SAMPLE_DIR, "b.flac")]

try:  # pytest is optional on the cluster
    import pytest

    class SkipTest(Exception):  # only used by the __main__ runner
        pass

    def skip(reason: str):
        pytest.skip(reason)

    _SKIP_EXCEPTIONS: tuple = (pytest.skip.Exception, SkipTest)
except ImportError:  # pragma: no cover - exercised on the cluster
    pytest = None

    class SkipTest(Exception):
        pass

    def skip(reason: str):
        raise SkipTest(reason)

    _SKIP_EXCEPTIONS = (SkipTest,)


# ----------------------------------------------------------------------------- helpers
def _sine(sr: int, seconds: float, channels: int, freq: float = 440.0, seed: int = 0) -> np.ndarray:
    """Deterministic (L, C) float32 test signal with a little noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds), dtype=np.float64) / sr
    base = 0.5 * np.sin(2 * np.pi * freq * t)
    out = np.stack([base * (0.8 ** c) + 0.01 * rng.standard_normal(t.shape) for c in range(channels)], axis=1)
    return out.astype(np.float32)


def _flac_bytes(x: np.ndarray, sr: int, subtype: str = "PCM_24") -> bytes:
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="FLAC", subtype=subtype)
    return buf.getvalue()


def _npy_bytes(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, x)
    return buf.getvalue()


def _write_tar(path: str, members: dict[str, bytes]) -> None:
    with tarfile.open(path, "w") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))


def _masker() -> TimeInverseBlockMasker:
    # configs/masker/AudioSet.yaml numbers
    return TimeInverseBlockMasker(
        target_masks_per_context=4,
        context_mask_prob=0.65,
        context_mask_length=10,
        target_prob=0.25,
        target_length=10,
        ratio_cutoff=0.1,
        channel_based_masking=False,
    )


_MEMBER_CACHE: dict | None = None


def _test_members() -> dict:
    """Synthetic clips and their encoded tar members (built once per process: FLAC encoding is slow)."""
    global _MEMBER_CACHE
    if _MEMBER_CACHE is None:
        clip_441_stereo = _sine(44100, 10.0, 2, freq=440.0, seed=1)
        clip_48_mono = _sine(48000, 10.0, 1, freq=660.0, seed=2)
        clip_short = _sine(22050, 4.0, 1, freq=220.0, seed=3)  # shorter than 10 s
        npy_int16 = (np.clip(_sine(16000, 10.0, 1, freq=330.0, seed=4)[:, 0], -1, 1) * 32767).astype(np.int16)
        flac_members = {
            "audio/unbal_train/clipA.flac": _flac_bytes(clip_441_stereo, 44100),
            "audio/unbal_train/clipB.flac": _flac_bytes(clip_48_mono, 48000),
            "audio/unbal_train/clipC.flac": _flac_bytes(clip_short, 22050, subtype="PCM_16"),
        }
        native_members = dict(flac_members)
        native_members["audio/unbal_train/clipD.npy"] = _npy_bytes(npy_int16)
        native_members["audio/unbal_train/clipD.json"] = b'{"label": "npy clip"}'  # extra key, must be ignored
        _MEMBER_CACHE = dict(
            clip_441_stereo=clip_441_stereo, clip_48_mono=clip_48_mono, clip_short=clip_short,
            npy_int16=npy_int16, flac_members=flac_members, native_members=native_members,
        )
    return _MEMBER_CACHE


class _TmpShards:
    """Context manager creating tiny native / legacy test shards in a temp dir.

    The directory comes from ``$WAVJEPA_TEST_TMP`` if set, else the default temp dir.
    """

    def __enter__(self):
        self.dir = tempfile.mkdtemp(prefix="wavjepa_dm_test_", dir=os.environ.get("WAVJEPA_TEST_TMP") or None)
        cache = _test_members()
        self.clip_441_stereo = cache["clip_441_stereo"]
        self.clip_48_mono = cache["clip_48_mono"]
        self.clip_short = cache["clip_short"]
        self.npy_int16 = cache["npy_int16"]
        self.legacy_tar = os.path.join(self.dir, "legacy-000.tar")
        _write_tar(self.legacy_tar, cache["flac_members"])
        self.native_tar = os.path.join(self.dir, "native-000.tar")
        _write_tar(self.native_tar, cache["native_members"])
        return self

    def __exit__(self, *exc):
        for name in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, name))
        os.rmdir(self.dir)
        return False


# ----------------------------------------------------------------------------- decode
def test_decode_matches_wds_torch_audio_on_sample_flacs():
    """soundfile-from-bytes == wds.torch_audio (torchaudio soundfile backend), channel 0."""
    if not all(os.path.exists(p) for p in SAMPLE_FLACS):
        skip(f"sample FLACs not found under {SAMPLE_DIR}")
    for path in SAMPLE_FLACS:
        with open(path, "rb") as f:
            data = f.read()
        audio, sr = decode_audio_bytes(".flac", data)
        ref, ref_sr = wds.torch_audio(".flac", data)  # (C, L) float32
        assert sr == ref_sr, (path, sr, ref_sr)
        assert audio.dtype == np.float32 and audio.ndim == 1
        assert audio.shape[0] == ref.shape[1], (audio.shape, ref.shape)
        assert torch.allclose(torch.from_numpy(audio), ref[0], atol=1e-6, rtol=0.0), path
        assert sr in (44100, 48000) and audio.shape[0] == sr * 10, (path, sr, audio.shape)


def test_decode_matches_torchaudio_on_synthetic_flacs():
    """Same comparison on synthetic 24-bit FLACs so it also runs without the sample files."""
    for sr, ch in [(44100, 2), (48000, 1), (22050, 1)]:
        data = _flac_bytes(_sine(sr, 1.0, ch, seed=sr), sr)
        audio, got_sr = decode_audio_bytes(".flac", data)
        ref, ref_sr = wds.torch_audio(".flac", data)
        assert got_sr == ref_sr == sr
        assert audio.shape == (sr,)
        assert torch.allclose(torch.from_numpy(audio), ref[0], atol=1e-6, rtol=0.0)


def test_decode_wav_and_unknown_extension():
    x = _sine(16000, 0.5, 1)
    buf = io.BytesIO()
    sf.write(buf, x, 16000, format="WAV", subtype="PCM_16")
    audio, sr = decode_audio_bytes(".wav", buf.getvalue())
    assert sr == 16000 and audio.shape == (8000,) and audio.dtype == np.float32
    assert decode_audio_bytes(".json", b"{}") is None
    assert decode_audio_bytes(".txt", b"hello") is None
    # extension parsing is case-insensitive and tolerates a bare extension
    audio2, sr2 = decode_audio_bytes("FLAC", _flac_bytes(x, 16000))
    assert sr2 == 16000 and audio2.shape == (8000,)


def test_decode_npy_int16():
    x = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    audio, sr = decode_audio_bytes(".npy", _npy_bytes(x))
    assert sr == NPY_SAMPLE_RATE == 16000
    assert audio.dtype == np.float32 and audio.shape == (5,)
    expected = x.astype(np.float32) / 32768.0
    assert np.array_equal(audio, expected), (audio, expected)
    # 2-D (C, L) -> channel 0
    two = np.stack([x, np.zeros_like(x)], axis=0)
    audio2, _ = decode_audio_bytes(".npy", _npy_bytes(two))
    assert np.array_equal(audio2, expected)
    # float arrays pass through as float32
    audio3, _ = decode_audio_bytes(".npy", _npy_bytes(np.array([0.5, -0.25], dtype=np.float64)))
    assert audio3.dtype == np.float32 and np.allclose(audio3, [0.5, -0.25])


# ----------------------------------------------------------------------------- pad / prepare / collate
def test_pad_or_crop_to():
    t = torch.arange(5, dtype=torch.float32)
    assert pad_or_crop_to(t, 5) is t
    padded = pad_or_crop_to(t, 8)
    assert padded.shape == (8,) and torch.equal(padded[:5], t) and torch.all(padded[5:] == 0)
    assert torch.equal(pad_or_crop_to(t, 3), t[:3])
    a = np.arange(5, dtype=np.float32)
    pa = pad_or_crop_to(a, 7)
    assert isinstance(pa, np.ndarray) and pa.shape == (7,) and pa[6] == 0 and pa[4] == 4
    assert pad_or_crop_to(a, 2).shape == (2,)
    # trailing-axis semantics on 2-D input
    m = torch.ones(2, 3)
    assert pad_or_crop_to(m, 5).shape == (2, 5) and pad_or_crop_to(m, 1).shape == (2, 1)


def test_prepare_native_sample():
    long = np.ones(48000 * 11, dtype=np.float32)
    s = prepare_native_sample(long, 48000)
    assert s["audio"].shape == (480000,) and s["audio"].dtype == torch.float32
    assert s["sr"] == 48000 and s["length"] == 480000
    short = np.ones(22050 * 4, dtype=np.float32)
    s = prepare_native_sample(short, 22050)
    assert s["audio"].shape == (220500,) and s["length"] == 22050 * 4
    assert torch.all(s["audio"][: 22050 * 4] == 1) and torch.all(s["audio"][22050 * 4 :] == 0)
    exact = torch.zeros(160000)
    s = prepare_native_sample(exact, 16000)
    assert s["audio"].shape == (160000,) and s["length"] == 160000


def test_collate_native_mixed_lengths():
    samples = [
        prepare_native_sample(np.full(441000, 0.25, dtype=np.float32), 44100),
        prepare_native_sample(np.full(480000, -0.5, dtype=np.float32), 48000),
        prepare_native_sample(np.full(100000, 0.125, dtype=np.float32), 16000),  # short clip
    ]
    batch = collate_native(samples)
    assert set(batch.keys()) == {"audio", "sr", "length"}
    assert batch["audio"].shape == (3, 480000) and batch["audio"].dtype == torch.float32
    assert batch["sr"].dtype == torch.int64 and batch["sr"].shape == (3,)
    assert batch["length"].dtype == torch.int64 and batch["length"].shape == (3,)
    assert batch["sr"].tolist() == [44100, 48000, 16000]
    assert batch["length"].tolist() == [441000, 480000, 100000]
    audio = batch["audio"]
    assert torch.all(audio[0, :441000] == 0.25) and torch.all(audio[0, 441000:] == 0)
    assert torch.all(audio[1] == -0.5)
    assert torch.all(audio[2, :100000] == 0.125) and torch.all(audio[2, 100000:] == 0)
    # single-sample batch keeps its own length
    one = collate_native(samples[2:])
    assert one["audio"].shape == (1, 160000) and one["length"].item() == 100000


# ----------------------------------------------------------------------------- pipelines
def _batches(dm: WebAudioDataModule, n: int) -> list:
    dm.setup("fit")
    loader = dm.train_dataloader()
    it = iter(loader)
    out = [next(it) for _ in range(n)]
    del it, loader
    gc.collect()
    return out


def test_native_pipeline_end_to_end():
    """Real WebDataset pipeline (num_workers=0) over a synthetic shard with flac + npy members."""
    with _TmpShards() as shards:
        dm = WebAudioDataModule(
            masker=None,
            data_dirs=shards.native_tar,
            mixing_weights=None,
            batch_size=4,
            legacy_pipeline=False,
            num_workers=0,
            shuffle_buffer=8,
        )
        batches = _batches(dm, 6)
        seen_sr = set()
        for batch in batches:
            assert isinstance(batch, dict) and set(batch.keys()) == {"audio", "sr", "length"}
            audio, sr, length = batch["audio"], batch["sr"], batch["length"]
            assert audio.dtype == torch.float32 and sr.dtype == torch.int64 and length.dtype == torch.int64
            assert audio.shape[0] == 4 and sr.shape == (4,) and length.shape == (4,)
            assert audio.shape[1] == int(sr.max().item()) * WebAudioDataModule.TARGET_SECONDS
            for i in range(4):
                s, n = int(sr[i]), int(length[i])
                seen_sr.add(s)
                assert n <= s * 10
                assert torch.all(audio[i, n:] == 0), "padding must be zero"
                if s == 44100:
                    assert n == 441000
                    ref = torch.from_numpy(shards.clip_441_stereo[:, 0])  # channel 0 of the stereo clip
                    assert torch.allclose(audio[i, :n], ref, atol=2e-4)  # 24-bit quantisation
                elif s == 48000:
                    assert n == 480000
                    assert torch.allclose(audio[i, :n], torch.from_numpy(shards.clip_48_mono[:, 0]), atol=2e-4)
                elif s == 22050:
                    assert n == 22050 * 4, "short clip keeps its true length"
                elif s == 16000:
                    assert n == 160000
                    ref = torch.from_numpy(shards.npy_int16.astype(np.float32) / 32768.0)
                    assert torch.equal(audio[i, :n], ref), ".npy int16 must be scaled by 1/32768"
                else:
                    raise AssertionError(f"unexpected sr {s}")
        assert seen_sr == {44100, 48000, 22050, 16000}, seen_sr


def test_native_pipeline_with_worker_process():
    """Same pipeline through a forked DataLoader worker (persistent, prefetch)."""
    # torch's default tensor sharing opens an AF_UNIX listener under $TMPDIR (108-byte
    # path limit); with a long TMPDIR the worker cannot hand batches back and the
    # loader would block forever, so skip instead of hanging.
    if len(tempfile.gettempdir()) > 70:
        skip(f"TMPDIR too long for multiprocessing AF_UNIX sockets: {tempfile.gettempdir()}")
    with _TmpShards() as shards:
        dm = WebAudioDataModule(
            masker=None,
            data_dirs=shards.native_tar,
            mixing_weights=None,
            batch_size=2,
            legacy_pipeline=False,
            num_workers=1,
            prefetch_factor=2,
            shuffle_buffer=4,
        )
        batches = _batches(dm, 3)
        for batch in batches:
            assert batch["audio"].shape[0] == 2 and batch["audio"].dtype == torch.float32
            assert batch["sr"].dtype == torch.int64 and batch["length"].dtype == torch.int64


def test_mixed_native_pipeline():
    """RandomMix over two shard patterns still yields the dict contract."""
    with _TmpShards() as shards:
        dm = WebAudioDataModule(
            masker=None,
            data_dirs=[shards.native_tar, shards.legacy_tar],
            mixing_weights=[0.5, 0.5],
            batch_size=3,
            legacy_pipeline=False,
            num_workers=0,
            shuffle_buffer=4,
        )
        for batch in _batches(dm, 3):
            assert set(batch.keys()) == {"audio", "sr", "length"} and batch["audio"].shape[0] == 3


def test_legacy_pipeline_end_to_end_and_resampler_cache():
    """Legacy tuple path: shapes as before, Resample cached per source sr."""
    with _TmpShards() as shards:
        torch.manual_seed(0)
        dm = WebAudioDataModule(
            masker=_masker(),
            data_dirs=shards.legacy_tar,
            mixing_weights=None,
            batch_size=2,
            nr_samples_per_audio=8,
            nr_time_points=200,
            in_channels=1,
            sr=16000,
            legacy_pipeline=True,
            num_workers=0,
            shuffle_buffer=4,
        )
        batches = _batches(dm, 3)
        for batch in batches:
            assert isinstance(batch, (tuple, list)) and len(batch) == 4
            audio, ctx, tgt, ctx_tgt = batch
            assert audio.shape == (2, 1, 160000) and audio.dtype == torch.float32
            assert ctx.shape == (2, 8, 200) and ctx.dtype == torch.bool
            assert tgt.shape == (2, 8, 4, 200) and tgt.dtype == torch.bool
            assert ctx_tgt.shape == (2, 8, 4, 200) and ctx_tgt.dtype == torch.bool
            assert torch.equal(ctx_tgt, torch.logical_xor(ctx.unsqueeze(2), tgt))
        # with num_workers=0 the cache lives in this process: one entry per source sr seen
        cached = sorted(k[0] for k in dm._resamplers)
        assert cached == [22050, 44100, 48000], cached

        # the cached transform is numerically the legacy per-sample construction
        x = torch.from_numpy(shards.clip_48_mono[:, 0])
        import torchaudio
        fresh = torchaudio.transforms.Resample(
            48000, 16000, lowpass_filter_width=64, rolloff=0.9475937167399596,
            resampling_method="sinc_interp_kaiser", dtype=torch.float32, beta=14.769656459379492,
        )
        assert torch.equal(dm._get_resampler(48000, torch.float32)(x), fresh(x))
        assert pre_process(fresh(x), 16000).shape == (1, 160000)


# ----------------------------------------------------------------------------- runner
def _run_all() -> int:
    tests = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]
    failed = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
        except _SKIP_EXCEPTIONS as exc:
            print(f"SKIP  {name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}: {type(exc).__name__}: {exc}")
            import traceback

            traceback.print_exc()
        else:
            print(f"PASS  {name} ({time.perf_counter() - t0:.1f}s)")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
