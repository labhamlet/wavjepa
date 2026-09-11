"""E1 objective ablation: the ``latent`` (arm B) and ``bestrq`` (arm C) objectives of :class:`wavjepa.jepa.JEPA`
and the :class:`wavjepa.jepa.RandomProjectionQuantizer` (CPU, tiny model)."""
import pytest
import torch

from wavjepa.extractors import ConvFeatureExtractor
from wavjepa.jepa import JEPA, RandomProjectionQuantizer, conv_geometry
from wavjepa.masking import TimeInverseBlockMasker
from wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG

from test_packing import AUDIOSET_CFG, SR, TINY_CONV, _audio_len_for_tokens

S = 64  # tokens per crop
B = 3


def _noise_stats(n_mels: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-bin log-mel mean / std of white noise (the tests' audio), as the offline AudioSet statistics are for real audio."""
    torch.manual_seed(seed)
    q = RandomProjectionQuantizer(torch.zeros(n_mels), torch.ones(n_mels), n_mels=n_mels)
    x = q.log_mel(torch.randn(32, 16000)).reshape(-1, n_mels)
    return x.mean(dim=0), x.std(dim=0)


NOISE_STATS_16 = _noise_stats(16)


HOP160_CONV = [(64, 10, 5)] + [(64, 3, 2)] * 4 + [(64, 2, 2)]  # configs/extractor/wavjepa.yaml with 64 channels: hop 5*2^5 = 160, rf 240


def _hop160_audio_len(n_tokens: int) -> int:
    length = n_tokens
    for _dim, k, stride in reversed(HOP160_CONV):
        length = (length - 1) * stride + k
    return length


def _model(objective: str, tiny: bool = False, **kw) -> JEPA:
    torch.manual_seed(0)
    spec = TINY_CONV if tiny else HOP160_CONV
    extractor = ConvFeatureExtractor(conv_layers_spec=spec, in_channels=1, depthwise=False)
    audio_len = _audio_len_for_tokens(S) if tiny else _hop160_audio_len(S)
    model = JEPA(
        feature_extractor=extractor,
        transformer_encoder_layers_cfg=TransformerLayerCFG.create(d_model=64, nhead=4),
        transformer_encoder_cfg=TransformerEncoderCFG.create(num_layers=3),
        transformer_decoder_layers_cfg=TransformerLayerCFG.create(d_model=32, nhead=4),
        transformer_decoder_cfg=TransformerEncoderCFG.create(num_layers=2),
        decoder_embedding_dim=32,
        average_top_k_layers=2,
        resample_sr=SR,
        process_audio_seconds=(audio_len + 0.5) / SR,
        nr_samples_per_audio=1,
        compile_modules=False,
        objective=objective,
        bestrq_codebook_size=kw.pop("codebook", 256),
        bestrq_code_dim=8,
        bestrq_n_mels=16,
        bestrq_mel_stats=NOISE_STATS_16,
    )
    assert model.total_patches == S
    return model.train()


def _batch(model: JEPA, seed: int = 1):
    torch.manual_seed(seed)
    masker = TimeInverseBlockMasker(**AUDIOSET_CFG)
    ctx, tgt, ctx_tgt = masker.sample(B, S, in_channels=1, device="cpu", generator=torch.Generator().manual_seed(seed))
    audio = torch.randn(B, 1, model.target_length)
    return audio, ctx.bool(), tgt.bool(), ctx_tgt.bool()


def test_jepa_objective_unchanged():
    model = _model("jepa")
    assert not hasattr(model, "enc_mask_token") and not hasattr(model, "latent_head") and not hasattr(model, "quantizer")
    assert model.teacher_encoder is not None
    out = model(*_batch(model))
    assert torch.isfinite(out["loss"]) and "loss_targets" not in out


@pytest.mark.parametrize("objective", ["latent", "bestrq"])
def test_masked_encoder_objectives(objective):
    model = _model(objective)
    audio, ctx, tgt, ctx_tgt = _batch(model)
    out = model(audio, ctx, tgt, ctx_tgt)
    assert out["contextual_features"].shape == (B, S, 64)
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
    assert "loss_targets" not in out and "masked_fraction" not in out  # no bookkeeping
    # gradients reach the extractor, the encoder and the head; the (unused) predictor gets none
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.extract_audio.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())
    assert model.enc_mask_token.grad is not None
    if objective == "latent":
        assert model.teacher_encoder is not None and model.latent_head.weight.grad is not None
        assert out["preds"].shape == (B, S, 64) and out["targets"].shape == (B, S, 64)
        # B regresses exactly A's targets: the same teacher and the same construction on the unmasked tokens
        with torch.no_grad():
            lf = model.feature_norms(model.extract_audio(audio))
            if model.post_extraction_mapper is not None:
                lf = model.post_extraction_mapper(lf)
            lf = lf + model.pos_encoding_encoder
            assert torch.allclose(out["targets"], model._forward_teacher(lf), atol=1e-5)
    else:
        assert model.teacher_encoder is None and model.bestrq_head.weight.grad is not None
        labels = out["targets"]
        assert labels.shape == (B, S) and labels.dtype == torch.int64
        assert labels.min() >= 0 and labels.max() < 256
        assert 1.0 <= out["code_perplexity"] <= 256.0
        assert out["preds"].shape == (B, S, 256)


def test_quantizer_is_frozen_random_and_deterministic():
    stats = NOISE_STATS_16
    q1 = RandomProjectionQuantizer(*stats, n_mels=16, frames_per_token=1, codebook_size=256, code_dim=8, seed=3)
    q2 = RandomProjectionQuantizer(*stats, n_mels=16, frames_per_token=1, codebook_size=256, code_dim=8, seed=3)
    assert torch.equal(q1.proj, q2.proj) and torch.equal(q1.codebook, q2.codebook)
    assert len(list(q1.parameters())) == 0  # nothing trainable: buffers only
    assert torch.allclose(q1.codebook.norm(dim=-1), torch.ones(256))
    q1.train()
    torch.manual_seed(5)
    wave = torch.randn(2, 160 * 40 + 200)
    n_tokens = 40
    l1 = q1(wave, n_tokens)
    assert l1.shape == (2, n_tokens) and l1.dtype == torch.int64
    assert torch.equal(l1, q1(wave, n_tokens))  # deterministic: a fixed hash of the input
    assert torch.equal(l1, q2(wave, n_tokens))  # same seed, same labels
    # with matched statistics the codes are spread over the codebook ...
    ppl = RandomProjectionQuantizer.perplexity(l1, 256)
    assert ppl > 8, ppl
    # ... and depend on the input
    assert not torch.equal(l1, q1(torch.randn(2, 160 * 40 + 200), n_tokens))
    # short input: the missing frames are zero-filled, no error
    assert q1(torch.randn(2, 160 * 30), n_tokens).shape == (2, n_tokens)
    # mismatched statistics: a common offset dominates every frame and the codes collapse (why the AudioSet
    # statistics matter); the labels also differ from the matched ones
    q3 = RandomProjectionQuantizer(torch.full((16,), -5.0), torch.full((16,), 2.0), n_mels=16, frames_per_token=1,
                                   codebook_size=256, code_dim=8, seed=3)
    l3 = q3(wave, n_tokens)
    assert RandomProjectionQuantizer.perplexity(l3, 256) < ppl
    assert not torch.equal(l1, l3)
    with pytest.raises(ValueError):
        RandomProjectionQuantizer(torch.zeros(8), torch.ones(8), n_mels=16)


def test_frames_per_token_and_offset_follow_the_extractor():
    model = _model("bestrq")
    rf, hop = conv_geometry(model.extract_audio)
    assert (rf, hop) == (240, 160)  # the WavJEPA geometry: 15 ms receptive field, 10 ms hop
    # the analytic geometry matches the extractor's own token count: n = (L - rf) // hop + 1
    for L in (240, 399, 400, 32159):
        assert model.extract_audio.total_patches(L) == (L - rf) // hop + 1
    assert model.extract_audio.total_patches(32159) == 200  # a 2.01 s crop
    assert model.quantizer.frames_per_token == 1
    assert model.quantizer.frame_offset == RandomProjectionQuantizer.frame_offset_for(rf, hop, 1)


def test_bestrq_refuses_other_hops():
    # TINY_CONV hops 20 samples, not 160: the BEST-RQ arm is defined for the 100 Hz WavJEPA front end only
    with pytest.raises(ValueError, match="100 Hz"):
        _model("bestrq", tiny=True)


def test_frame_offset_centres_the_frames_on_the_token():
    # 100 Hz WavJEPA front end: token j = samples [160 j, 160 j + 240), centre 160 j + 120; frame k centred at 160 k -> frame j + 1
    assert RandomProjectionQuantizer.frame_offset_for(receptive_field=240, hop=160, frames_per_token=1) == 1
    q = RandomProjectionQuantizer(torch.zeros(4), torch.ones(4), n_mels=4, frames_per_token=1, frame_offset=1, codebook_size=16, code_dim=4)
    x = torch.arange(6, dtype=torch.float32)[None, :, None].expand(1, 6, 4)  # frame index as value
    assert torch.equal(q.token_frames(x, 3)[0, :, 0], torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(q.token_frames(x, 6)[0, :, 0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 0.0]))  # past the end: zero frame


def test_masking_is_the_papers_context_set():
    """Arms B / C: the context is the masker's context, every other position is masked (nothing else)."""
    model = _model("latent")
    audio, ctx, tgt, ctx_tgt = _batch(model)
    seen = {}
    core = model._masked_encoder_core

    def spy(x):
        seen["x"] = x.detach().clone()
        return core(x)

    model._masked_encoder_core = spy
    model(audio, ctx, tgt, ctx_tgt)
    x = seen["x"]
    fill = (model.enc_mask_token + model.pos_encoding_encoder).detach()[0]
    is_fill = torch.isclose(x, fill.expand_as(x), atol=1e-6).all(dim=-1)
    assert torch.equal(is_fill, ctx)  # masked exactly where the masker says "not context"



# ---------------------------------------------------------------------------------------------------------------
# Arm D: data2vec 2.0 (context-only encoder, conv decoder, one pass, M masks per clip)
# ---------------------------------------------------------------------------------------------------------------
from wavjepa.jepa import D2v2ConvDecoder  # noqa: E402


def test_d2v2_conv_decoder_geometry():
    dec = D2v2ConvDecoder(input_dim=64, dim=32, groups=8, kernel=7, layers=4)
    x = torch.randn(3, 50, 64)
    y = dec(x)
    assert y.shape == (3, 50, 64)
    # first block changes the width (no residual), the other three add residuals; final proj back to the input width
    assert dec.blocks[0][0].in_channels == 64 and dec.blocks[1][0].in_channels == 32
    assert all(b[0].groups == 8 and b[0].kernel_size == (7,) and b[0].padding == (3,) for b in dec.blocks)
    assert dec.proj.in_features == 32 and dec.proj.out_features == 64
    assert all(not isinstance(m, torch.nn.LayerNorm) for m in dec.modules())  # channel norm without affine params


def test_d2v2_objective_forward_and_multi_mask():
    M = 4
    torch.manual_seed(0)
    extractor = ConvFeatureExtractor(conv_layers_spec=HOP160_CONV, in_channels=1, depthwise=False)
    audio_len = _hop160_audio_len(S)
    model = JEPA(
        feature_extractor=extractor,
        transformer_encoder_layers_cfg=TransformerLayerCFG.create(d_model=64, nhead=4),
        transformer_encoder_cfg=TransformerEncoderCFG.create(num_layers=3),
        transformer_decoder_layers_cfg=TransformerLayerCFG.create(d_model=32, nhead=4),
        transformer_decoder_cfg=TransformerEncoderCFG.create(num_layers=2),
        decoder_embedding_dim=32, average_top_k_layers=2, resample_sr=SR,
        process_audio_seconds=(audio_len + 0.5) / SR, nr_samples_per_audio=1, compile_modules=False,
        objective="d2v2", d2v2_decoder_type="conv", d2v2_masks_per_clip=M, d2v2_decoder_dim=32,
        d2v2_decoder_groups=8, masker=TimeInverseBlockMasker(**AUDIOSET_CFG),
    ).train()
    assert model.masks_per_clip == M and model.teacher_encoder is not None
    # the native path draws M masks per crop and keeps crops and masks aligned under the shuffle
    audio_full = torch.randn(B, 1, model.target_length + 500)
    model._generator = None
    a, ctx, tgt, ctx_tgt = model._crop_and_mask(audio_full)
    assert a.shape[0] == B and ctx.shape == (B * M, S) and tgt.shape[1] == AUDIOSET_CFG["target_masks_per_context"]
    assert not torch.equal(ctx[0], ctx[1])  # different masks of the same crop
    out = model(a.float(), ctx, tgt, ctx_tgt)
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
    assert out["preds"].shape == (B * M, S, 64) and out["targets"].shape == (B, S, 64)  # teacher once per clip
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.extract_audio.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.d2v2_decoder.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())  # A's transformer predictor is unused
    assert not hasattr(model, "enc_mask_token")  # the encoder never sees mask tokens


def test_d2v2_decoder_input_is_noise_at_masked_positions():
    """The decoder receives the encoder outputs at context positions and N(0, noise_std) elsewhere; no positions added."""
    torch.manual_seed(0)
    extractor = ConvFeatureExtractor(conv_layers_spec=HOP160_CONV, in_channels=1, depthwise=False)
    audio_len = _hop160_audio_len(S)
    model = JEPA(
        feature_extractor=extractor,
        transformer_encoder_layers_cfg=TransformerLayerCFG.create(d_model=64, nhead=4),
        transformer_encoder_cfg=TransformerEncoderCFG.create(num_layers=3),
        transformer_decoder_layers_cfg=TransformerLayerCFG.create(d_model=32, nhead=4),
        transformer_decoder_cfg=TransformerEncoderCFG.create(num_layers=2),
        decoder_embedding_dim=32, average_top_k_layers=2, resample_sr=SR,
        process_audio_seconds=(audio_len + 0.5) / SR, nr_samples_per_audio=1, compile_modules=False,
        objective="d2v2", d2v2_decoder_type="conv", d2v2_masks_per_clip=1, d2v2_decoder_dim=32,
        d2v2_decoder_groups=8, d2v2_input_dropout=0.0,
    ).eval()
    audio, ctx, tgt, ctx_tgt = _batch(model)
    seen = {}
    core = model._d2v2_decoder_core
    model._d2v2_decoder_core = lambda x: (seen.__setitem__("x", x.detach().clone()), core(x))[1]
    with torch.no_grad():
        model(audio, ctx, tgt, ctx_tgt)
    x = seen["x"]
    masked_vals = x[ctx]
    assert masked_vals.abs().max() < 0.1 and abs(masked_vals.std().item() - 0.01) < 0.005  # noise, std 0.01
    assert x[~ctx].abs().mean() > 0.1  # real encoder outputs at context positions


def _d2v2_model(decoder_type: str, M: int = 4, **kw) -> JEPA:
    torch.manual_seed(0)
    extractor = ConvFeatureExtractor(conv_layers_spec=HOP160_CONV, in_channels=1, depthwise=False)
    audio_len = _hop160_audio_len(S)
    return JEPA(
        feature_extractor=extractor,
        transformer_encoder_layers_cfg=TransformerLayerCFG.create(d_model=64, nhead=4),
        transformer_encoder_cfg=TransformerEncoderCFG.create(num_layers=3),
        transformer_decoder_layers_cfg=TransformerLayerCFG.create(d_model=32, nhead=4),
        transformer_decoder_cfg=TransformerEncoderCFG.create(num_layers=2),
        decoder_embedding_dim=32, average_top_k_layers=2, resample_sr=SR,
        process_audio_seconds=(audio_len + 0.5) / SR, nr_samples_per_audio=1, compile_modules=False,
        objective="d2v2", d2v2_decoder_type=decoder_type, d2v2_masks_per_clip=M,
        d2v2_decoder_dim=32, d2v2_decoder_groups=8, masker=TimeInverseBlockMasker(**AUDIOSET_CFG), **kw,
    ).train()


def test_d2v2_transformer_decoder_reuses_arm_a_predictor():
    """The transformer variant decodes with A's predictor: learned mask token + sin-cos positions, one joint pass."""
    model = _d2v2_model("transformer", M=4)
    assert not hasattr(model, "d2v2_decoder")          # no conv decoder built
    assert not hasattr(model, "enc_mask_token")        # the encoder still never sees a mask token
    audio_full = torch.randn(B, 1, model.target_length + 500)
    model._generator = None
    a, ctx, tgt, ctx_tgt = model._crop_and_mask(audio_full)
    seen = {}
    core = model._d2v2_transformer_core
    model._d2v2_transformer_core = lambda x: (seen.__setitem__("x", x.detach().clone()), core(x))[1]
    out = model(a.float(), ctx, tgt, ctx_tgt)
    x = seen["x"]
    assert x.shape == (B * 4, S, 32)                   # predictor width, one row per (clip, mask)
    # masked positions hold exactly mask_token + sin-cos position; context positions hold something else
    expect = (model.mask_token[0, 0] + model.pos_encoding_decoder[0]).detach()
    got = x[ctx]
    ref = expect.expand(S, -1).repeat(B * 4, 1, 1)[ctx]
    assert torch.allclose(got, ref, atol=1e-5)
    assert not torch.allclose(x[~ctx], expect.expand(S, -1).repeat(B * 4, 1, 1)[~ctx], atol=1e-3)
    # one joint pass, loss over every masked position, gradients through A's predictor and the mask token
    assert torch.isfinite(out["loss"]) and out["loss"] > 0
    assert out["preds"].shape == (B * 4, S, 64) and out["targets"].shape == (B, S, 64)
    out["loss"].backward()
    for mod in (model.decoder, model.encoder, model.extract_audio):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in mod.parameters())
    assert model.mask_token.grad is not None and model.mask_token.grad.abs().sum() > 0


def test_d2v2_conv_decoder_variant_still_works():
    model = _d2v2_model("conv", M=2, d2v2_decoder_kernel=7, d2v2_decoder_layers=6)
    assert hasattr(model, "d2v2_decoder") and len(model.d2v2_decoder.blocks) == 6
    audio, ctx, tgt, ctx_tgt = _batch(model)
    ctx4 = ctx.repeat_interleave(2, dim=0)
    out = model(audio, ctx4, tgt, ctx_tgt)
    assert torch.isfinite(out["loss"]) and out["preds"].shape == (B * 2, S, 64)
    out["loss"].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.d2v2_decoder.parameters())
    assert all(p.grad is None for p in model.decoder.parameters())   # A's predictor unused in the conv variant
