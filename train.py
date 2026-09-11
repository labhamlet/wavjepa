import gc
import os

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from pytorch_lightning.loggers import TensorBoardLogger

from utils import get_identity_from_cfg
from data_modules import WebAudioDataModule

from wavjepa.jepa import JEPA
from wavjepa.masking import TimeInverseBlockMasker
from wavjepa.extractors import ConvFeatureExtractor, Extractor
from wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG



# Component registries
NETWORKS = {"JEPA": JEPA}
MASKERS = {
    "time-inverse": TimeInverseBlockMasker,
}
EXTRACTORS = {
    "wav2vec2": ConvFeatureExtractor,
    "wavjepa": ConvFeatureExtractor,
}
ENCODERS = {
    "Transformer": {
        "LayerCFG": TransformerLayerCFG,
        "EncoderCFG": TransformerEncoderCFG,
    }
}

# Defaults for the loader / packing knobs when a config does not set them
# (``cfg.data.<key>`` wins over ``cfg.trainer.<key>``, see ``_loader_setting``).
LOADER_DEFAULTS = {
    "legacy_pipeline": False,
    "num_workers": 16,
    "prefetch_factor": 4,
    "shuffle_buffer": 1000,
}
PACKING_DEFAULTS = {"use_packing": True, "pad_multiple": 32}

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


# Enable cuDNN benchmarking for consistent input sizes
torch.backends.cudnn.benchmark = True


def _loader_setting(cfg, key: str):
    """Loader knob ``key``: ``cfg.data`` first, then ``cfg.trainer``, then ``LOADER_DEFAULTS``."""
    default = LOADER_DEFAULTS[key]
    return cfg.data.get(key, cfg.trainer.get(key, default))


def default_transformer_cfgs(predictor_layers: int = 12) -> dict:
    """The transformer configs ``train.py`` builds the production model with.

    Returned as keyword arguments of :class:`JEPA` so that benchmarks / smoke tests
    can build a smaller model through the very same factory code path by passing
    their own dict to :meth:`ComponentFactory.create_network`. ``predictor_layers``
    (``trainer.predictor_layers``, default 12 = the paper) is the E6 predictor-depth knob.
    """
    return dict(
        transformer_encoder_cfg=TransformerEncoderCFG.create(),
        transformer_encoder_layers_cfg=TransformerLayerCFG.create(),
        transformer_decoder_cfg=TransformerEncoderCFG.create(num_layers=int(predictor_layers)),
        transformer_decoder_layers_cfg=TransformerLayerCFG.create(d_model=384),
    )


class ComponentFactory:
    """Factory class for creating model components with proper error handling."""

    @staticmethod
    def create_extractor(cfg) -> Extractor:
        """Create and configure the extractor component."""
        extractor_class = EXTRACTORS.get(cfg.extractor.name)
        if extractor_class is None:
            raise ValueError(
                f"Unknown extractor type: {cfg.extractor.name}. "
                f"Available extractors: {list(EXTRACTORS.keys())}"
            )

        weight_sharing = cfg.extractor.get("share_weights_over_channels", None)
        return extractor_class(
                conv_layers_spec=eval(cfg.extractor.conv_layers_spec),
                in_channels=cfg.data.in_channels,
                depthwise = cfg.extractor.depthwise,
                share_weights_over_channels = weight_sharing,
            )


    @staticmethod
    def create_masker(cfg):
        """Create and configure the masker component.

        The same instance is handed to the model (vectorised ``masker.sample`` on the
        GPU, native loader path) and to the data module (loop ``forward`` in the
        workers, legacy loader path).
        """
        masker_class = MASKERS.get(cfg.masker.name)
        if masker_class is None:
            raise ValueError(
                f"Unknown masker type: {cfg.masker.name}. "
                f"Available maskers: {list(MASKERS.keys())}"
            )

        if cfg.masker.name == "time-inverse":
            # configs/masker/AudioSet.yaml uses ``context_mask_prob`` / ``context_mask_length``;
            # older configs used ``context_prob`` / ``context_length``. Accept both.
            context_prob = cfg.masker.get("context_mask_prob", cfg.masker.get("context_prob", 0.65))
            context_length = cfg.masker.get("context_mask_length", cfg.masker.get("context_length", 10))
            return TimeInverseBlockMasker(
                    target_masks_per_context=cfg.masker.target_masks_per_context,
                    context_mask_prob=context_prob,
                    context_mask_length=context_length,
                    target_prob=cfg.masker.target_prob,
                    target_length=cfg.masker.target_length,
                    ratio_cutoff=cfg.masker.ratio_cutoff,
                    channel_based_masking=cfg.masker.channel_based_masking,
                )
        else:
            raise Exception("No masker found")


    @staticmethod
    def create_network(cfg, extractor : Extractor, masker=None, transformer_cfgs: dict | None = None) -> JEPA:
        """Create and configure the main network.

        Args:
            cfg: hydra config.
            extractor: the waveform feature extractor (:meth:`create_extractor`).
            masker: mask generator passed to the model for the native batch path
                (``JEPA.on_after_batch_transfer`` calls ``masker.sample`` on the GPU).
            transformer_cfgs: optional override of :func:`default_transformer_cfgs`
                (used by benchmarks / smoke tests to build a small model through the
                same code path). ``None`` = the production configuration.
        """
        network_class = NETWORKS.get(cfg.model)
        if network_class is None:
            raise ValueError(
                f"Unknown network type: {cfg.model}. "
                f"Available networks: {list(NETWORKS.keys())}"
            )

        try:
            return network_class(
                feature_extractor=extractor,
                **(transformer_cfgs if transformer_cfgs is not None
                   else default_transformer_cfgs(predictor_layers=int(cfg.trainer.get("predictor_layers", 12)))),
                lr=cfg.optimizer.lr,
                adam_betas=(cfg.optimizer.b1, cfg.optimizer.b2),
                adam_weight_decay=cfg.optimizer.weight_decay,
                resample_sr=cfg.data.sr,
                process_audio_seconds=cfg.data.process_seconds,
                nr_samples_per_audio=cfg.data.samples_per_audio,
                compile_modules = cfg.trainer.compile_modules,
                average_top_k_layers = cfg.trainer.average_top_k_layers,
                # Schedule knobs (paper defaults); configs/trainer/ablation_100k.yaml compresses them.
                ema_decay = float(cfg.trainer.get("ema_decay", 0.999)),
                ema_end_decay = float(cfg.trainer.get("ema_end_decay", 0.99999)),
                ema_anneal_end_step = int(cfg.trainer.get("ema_anneal_end_step", 100000)),
                warmup_steps = int(cfg.trainer.get("warmup_steps", 100000)),
                size = cfg.trainer.get("size", "base"),
                use_packing = cfg.trainer.get("use_packing", PACKING_DEFAULTS["use_packing"]),
                pad_multiple = cfg.trainer.get("pad_multiple", PACKING_DEFAULTS["pad_multiple"]),
                masker = masker,
                # E1 objective ablation: jepa (paper) | latent (arm B, data2vec-style) | bestrq (arm C)
                objective = str(cfg.trainer.get("objective", "jepa")),
                bestrq_codebook_size = int(cfg.trainer.get("bestrq_codebook_size", 8192)),
                bestrq_code_dim = int(cfg.trainer.get("bestrq_code_dim", 16)),
                bestrq_n_mels = int(cfg.trainer.get("bestrq_n_mels", 80)),
                bestrq_stats_path = cfg.trainer.get("bestrq_stats_path", None),
                # arm D (data2vec 2.0): masks per clip and the conv decoder
                d2v2_masks_per_clip = int(cfg.trainer.get("d2v2_masks_per_clip", 4)),
                d2v2_decoder_type = str(cfg.trainer.get("d2v2_decoder_type", "transformer")),
                d2v2_decoder_dim = int(cfg.trainer.get("d2v2_decoder_dim", 384)),
                d2v2_decoder_groups = int(cfg.trainer.get("d2v2_decoder_groups", 16)),
                d2v2_decoder_kernel = int(cfg.trainer.get("d2v2_decoder_kernel", 7)),
                d2v2_decoder_layers = int(cfg.trainer.get("d2v2_decoder_layers", 20)),
                d2v2_input_dropout = float(cfg.trainer.get("d2v2_input_dropout", 0.1)),
                d2v2_mask_noise_std = float(cfg.trainer.get("d2v2_mask_noise_std", 0.01)),
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create network instance: {str(e)}")


def setup_logger(cfg) -> TensorBoardLogger:
    """Set up TensorBoard logger with proper configuration."""
    identity = get_identity_from_cfg(cfg)
    return TensorBoardLogger(
        f"{cfg.save_dir}/tb_logs_jepa",
        name=identity.replace("_", "/"),
    )


def setup_callbacks(cfg):
    """Set up training callbacks."""
    identity = get_identity_from_cfg(cfg)

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.save_dir}/saved_models_jepa_new_masking/{identity.replace('_', '/')}",
        filename="{step}",
        verbose=True,
        every_n_train_steps=int(cfg.trainer.get("checkpoint_every_n_steps", 25000)),
        save_last=True,
        enable_version_counter=True,
        save_top_k=-1,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    # Per-step wall time and DataLoader wait, logged to TensorBoard and printed every N steps.
    return [checkpoint_callback, lr_monitor]


def setup_trainer(cfg, logger, callbacks) -> pl.Trainer:
    """Set up PyTorch Lightning trainer with proper configuration."""
    num_gpus = int(cfg.trainer.num_gpus)

    return pl.Trainer(
        logger=logger,
        accelerator=cfg.trainer.accelerator,
        max_epochs=cfg.trainer.epochs,
        max_steps=cfg.trainer.steps,
        precision=cfg.trainer.precision,
        deterministic=False,
        callbacks=callbacks,
        log_every_n_steps=1,
        check_val_every_n_epoch=100,
        num_nodes=1,
        use_distributed_sampler=False,
        devices=num_gpus,
        gradient_clip_val=5,
        gradient_clip_algorithm = "norm",
        strategy="ddp" if num_gpus > 1 else "auto",
        # Gradient accumulation keeps the effective batch (clips per optimizer step) when the
        # per-GPU micro-batch has to shrink for memory; the EMA update follows optimizer steps.
        accumulate_grad_batches=int(cfg.trainer.get("accumulate_grad_batches", 1)),
    )


def create_data_module(cfg, nr_patches, masker=None) -> pl.LightningDataModule:
    """Create and configure the data module.

    Loader knobs (``legacy_pipeline``, ``num_workers``, ``prefetch_factor``,
    ``shuffle_buffer``) are read from ``cfg.data`` first, then ``cfg.trainer``,
    then :data:`LOADER_DEFAULTS`. ``masker`` is only used by the legacy pipeline
    (the native pipeline makes the masks in the model); it is created here when
    not given.
    """
    if masker is None:
        masker = ComponentFactory.create_masker(cfg)

    return WebAudioDataModule(
        data_dirs=cfg.data.data_dirs,
        mixing_weights=cfg.data.get("mixing_weights", None),
        batch_size=cfg.trainer.batch_size,
        masker=masker,
        nr_samples_per_audio=cfg.data.samples_per_audio,
        nr_time_points=nr_patches,
        in_channels=cfg.data.in_channels,
        sr = cfg.data.sr,
        legacy_pipeline=bool(_loader_setting(cfg, "legacy_pipeline")),
        num_workers=int(_loader_setting(cfg, "num_workers")),
        prefetch_factor=int(_loader_setting(cfg, "prefetch_factor")),
        shuffle_buffer=int(_loader_setting(cfg, "shuffle_buffer")),
    )

def build_model(cfg, masker=None) -> tuple[torch.nn.Module, int]:
    """Build the complete model with all components.

    Returns the network and the number of tokens per ``process_seconds`` crop.
    ``masker`` (created when ``None``) is passed to the model for the native path.
    """
    factory = ComponentFactory()

    # Create components in order of dependency
    if masker is None:
        masker = factory.create_masker(cfg)
    extractor = factory.create_extractor(cfg)
    network = factory.create_network(cfg, extractor, masker=masker)

    return network, extractor.total_patches(int(cfg.data.sr * cfg.data.process_seconds))


def print_training_info(cfg):
    """Print training information w.r.t to the effective batch size."""
    effective_batch_size = (
        cfg.trainer.batch_size *
        cfg.data.samples_per_audio *
        cfg.trainer.num_gpus
    )
    print(f"Effective Batch Size is: {effective_batch_size}")
    print(
        "Loader: legacy_pipeline={} num_workers={} prefetch_factor={} shuffle_buffer={} | "
        "Model: use_packing={} pad_multiple={} compile_modules={}".format(
            _loader_setting(cfg, "legacy_pipeline"),
            _loader_setting(cfg, "num_workers"),
            _loader_setting(cfg, "prefetch_factor"),
            _loader_setting(cfg, "shuffle_buffer"),
            cfg.trainer.get("use_packing", PACKING_DEFAULTS["use_packing"]),
            cfg.trainer.get("pad_multiple", PACKING_DEFAULTS["pad_multiple"]),
            cfg.trainer.compile_modules,
        )
    )


def cleanup_memory():
    """Clean up GPU and system memory."""
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="./configs", config_name="base")
def main(cfg):
    """Main training function."""
    try:
        # Set random seed for reproducibility
        seed_everything(cfg.seed, workers=True)
        # Chained short jobs (E1, 4 h walltime cap): `resume_last=true` resumes from the run's own
        # last.ckpt when it exists, and exits at once if that checkpoint has already reached trainer.steps.
        ckpt_path = cfg.get("ckpt_path", None)
        if cfg.get("resume_last", False) and ckpt_path is None:
            identity = get_identity_from_cfg(cfg)
            last = f"{cfg.save_dir}/saved_models_jepa_new_masking/{identity.replace('_', '/')}/last.ckpt"
            if os.path.exists(last):
                step = int(torch.load(last, map_location="cpu", weights_only=False).get("global_step", 0))
                if step >= int(cfg.trainer.steps):
                    print(f"resume_last: {last} is at step {step} >= trainer.steps={cfg.trainer.steps}: nothing to do")
                    return
                print(f"resume_last: resuming from {last} (step {step})")
                ckpt_path = last
            else:
                print(f"resume_last: no {last} yet, starting from scratch")

        # Setup training components
        logger = setup_logger(cfg)
        callbacks = setup_callbacks(cfg)
        trainer = setup_trainer(cfg, logger, callbacks)

        # Build model and data (one masker instance shared by both)
        masker = ComponentFactory.create_masker(cfg)
        model, patches = build_model(cfg, masker=masker)
        data_module = create_data_module(cfg, patches, masker=masker)
        # Print training information
        print_training_info(cfg)

        # Start training
        # `ckpt_path=<last.ckpt>` on the command line resumes an interrupted run (optimizer, EMA, step).
        trainer.fit(model, data_module, ckpt_path=ckpt_path)

    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise
    finally:
        cleanup_memory()


if __name__ == "__main__":
    cleanup_memory()  # Clean up before starting
    main()
    cleanup_memory()  # Clean up after finishing
