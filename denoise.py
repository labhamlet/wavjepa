import gc

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import get_identity_from_cfg_denoise
from data_modules import WebAudioDataModuleDenoiser

from wavjepa.jepa import JEPA
from wavjepa.denoiser import Denoiser

from wavjepa.extractors import ConvFeatureExtractor, Extractor
from wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG
import wavjepa 
import sys 
sys.modules['sjepa'] = wavjepa

ORIGINAL_SR = 32000

EXTRACTORS = {"wav2vec2": ConvFeatureExtractor, 
              "wavjepa": ConvFeatureExtractor}

ENCODERS = {"Transformer" : {"LayerCFG" : TransformerLayerCFG, "EncoderCFG": TransformerEncoderCFG}}

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True


# Enable cuDNN benchmarking for consistent input sizes
torch.backends.cudnn.benchmark = True


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
    def create_network(cfg, extractor : Extractor) -> JEPA:
        """Create and configure the main network."""
        return Denoiser(
            feature_extractor=extractor,
            transformer_encoder_cfg = TransformerEncoderCFG.create(), 
            transformer_encoder_layers_cfg = TransformerLayerCFG.create(),
            lr=cfg.optimizer.lr,
            adam_betas=(cfg.optimizer.b1, cfg.optimizer.b2),
            adam_weight_decay=cfg.optimizer.weight_decay,
            resample_sr=cfg.data.sr,
            process_audio_seconds=cfg.data.process_seconds,
            nr_samples_per_audio=cfg.data.samples_per_audio,
            size = cfg.trainer.get("size", "base"),
            alpha=cfg.trainer.alpha
        )


def setup_logger(cfg) -> TensorBoardLogger:
    """Set up TensorBoard logger with proper configuration."""
    identity = get_identity_from_cfg_denoise(cfg)
    return TensorBoardLogger(
        f"{cfg.save_dir}/tb_logs_jepa_denoised/",
        name=identity.replace("_", "/"),
    )


def setup_callbacks(cfg):
    """Set up training callbacks."""
    identity = get_identity_from_cfg_denoise(cfg)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.save_dir}/saved_models_jepa_denoised/{identity.replace('_', '/')}",
        filename="{step}",
        verbose=True,
        every_n_train_steps=2500,
        save_last=True,
        enable_version_counter=True,
        save_top_k=-1,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")
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
        gradient_clip_val=1.0,
        gradient_clip_algorithm = "norm",
        strategy='ddp_find_unused_parameters_true' if num_gpus > 1 else "auto",
    )


def create_data_module(cfg, nr_patches) -> pl.LightningDataModule:
    """Create and configure the data module."""

    return WebAudioDataModuleDenoiser(
        data_dir=cfg.data.data_dir,
        noise_dir=cfg.data.noise_dir,
        rir_dir=cfg.data.rir_dir,
        batch_size=cfg.trainer.batch_size,
        nr_samples_per_audio=cfg.data.samples_per_audio,
        nr_time_points=nr_patches,
        with_rir=cfg.data.with_rir,
        with_noise=cfg.data.with_noise,
        snr_high=cfg.data.snr_high, 
        snr_low=cfg.data.snr_low
    )


def build_model(cfg) -> torch.nn.Module:
    """Build the complete model with all components."""
    factory = ComponentFactory()
    
    # Create components in order of dependency
    extractor = factory.create_extractor(cfg)
    network = factory.create_network(cfg, extractor)

    return network, extractor.total_patches(int(cfg.data.sr * cfg.data.process_seconds))


def print_training_info(cfg):
    """Print training information w.r.t to the effective batch size."""
    effective_batch_size = (
        cfg.trainer.batch_size * 
        cfg.data.samples_per_audio * 
        cfg.trainer.num_gpus
    )
    print(f"Effective Batch Size is: {effective_batch_size}")


def cleanup_memory():
    """Clean up GPU and system memory."""
    gc.collect()
    torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="./configs", config_name="denoise")
def main(cfg):
    """Main training function."""
    try:
        # Set random seed for reproducibility
        seed_everything(cfg.seed, workers=True)
        
        # Setup training components
        logger = setup_logger(cfg)
        callbacks = setup_callbacks(cfg)
        trainer = setup_trainer(cfg, logger, callbacks)
        
        # Build model and data
        model, patches = build_model(cfg)
        data_module = create_data_module(cfg, patches)
        # Print training information
        print_training_info(cfg)

        #Load WavJEPA-Clean weights.
        weights = torch.load(cfg.trainer.teacher_ckpt_weights, weights_only=False)
        new_state_dict = {}
        for key, value in weights["state_dict"].items():
            if key.startswith("extract_audio._orig_mod"):
                new_key = key.replace("extract_audio._orig_mod", "extract_audio")
                new_state_dict[new_key] = value
            elif key.startswith("encoder._orig_mod"):
                new_key = key.replace("encoder._orig_mod", "encoder")
                new_state_dict[new_key] = value
            elif key.startswith("decoder._orig_mod"):
                new_key = key.replace("decoder._orig_mod", "decoder")
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value

        model.load_state_dict(new_state_dict, strict=False)
        model._set_teacher(cfg.trainer.teacher_ckpt_weights)            

        # Start training
        trainer.fit(model, data_module, ckpt_path=cfg.ckpt_path if "ckpt_path" in cfg else None)
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise
    finally:
        cleanup_memory()


if __name__ == "__main__":
    cleanup_memory()  # Clean up before starting
    main()
    cleanup_memory()  # Clean up after finishing