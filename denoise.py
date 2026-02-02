import gc

import hydra
import pytorch_lightning as pl
import torch
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from utils import get_identity_from_cfg
from data_modules import WebAudioDataModule

from wavjepa.jepa import JEPA
from wavjepa.jepa_denoise import DenoiseJEPA
from wavjepa.simclr_denoise import SimCLRDenoise

from wavjepa.masking import RandomClusterMaskMaker, RandomMaskMaker, TimeInverseBlockMasker, MultiBlockMaskMaker
from wavjepa.extractors import ConvFeatureExtractor, Extractor
from wavjepa.types import TransformerEncoderCFG, TransformerLayerCFG
import wavjepa 
import sys 
sys.modules['sjepa'] = wavjepa

ORIGINAL_SR = 32000

# Component registries
NETWORKS = {"JEPA": DenoiseJEPA, 'SimCLR': SimCLRDenoise}
MASKERS = {"random-masker": RandomMaskMaker, 'random-cluster-masker': RandomClusterMaskMaker, 'time-inverse-masker' : TimeInverseBlockMasker, 'multi-block-masker': MultiBlockMaskMaker}
EXTRACTORS = {"spatial-conv-extractor": ConvFeatureExtractor, 
              "conv-extractor": ConvFeatureExtractor}

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


        if cfg.extractor.name == "spectrogram":
            return extractor_class(
                n_mels = cfg.extractor.n_mels,
                sr = cfg.data.sr,
                embedding_dim = cfg.extractor.embedding_dim,
                in_channels = cfg.data.in_channels,
                fshape = cfg.extractor.fshape,
                tshape= cfg.extractor.tshape,
                fstride= cfg.extractor.fstride,
                tstride= cfg.extractor.tstride,
                trainable= cfg.extractor.trainable
            )
        else:
            # Weight sharing is enabled in only ConvChannelFeatureExtractor
            return extractor_class(
                    conv_layers_spec=eval(cfg.extractor.conv_layers_spec),
                    in_channels=cfg.data.in_channels,
                    depthwise = cfg.extractor.depthwise,
                    share_weights_over_channels = weight_sharing,
                )
    
    
    @staticmethod
    def create_masker(cfg):
        """Create and configure the masker component."""
        masker_class = MASKERS.get(cfg.masker.name)
        if masker_class is None:
            raise ValueError(
                f"Unknown masker type: {cfg.masker.name}. "
                f"Available maskers: {list(MASKERS.keys())}"
            )
        
        if cfg.masker.name == "inverse-masker":
            return masker_class( 
                mask_prob = cfg.masker.mask_prob,
                mask_length = cfg.masker.mask_length,
                channel_based_masking=cfg.masker.channel_based_masking,
            )
        elif cfg.masker.name == "block-masker" or cfg.masker.name=="random-masker" or cfg.masker.name=="random-cluster-masker":
            return masker_class(
                target_masks_per_context  = cfg.masker.target_masks_per_context,
                context_cluster_d = cfg.masker.context_cluster_d,
                context_cluster_u = cfg.masker.context_cluster_u,
                target_cluster_d = cfg.masker.target_cluster_d,
                target_cluster_u = cfg.masker.target_cluster_u,
                channel_based_masking = cfg.masker.channel_based_masking,
            )
        elif cfg.masker.name == "time-inverse-masker":
            return masker_class(
                target_masks_per_context = cfg.masker.target_masks_per_context,
                context_mask_prob = cfg.masker.context_mask_prob,
                context_mask_length = cfg.masker.context_mask_length,
                target_prob = cfg.masker.target_prob,
                target_length = cfg.masker.target_length,
                ratio_cutoff = cfg.masker.ratio_cutoff,
                channel_based_masking = cfg.masker.channel_based_masking,
            )
        else:
            return masker_class(
                mask_prob = cfg.masker.mask_prob,
                cluster= cfg.masker.cluster,
                channel_based_masking=cfg.masker.channel_based_masking,
                )
    
    @staticmethod
    def create_network(cfg, extractor : Extractor) -> JEPA:
        """Create and configure the main network."""
        network_class = NETWORKS.get(cfg.model)
        if network_class is None:
            raise ValueError(
                f"Unknown network type: {cfg.model}. "
                f"Available networks: {list(NETWORKS.keys())}"
            )
        
        try:
            return network_class(
                feature_extractor=extractor,
                transformer_encoder_cfg = TransformerEncoderCFG.create(), 
                transformer_encoder_layers_cfg = TransformerLayerCFG.create(),
                transformer_decoder_cfg = TransformerEncoderCFG.create(), 
                transformer_decoder_layers_cfg = TransformerLayerCFG.create(d_model = 384),
                lr=cfg.optimizer.lr,
                adam_betas=(cfg.optimizer.b1, cfg.optimizer.b2),
                adam_weight_decay=cfg.optimizer.weight_decay,
                in_channels=cfg.data.in_channels,
                resample_sr=cfg.data.sr,
                process_audio_seconds=cfg.data.process_seconds,
                use_gradient_checkpointing =cfg.trainer.use_gradient_checkpointing,
                nr_samples_per_audio=cfg.data.samples_per_audio,
                compile_modules = cfg.trainer.compile_modules,
                average_top_k_layers = cfg.trainer.average_top_k_layers,
                is_spectrogram = cfg.extractor.name == "spectrogram",
                clean_data_ratio = cfg.data.get("clean_data_ratio", 0.0),
                size = cfg.trainer.get("size", "base")
            )
        except Exception as e:
            raise RuntimeError(f"Failed to create network instance: {str(e)}")


def setup_logger(cfg) -> TensorBoardLogger:
    """Set up TensorBoard logger with proper configuration."""
    identity = get_identity_from_cfg(cfg)
    return TensorBoardLogger(
        f"{cfg.save_dir}/tb_logs_jepa_denoised_conv_05/",
        name=identity.replace("_", "/"),
    )


def setup_callbacks(cfg):
    """Set up training callbacks."""
    identity = get_identity_from_cfg(cfg)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.save_dir}/saved_models_jepa_denoised_conv_05/{identity.replace('_', '/')}",
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
        gradient_clip_val=5,
        gradient_clip_algorithm = "norm",
        strategy='ddp_find_unused_parameters_true' if num_gpus > 1 else "auto",
    )


def create_data_module(cfg, nr_patches) -> pl.LightningDataModule:
    """Create and configure the data module."""
    factory = ComponentFactory()
    masker = factory.create_masker(cfg)

    return WebAudioDataModule(
        data_dirs=cfg.data.data_dirs,
        mixing_weights=cfg.data.mixing_weights,
        noise_dir=cfg.data.noise_dir,
        rir_dir=cfg.data.rir_dir,
        batch_size=cfg.trainer.batch_size,
        masker=masker,
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


@hydra.main(version_base=None, config_path="./configs", config_name="base")
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
        weights = torch.load(cfg.trainer.ckpt_weights, weights_only=False)
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

        #Teacher becomes the WavJEPA-Clean
        if cfg.model != "JEPA":
            model._set_teacher(cfg.trainer.ckpt_weights)

        # Start training
        trainer.fit(model, data_module, ckpt_path=None)
        
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise
    finally:
        cleanup_memory()


if __name__ == "__main__":
    cleanup_memory()  # Clean up before starting
    main()
    cleanup_memory()  # Clean up after finishing