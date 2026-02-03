import copy
import transformers 
import numpy as np 

from typing import List, Any, Optional, Tuple

import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
import torchaudio.functional as F_audio
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl


from wavjepa.pos_embed import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed, get_binaural_pos_embed

from wavjepa.functions import trunc_normal_
from wavjepa.extractors.audio_extractor import Extractor
from wavjepa.types import ForwardReturn, TransformerLayerCFG, TransformerEncoderCFG
from wavjepa.jepa import JEPA 
from wavjepa.extractors import ConvFeatureExtractor

from data_modules.scene_module import generate_scenes_batch, generate_scenes
from data_modules.dataset_functions import pad_or_truncate_batch, normalize_audio_batch, normalize_audio
from data_modules.scene_module.generate_scenes_batch import convolve_with_rir

ORIGINAL_SR=32000
#Think about using weight decay from 0.04 to 0.4?
torch._dynamo.config.capture_dynamic_output_shape_ops = True

def collate_fn(batch : List[torch.Tensor]) -> torch.Tensor:
    return batch.flatten(start_dim = 0, end_dim = 1)

def resample(audio: torch.Tensor, resample_sr: int, original_sr = 32000) -> torch.Tensor:
    return torchaudio.functional.resample(
        audio,
        original_sr,
        resample_sr,
        lowpass_filter_width=64,
        rolloff=0.9475937167399596,
        resampling_method="sinc_interp_kaiser",
        beta=14.769656459379492,
    )


class Denoiser(pl.LightningModule):
    """
    Joint-Embedding Predictive Architecture (JEPA).

    This implementation is inspired by:
        * I-JEPA http://arxiv.org/abs/2301.08243
        * Data2vec 2.0 http://arxiv.org/abs/2212.07525

    Args:
        feature_encoder:
            Does the local feature encoding. Will be shared between teacher and student.

                * Input: dict with keys: ``**batch``
                * Output: ``local_features`` (batch_size, n_patches, emb_dim)

        mask_maker:
            Computes the training masks as indices.

                * Input: dict with keys:
                    - ``**batch``
                    - ``local_features`` (batch_size, n_patches, emb_dim)

                * output: tuple:
                    - ``idxs_context`` (batch_size, self.n_contexts_per_input, n_context_patches)
                    - ``idxs_target`` (batch_size, self.n_contexts_per_input, self.n_targets_per_context, n_target_patches)

        transformer_kwargs:
            Arguments for :class:`nn.Transformer`. The transformer will have the
            following signature:

                * Input: (batch_size, n_context_patches, emb_dim)
                * Output: (batch_size, n_target_patches, emb_dim)

        loss_fn: nn.Module
            Loss function to use between the ``preds`` (output of the transformer)
            and the ``targets``.
        ema_decay: float
            initial ema decay rate.
        ema_end_decay: float
            final ema decay rate.
        ema_anneal_end_step: int
            when to finish annealing ema decay rate.
        average_top_k_layers: int
            The targets are the average of the outputs of the last k layers of
            the teacher encoder. This parameter specifies the number of layers to
            use for the average.
    """
    TARGET_SECONDS: int = 10
    def __init__(
        self,
        feature_extractor: Extractor,
        transformer_encoder_layers_cfg : TransformerLayerCFG,
        transformer_encoder_cfg : TransformerEncoderCFG,
        lr: float = 0.0002,
        adam_betas: tuple[float, float] = (0.9, 0.98),        
        adam_eps: float = 1e-06,
        adam_weight_decay: float = 0.01,
        resample_sr : int = 16000,
        process_audio_seconds: float = 2.00,
        in_channels : int = 2,
        nr_samples_per_audio = 16,
        size : str = "base",
        **kwargs : dict[str, Any],
    ):
        super().__init__(**kwargs)
        self.resampler_48k = torchaudio.transforms.Resample(48000, ORIGINAL_SR).to(
            self.device
        )
        self.resampler_44k = torchaudio.transforms.Resample(44100, ORIGINAL_SR).to(
            self.device
        )
        self.valid_len_44k = int(self.TARGET_SECONDS * 44100)
        self.valid_len_32k = int(self.TARGET_SECONDS * 32000)
        self.target_audio_length = self.TARGET_SECONDS * ORIGINAL_SR

        self.sr = resample_sr 
        self.process_audio_seconds = process_audio_seconds
        self.nr_samples_per_audio = nr_samples_per_audio
        self.target_length = int(resample_sr * process_audio_seconds)
        self.total_patches = feature_extractor.total_patches(self.target_length)
        self.in_channels = in_channels
        self.save_hyperparameters(
            ignore=["feature_encoder", "feature_extractor", "loss_fn"]
        )
        self.extract_audio = feature_extractor
        self.feature_norms : nn.Module = nn.LayerNorm(self.extract_audio.embedding_dim)

        # If size is large, then alter the encoder parameters to mimic VIT-Large. Should results in ~300m parameters.
        if size == "large": 
            transformer_encoder_layers_cfg["nhead"] = 16
            transformer_encoder_layers_cfg["d_model"] = 1024
            transformer_encoder_layers_cfg["dim_feedforward"] = 1024 * 4
            transformer_encoder_cfg["num_layers"] = 24


        self.n_encoder_heads = transformer_encoder_layers_cfg["nhead"]
        self.encoder_embedding_dim = transformer_encoder_layers_cfg["d_model"]

        encoder_layer = nn.TransformerEncoderLayer(**transformer_encoder_layers_cfg)
        self.encoder = nn.TransformerEncoder(encoder_layer, norm = nn.LayerNorm(self.encoder_embedding_dim), **transformer_encoder_cfg)
        self.post_extraction_mapper : Optional[nn.Module] = nn.Linear(feature_extractor.embedding_dim, self.encoder_embedding_dim) if feature_extractor.embedding_dim != self.encoder_embedding_dim else None

        self.pos_encoding_encoder = self._get_pos_embed_params(self.encoder_embedding_dim)


    def _set_teacher(self, weights_ckpt): 
        weights = torch.load(weights_ckpt, weights_only=False)
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

        extractor = ConvFeatureExtractor(
            conv_layers_spec=eval("[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)]"),
            in_channels=1,
        )


        model = JEPA(
            feature_extractor=extractor,
            transformer_encoder_cfg=TransformerEncoderCFG.create(),
            transformer_encoder_layers_cfg=TransformerLayerCFG.create(),
            transformer_decoder_cfg=TransformerEncoderCFG.create(),
            transformer_decoder_layers_cfg=TransformerLayerCFG.create(d_model=384),
            in_channels=self.in_channels,
            resample_sr=self.sr,
            size="base",
            is_spectrogram=False,
            process_audio_seconds=self.process_audio_seconds,
        )
        model.load_state_dict(new_state_dict, strict=False)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        self.teacher = model

    def _compile(self):

        try:
            self.encoder_forward = torch.compile(self.encoder_forward, fullgraph=True)
            self.extract_audio = torch.compile(self.extract_audio)
            self.pad_or_truncate_batch = torch.compile(pad_or_truncate_batch)
            self.collate_fn = torch.compile(collate_fn)
            self.resample = torch.compile(resample) 

        except Exception as e:
            print(f"Warning: Could not compile operations: {e}")
            self.use_compiled_forward = False
            self.pad_or_truncate_batch = pad_or_truncate_batch
            self.collate_fn = collate_fn
            self.resample = resample


    def _get_pos_embed_params(self, embedding_dim):
        """Calculates the pos embedding embedding parameters and returns them."""
        # Update positional embedding
        pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.total_patches,
                embedding_dim,
            ),
            requires_grad=False,
        )
        positions = np.arange(self.total_patches, dtype=np.float64)
        #TODO! Remove this total patches later.
        if self.in_channels == 2 and (self.total_patches == 400):
            # We use 1D sincos embeddings with channel number indicated on the last 384 dimensions.
            print("Using Binaural Positional Embeddings")
            pos_embed_data = get_binaural_pos_embed(embedding_dim, time_steps=self.total_patches // self.in_channels
            )
        elif self.in_channels == 2 and (self.total_patches == 200):
            #Use 1D pos_embeddings if channel-mixing feature extractor
            pos_embed_data = get_1d_sincos_pos_embed_from_grid(
                embedding_dim,
                positions,
            )     
        elif self.in_channels == 1 and (self.total_patches == 200):
            # IF it is plain audio, we used 1d sincos embeddings
            pos_embed_data = get_1d_sincos_pos_embed_from_grid(
                embedding_dim,
                positions,
            )
        else:
            raise Exception(f"Not implemented for more in_channels, {self.in_channels}, {self.total_patches}")
        pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float().unsqueeze(0))
        return pos_embed


    def _compile_operations(self):
        """
        Use torch.compile on the extractor, encoder and decoder blocks for faster forward
        """
        try:
            self.encoder_forward = torch.compile(self.encoder_forward, fullgraph=True)
            self.extract_audio = torch.compile(self.extract_audio)

        except Exception as e:
            print(f"Warning: Could not compile operations: {e}")
            self.use_compiled_forward = False

    def configure_optimizers(self):
        trainables = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainables,
            lr=self.hparams.lr,
            betas=self.hparams.adam_betas,
            eps=self.hparams.adam_eps,
            weight_decay=self.hparams.adam_weight_decay,
        )
        cosine_annealing = transformers.get_cosine_schedule_with_warmup(optimizer,
                                 num_warmup_steps=5000, num_training_steps=self.trainer.max_steps)

        return {"optimizer": optimizer,
                'lr_scheduler' : {"scheduler": cosine_annealing, "interval": "step"}}


    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Runs on GPU. Splits batch by SR, resamples, recombines.
        """

        def index_select_and_normalize(audio, indices):
            audio = audio.unsqueeze(1).expand(-1, self.nr_samples_per_audio, -1, -1)
            return_audio = torch.gather(audio, 3, indices)

            mean = return_audio.mean(dim=(-2, -1), keepdim=True)
            std = return_audio.std(dim=(-2, -1), keepdim=True)
            normalized = (return_audio - mean) / (std + 1e-5) # Add epsilon for stability
            return normalized 

        # Unpack batch (Audio is all 480,000 length here, and noise is 320,000)
        (
            audio_batch,
            sr_batch,
            source_rir,
            noise,
            noise_lengths,
            snr,
            ctx_masks,
            target_indices,
            ctx_and_target_masks,
        ) = batch

        placed_noise_batch = [None]

        batch_size = audio_batch.shape[0]
        
        # 2. RESAMPLE AUDIO (Vectorized)
        mask_48k = sr_batch == 48000
        mask_44k = sr_batch == 44100
        mask_32k = sr_batch == 32000  # resampled librispeech

        final_audio = torch.zeros(
            (batch_size, self.target_audio_length),
            device=self.device,
            dtype=audio_batch.dtype,
        )

        if mask_48k.any():
            audio = normalize_audio_batch(audio_batch[mask_48k])
            final_audio[mask_48k] = self.resampler_48k(audio)
        if mask_32k.any():
            # If audio was 32khz we had lots of padding to match 10 seconds od 48kHz,
            # Remove that padding.
            audio = normalize_audio_batch(
                audio_batch[mask_32k][..., : self.valid_len_32k]
            )
            final_audio[mask_32k] = audio
        if mask_44k.any():
            # If audio was 44.1khz we had lots of padding to match 10 seconds od 48kHz,
            # Remove that padding.
            audio = normalize_audio_batch(
                audio_batch[mask_44k][..., : self.valid_len_44k]
            )
            final_audio[mask_44k] = self.resampler_44k(audio)

        #Get the noise
        if noise[0] is not None:
            # This is the real length of the noise actually.
            placed_noise_batch = torch.zeros_like(final_audio)
            for i in range(batch_size):
                #Did we pad the noise?
                valid_len = min(noise_lengths[i].item(), noise.shape[-1])
                current_noise = noise[i, :valid_len]

                #Normalize only the non-faded part.
                current_noise = normalize_audio(current_noise)
                #Fade in and out the noise.
                #If the real length was longer than the audio 
                #we apply both fade-in and fade-out
                #Else
                #we apply fade-out only.
                current_noise = generate_scenes.fade_noise(
                    noise_lengths[i].item(), current_noise, final_audio[i], ORIGINAL_SR
                )
                
                #If target audio length is bigger than the valid length
                #Place the noise randomly.
                if self.target_audio_length > valid_len:
                    start_idx = torch.randint(
                        0, self.target_audio_length - valid_len, (1,)
                    ).item()
                    # Place current noise randomly to the audio file.
                    placed_noise_batch[i, start_idx : start_idx + valid_len] = (
                        current_noise
                    )
                else: 
                    placed_noise_batch[i] = current_noise[: self.target_audio_length]
        
        # Generate a naturalistic scene
        # This handles the sitatuion when rir is [None, None], and placed_noise_batch is [None, None]
        generated_scene = generate_scenes_batch.generate_scene(
            source_rir=source_rir,
            source=final_audio,
            noise=placed_noise_batch,
            snr=snr       
        )
  
        generated_scene = self.pad_or_truncate_batch(generated_scene, 10 * ORIGINAL_SR)

        #Add channel dimension to the final audio as well.
        if final_audio.ndim != 3:
            final_audio = final_audio.unsqueeze(1)

        assert generated_scene.ndim == final_audio.ndim


        clean_audio = pad_or_truncate_batch(final_audio, 10 * ORIGINAL_SR)
        # We know that the original sr is 32000.
        if self.sr != ORIGINAL_SR:
            generated_scene = self.resample(generated_scene, resample_sr = self.sr, original_sr = ORIGINAL_SR)
            clean_scene = self.resample(clean_audio, resample_sr=self.sr, original_sr=ORIGINAL_SR)

        assert generated_scene.shape[1] <= self.in_channels, f"Generated scene has more channels than in channels, {generated_scene.shape}, {self.in_channels}"
        

        B, C, L_full = generated_scene.shape

        # Generate all random start indices at once
        rand_starts = torch.randint(
            0, L_full - self.target_length + 1,
            (B, self.nr_samples_per_audio),
            device=self.device
        )

        # Create indices for gathering
        # Shape: (B, nr_samples, target_length)
        indices = rand_starts.unsqueeze(-1) + torch.arange(self.target_length, device=self.device)
        indices = indices.unsqueeze(2).expand(-1, -1, C, -1)

        normalized_clean_audios = index_select_and_normalize(clean_scene, indices)
        normalized_generated_audios = index_select_and_normalize(generated_scene, indices)

        # Cast to bfloat16 and flatten batch and samples dimensions
        flattened_clean = self.collate_fn(normalized_clean_audios.to(torch.bfloat16))
        flattened_generated = self.collate_fn(normalized_generated_audios.to(torch.bfloat16))

        # Shuffle the samples
        idx = torch.randperm(flattened_generated.size(0))

        return flattened_generated[idx, ...], flattened_clean[idx, ...]

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> ForwardReturn:
        generated_scene, clean_scene = batch
        out = self(generated_scene, clean_scene)

        # Enhanced logging
        log_data = {
            "train/loss": out['loss'],
            "loss_clean": out["loss_clean"],
            "loss_denoise_dereverb": out["loss_denoise_dereverb"],
        }
            
        self.log_dict(log_data, prog_bar=True, sync_dist=True)

        return out

    

    def forward(self, generated_scene: torch.Tensor, clean_scene : torch.Tensor) -> ForwardReturn:
        """
        Args:
            batch: torch.Tensor
                audio data

        Returns:
            output: dict with keys:

                * loss: scalar
                * local_features: (batch_size, n_patches, emb_dim)
                    Output of ``feature_encoder``. Shared between teacher and student.
                * contextual_features: (batch_size, n_patches, emb_dim) if compute_loss=False, (batch_size, n_contexts_per_input, n_context_patches, emb_dim) if compute_loss=True.
                    Outpyt of the student ``tramnsformer.encoder``.
                * preds: (batch_size, n_contexts_per_input, n_targets_per_context, n_target_patches, emb_dim)
                    Output of the student transformer (encoder+decoder).
                * targets: (batch_size, n_patches, emb_dim)
                    Average of the outputs of the last k layers of the
                    teacher ``transformer.encoder``.
                * idxs_context: (batch_size, n_contexts_per_input, n_context_patches)
                    Indices of the unmasked patches to use to compute the contextual features.
                * idxs_target: (batch_size, n_contexts_per_input, n_targets_per_context, n_target_patches)
                    Indices of the patches that must be predicted by the student.
        """

        def _forward_features(scene):
            #Generated Audio
            local_features_generated = self.extract_audio(scene)
            local_features_generated = self.feature_norms(local_features_generated)
            if self.post_extraction_mapper is not None:
                local_features_generated = self.post_extraction_mapper(local_features_generated)
            local_features_generated = local_features_generated + self.pos_encoding_encoder 
            return self.encoder_forward(local_features_generated, src_key_padding_mask=None)

        #Clean Audio
        contextual_features_clean = _forward_features(clean_scene)
        contextual_features_generated = _forward_features(generated_scene)

        #Clean audio WavJEPA-Clean produces this!
        with torch.no_grad(): 
            clean_targets = self.teacher.get_audio_representation(clean_scene, padding_mask=None)
            clean_targets = clean_targets.clone()

        loss_clean = -F.cosine_similarity(contextual_features_clean.flatten(0,1), clean_targets.flatten(0,1), dim=-1).mean()
        loss_denoise_dereverb = -F.cosine_similarity(contextual_features_generated.flatten(0,1), clean_targets.flatten(0,1), dim=-1).mean()   
        
        loss = (0.8 * loss_clean) + (0.2 * loss_denoise_dereverb)
        
        return ForwardReturn(
            loss=loss,
            loss_clean=loss_clean,
            loss_denoise_dereverb = loss_denoise_dereverb,
        )



    #TODO use flex attention
    def encoder_forward(self, 
    x_contexts: torch.Tensor, 
    src_key_padding_mask : Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:

        contextual_features = self.encoder(x_contexts, src_key_padding_mask = src_key_padding_mask)
        return contextual_features
