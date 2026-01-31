import copy
import transformers 
import numpy as np 

from typing import List, Any, Optional, Tuple

import torch
import torchaudio
import random
from torch import nn
from einops import repeat, rearrange
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl


from wavjepa.pos_embed import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed, get_binaural_pos_embed

from wavjepa.functions import trunc_normal_
from wavjepa.extractors.audio_extractor import Extractor
from wavjepa.types import ForwardReturn, TransformerLayerCFG, TransformerEncoderCFG
from data_modules.scene_module import generate_scenes_batch, generate_scenes
from data_modules.dataset_functions import pad_or_truncate_batch, normalize_audio_batch, normalize_audio

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


class JEPA(pl.LightningModule):
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
    teacher_encoder: nn.Module
    def __init__(
        self,
        feature_extractor: Extractor,
        transformer_encoder_layers_cfg : TransformerLayerCFG,
        transformer_encoder_cfg : TransformerEncoderCFG,
        transformer_decoder_layers_cfg : TransformerLayerCFG,
        transformer_decoder_cfg : TransformerEncoderCFG,
        decoder_embedding_dim : int = 512,
        loss_fn: nn.Module = nn.MSELoss(reduction='none'),
        lr: float = 0.0002,
        adam_betas: tuple[float, float] = (0.9, 0.98),        
        adam_eps: float = 1e-06,
        adam_weight_decay: float = 0.01,
        ema_decay: float = 0.999,
        ema_end_decay: float = 0.99999,
        ema_anneal_end_step: int = 100000,
        average_top_k_layers: int = 12,
        resample_sr : int = 16000,
        process_audio_seconds: float = 2.00,
        in_channels : int = 2,
        nr_samples_per_audio = 16,
        use_gradient_checkpointing: bool = False,
        compile_modules : bool = False,
        is_spectrogram : bool = True,
        clean_data_ratio : float = 0.0,
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
        self.clean_data_ratio = clean_data_ratio

        self.sr = resample_sr 
        self.is_spectrogram = is_spectrogram
        self.nr_samples_per_audio = nr_samples_per_audio
        self.ema_end_step = ema_anneal_end_step
        self.target_length = int(resample_sr * process_audio_seconds)
        self.total_patches = feature_extractor.total_patches(self.target_length)
        self.use_compiled_forward = compile_modules
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.in_channels = in_channels
        self.save_hyperparameters(
            ignore=["feature_encoder", "feature_extractor", "loss_fn"]
        )
        self.extract_audio = feature_extractor
        self.feature_norms : nn.Module = nn.LayerNorm(self.extract_audio.embedding_dim)
        self.loss_fn = loss_fn


        # If size is large, then alter the encoder parameters to mimic VIT-Large. Should results in ~300m parameters.
        if size == "large": 
            transformer_encoder_layers_cfg["nhead"] = 16
            transformer_encoder_layers_cfg["d_model"] = 1024
            transformer_encoder_layers_cfg["dim_feedforward"] = 1024 * 4
            transformer_encoder_cfg["num_layers"] = 24


        self.n_encoder_heads = transformer_encoder_layers_cfg["nhead"]
        self.encoder_embedding_dim = transformer_encoder_layers_cfg["d_model"]
        self.n_decoder_heads = transformer_decoder_layers_cfg["nhead"]
        self.decoder_embedding_dim = transformer_decoder_layers_cfg["d_model"]

        encoder_layer = nn.TransformerEncoderLayer(**transformer_encoder_layers_cfg)
        self.encoder = nn.TransformerEncoder(encoder_layer, norm = nn.LayerNorm(self.encoder_embedding_dim), **transformer_encoder_cfg)
        self.post_extraction_mapper : Optional[nn.Module] = nn.Linear(feature_extractor.embedding_dim, self.encoder_embedding_dim) if feature_extractor.embedding_dim != self.encoder_embedding_dim else None
        decoder_layer = nn.TransformerEncoderLayer(**transformer_decoder_layers_cfg)
        self.decoder = nn.TransformerEncoder(decoder_layer, norm = nn.LayerNorm(self.decoder_embedding_dim), **transformer_decoder_cfg)
        self.decoder_to_encoder_mapper = nn.Linear(self.decoder_embedding_dim, self.encoder_embedding_dim, bias=True)
        self.encoder_to_decoder_mapper = nn.Linear(self.encoder_embedding_dim, self.decoder_embedding_dim)

        # For the autocast add batch dimensions.
        self.mask_token = nn.Parameter(
            torch.zeros(1, 1, self.decoder_embedding_dim, requires_grad=True)
        )
        torch.nn.init.normal_(self.mask_token, std=0.02)
        self.pos_encoding_encoder = self._get_pos_embed_params(self.encoder_embedding_dim)
        self.pos_encoding_decoder = self._get_pos_embed_params(self.decoder_embedding_dim)

        self.apply(self._init_weights)
        self._init_teacher()
        if compile_modules:
            self._compile_operations()
            self.pad_or_truncate_batch = torch.compile(pad_or_truncate_batch)
            self.collate_fn = torch.compile(collate_fn)
            self.resample = torch.compile(resample) 
        else:
            self.pad_or_truncate_batch = pad_or_truncate_batch
            self.collate_fn = collate_fn
            self.resample = resample

    def _init_weights(self, m : nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: # type: ignore
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

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
        if self.is_spectrogram:
            # If it is a spectrogram, we use 2d sincos embeddings.
            pos_embed_data = get_2d_sincos_pos_embed(
                embedding_dim, self.extract_audio.grid_size, cls_token_num=0
            )
        #TODO! Remove this total patches later.
        elif not self.is_spectrogram and self.in_channels == 2 and (self.total_patches == 400):
            # We use 1D sincos embeddings with channel number indicated on the last 384 dimensions.
            print("Using Binaural Positional Embeddings")
            pos_embed_data = get_binaural_pos_embed(embedding_dim, time_steps=self.total_patches // self.in_channels
            )
        elif not self.is_spectrogram and self.in_channels == 2 and (self.total_patches == 200):
            #Use 1D pos_embeddings if channel-mixing feature extractor
            pos_embed_data = get_1d_sincos_pos_embed_from_grid(
                embedding_dim,
                positions,
            )     
        elif not self.is_spectrogram and self.in_channels == 1 and (self.total_patches == 200):
            # IF it is plain audio, we used 1d sincos embeddings
            pos_embed_data = get_1d_sincos_pos_embed_from_grid(
                embedding_dim,
                positions,
            )
        else:
            raise Exception(f"Not implemented for more in_channels, {self.in_channels}, {self.total_patches}")
        pos_embed.data.copy_(torch.from_numpy(pos_embed_data).float().unsqueeze(0))
        return pos_embed

    def _init_teacher(self):
        self.teacher_encoder = copy.deepcopy(self.encoder)
        self.teacher_encoder.requires_grad_(False)

    def _get_ema_decay(self):
        if self.global_step >= self.ema_end_step:
            return self.hparams.ema_end_decay
        r = self.hparams.ema_end_decay - self.hparams.ema_decay
        pct_remaining = 1 - self.global_step / self.ema_end_step
        return self.hparams.ema_end_decay - r * pct_remaining

    @torch.no_grad()
    def _step_teacher(self):
        r = self._get_ema_decay()
        for student, teacher in zip(self.encoder.parameters(), self.teacher_encoder.parameters()):
            teacher.data.mul_(r).add_((1 - r) * student.detach().data)

    def _compile_operations(self):
        """
        Use torch.compile on the extractor, encoder and decoder blocks for faster forward
        """
        try:
            self.encoder_forward = torch.compile(self.encoder_forward, fullgraph=True)
            self.decoder_forward = torch.compile(self.decoder_forward, fullgraph=True)
            self._forward_teacher = torch.compile(self._forward_teacher, fullgraph=True)
            self.extract_audio = torch.compile(self.extract_audio)
            self.masked_loss = torch.compile(self.masked_loss)

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
                                 num_warmup_steps=100000, num_training_steps=self.trainer.max_steps)

        return {"optimizer": optimizer,
                'lr_scheduler' : {"scheduler": cosine_annealing, "interval": "step"}}

    def _make_targets(self, layer_outputs : List[torch.Tensor]):
        """
        Predicting targets which are the average of multiple layers is more robust than 
        predicting only the top most layer (K = 1) for most modalities.
        Args:
            layer_outputs: average_top_k_layers * (batch_size, n_patches, emb_dim)

        Returns:
            array of shape (batch_size, n_patches, emb_dim)
        """

        # They have for audioset -> instance_norm_target_layer: true
        # They have for audioset -> layer_norm_targets : true
        # So this is the way following the data2vec2 paper for audio.
        stacked_outputs = torch.stack(layer_outputs, )  # [num_layers, batch, seq_len, features]
        transposed = stacked_outputs.transpose(2, 3)   # [num_layers, batch, features, seq_len]

        # Apply instance norm to all layers simultaneously
        normalized = F.instance_norm(transposed)       # [num_layers, batch, features, seq_len]
        normalized = normalized.transpose(2, 3)        # [num_layers, batch, seq_len, features]

        # Compute mean across layers
        y = normalized.mean(dim=0)                     # [batch, seq_len, features]
        return y

    @torch.no_grad()
    def _forward_teacher(self, x : torch.Tensor) -> torch.Tensor:
        layer_outputs = []
        for i, bl in enumerate(self.teacher_encoder.layers): # type: ignore
            x : torch.Tensor = bl(x)
            if (
                len(self.teacher_encoder.layers) - i
                <= self.hparams.average_top_k_layers
            ):
                layer_outputs.append(x)

        if self.hparams.average_top_k_layers > 1:
            targets = self._make_targets(layer_outputs)  # (batch_size, n_patches, emb_dim)
        else:
            targets = layer_outputs[-1]
        return targets

    def get_aug_prob(self):
        return 1 - (self.global_step / self.trainer.max_steps)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """
        Runs on GPU. Splits batch by SR, resamples, recombines.
        """
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
        
        progress = self.global_step / max(1, self.trainer.max_steps)
        progress =  max(0.0, min(1.0, progress)) # Clamp between 0 and 1
        
        # At step 0: Boost SNR by +20dB (Noise is very quiet)
        # At step Max: Boost SNR by +0dB (Noise is at normal/hard levels)
        start_snr_boost = 40.0 
        current_snr_boost = start_snr_boost * (1.0 - progress)

        print(f"Average SNR is: {(snr + current_snr_boost).mean()}")

        # Generate a naturalistic scene
        # This handles the sitatuion when rir is [None, None], and placed_noise_batch is [None, None]
        generated_scene = generate_scenes_batch.generate_scene(
            source_rir=source_rir,
            source=final_audio,
            noise=placed_noise_batch,
            snr=snr + current_snr_boost      
        )

        generated_scene = self.pad_or_truncate_batch(generated_scene, 10 * ORIGINAL_SR)
        #Add channel dimension to the final audio as well.
        if final_audio.ndim != 3:
            final_audio = final_audio.unsqueeze(1)
        assert generated_scene.ndim == final_audio.ndim

        clean_audio = self.pad_or_truncate_batch(final_audio, 10 * ORIGINAL_SR)
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

        # Expand scene and indices for gathering
        # scene: (B, C, L_full) -> (B, 1, C, L_full) -> (B, nr_samples, C, L_full)
        generated_scene_expanded = generated_scene.unsqueeze(1).expand(-1, self.nr_samples_per_audio, -1, -1)
        # indices: (B, nr_samples, target_length) -> (B, nr_samples, 1, target_length) -> (B, nr_samples, C, target_length)
        indices_expanded = indices.unsqueeze(2).expand(-1, -1, C, -1)

        # Gather all audio windows in one operation
        # Shape: (B, nr_samples, C, target_length)
        return_generated_audios = torch.gather(generated_scene_expanded, 3, indices_expanded)

        # To preserve ITD and ILD, normalize jointly across channels and time.
        # Calculate mean and std over the last two dimensions (C, L).
        mean = return_generated_audios.mean(dim=(-2, -1), keepdim=True)
        std = return_generated_audios.std(dim=(-2, -1), keepdim=True)
        normalized_generated_audios = (return_generated_audios - mean) / (std + 1e-5) # Add epsilon for stability


        clean_scene_expanded = clean_scene.unsqueeze(1).expand(-1, self.nr_samples_per_audio, -1, -1)

        return_clean_audios = torch.gather(clean_scene_expanded, 3, indices_expanded)

        mean = return_clean_audios.mean(dim=(-2, -1), keepdim=True)
        std = return_clean_audios.std(dim=(-2, -1), keepdim=True)
        normalized_clean_audios = (return_clean_audios - mean) / (std + 1e-5) # Add epsilon for stability

        # Cast to bfloat16 and flatten batch and samples dimensions
        flattened_generated = self.collate_fn(normalized_generated_audios.to(torch.bfloat16))
        flattened_clean = self.collate_fn(normalized_clean_audios.to(torch.bfloat16))

        # Shuffle the samples
        idx = torch.randperm(flattened_generated.size(0))

        return flattened_generated[idx, ...], flattened_clean[idx, ...], self.collate_fn(ctx_masks), self.collate_fn(target_indices), self.collate_fn(ctx_and_target_masks)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> ForwardReturn:
        generated_scene, clean_scene, ctx_masks, target_indices, ctx_and_target_masks = batch
        out = self(generated_scene, clean_scene, ctx_masks, target_indices, ctx_and_target_masks)

        # Enhanced logging
        log_data = {
            "train/loss": out['loss'],
            "ema" : self._get_ema_decay(),
        }
            
        self.log_dict(log_data, prog_bar=True, sync_dist=True)

        with torch.amp.autocast('cuda', enabled=False):  # Force FP32 computation for stability
            self._step_teacher()
        
        return out

    def masked_loss(self, pred, target, target_indices):
        """
        Calculates the masked loss using broadcasting to avoid memory-heavy repeats.

        pred:   Tensor of shape [(B * N), T, D]
        target: Tensor of shape [B, T, D]
        mask:   Tensor of shape [B, N, T]
        """
        B, N, _ = target_indices.shape
        D = pred.shape[-1]

        pred_reshaped = pred.view(B, N, -1, D)

        # This makes it broadcastable to the shape of pred_reshaped [B, N, T, D] during the loss
        
        target = repeat(target, "B T D -> B N T D", N = N)
        loss = self.loss_fn(pred_reshaped, target)  # -> Shape: [B, N, T, D]

        loss_per_timestep = loss.mean(dim=-1)  # -> Shape: [B, N, T]

        # No rearrange is needed for the mask.
        masked_loss_tensor = loss_per_timestep * target_indices  # -> Shape: [B, N, T]

        # Calculate the final mean loss over only the masked elements.
        total_loss = masked_loss_tensor.sum()
        indices_count = target_indices.sum()

        return total_loss / (indices_count + 1e-8)
    

    def forward(self, generated_scene: torch.Tensor, clean_scene : torch.Tensor, ctx_masks, target_indices, ctx_and_target_masks) -> ForwardReturn:
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
        # # Compute the local representations from the waveform
        # This extract audio can be also channel based, if it is channel based the channel are flatten to the sequencel length
        local_features_generated = self.extract_audio(generated_scene)
        local_features_generated = self.feature_norms(local_features_generated)
        if self.post_extraction_mapper is not None:
            local_features_generated = self.post_extraction_mapper(local_features_generated)
        
        local_features_generated = local_features_generated + self.pos_encoding_encoder

        contextual_features = self.encoder_forward(local_features_generated, src_key_padding_mask=ctx_masks)
        # Accumulate contextual features on the batch dimensions
        contextual_features = contextual_features[~ctx_masks]
        contextual_features = self.encoder_to_decoder_mapper(contextual_features)
        
        preds = self.decoder_forward(contextual_features, ctx_masks, nr_targets = target_indices.shape[1], src_key_padding_mask=ctx_and_target_masks)
        
        # Compute the training targets using the teacher.
        local_features_clean = self.extract_audio(clean_scene)
        local_features_clean = self.feature_norms(local_features_clean)
        if self.post_extraction_mapper is not None:
            local_features_clean = self.post_extraction_mapper(local_features_clean)
        
        local_features_clean = local_features_clean + self.pos_encoding_encoder
        local_features_clean = local_features_clean.detach()
        targets = self._forward_teacher(local_features_clean)

        loss = self.masked_loss(preds, targets, target_indices)
        
        return ForwardReturn(
            local_features_clean=local_features_clean,
            local_features_generated=local_features_generated,
            contextual_features=contextual_features,
            loss=loss,
            preds=preds,
            targets=targets,
        )


    def decoder_forward(self, contextual_features: torch.Tensor, ctx_mask: torch.BoolTensor, nr_targets : int, src_key_padding_mask : Optional[torch.BoolTensor] = None) -> torch.Tensor:
        B = ctx_mask.shape[0]
        # Prepare the mask tokens.
        tgt = self.mask_token.repeat(B, self.total_patches, 1).type_as(contextual_features) # (B, seq_len, decoder_dim)
        tgt[~ctx_mask, :] = contextual_features.reshape((-1, self.decoder_embedding_dim))
        tgt = tgt.reshape((B, -1, self.decoder_embedding_dim))
        # Add positional encoding to the decoder
        tgt = tgt + self.pos_encoding_decoder

        # Repeat the context for every target, and absorb into batch dimension
        tgt = repeat(tgt, 'B Seq Emb -> B T Seq Emb', T = nr_targets)
        tgt = rearrange(tgt, 'B T Seq Emb -> (B T) Seq Emb')
        src_key_padding_mask = rearrange(src_key_padding_mask, 'B T Seq1 -> (B T) Seq1')
        
        #Decoder only attends to context tokens and target mask tokens.
        tgt = self.decoder(tgt, src_key_padding_mask = src_key_padding_mask)
        preds = self.decoder_to_encoder_mapper(tgt)
        return preds


    #TODO use flex attention
    def encoder_forward(self, 
    x_contexts: torch.Tensor, 
    src_key_padding_mask : Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:

        if self.use_gradient_checkpointing and self.training:
            contextual_features = checkpoint(self.encoder, x_contexts, use_reentrant=False)
        else:
            contextual_features = self.encoder(x_contexts, src_key_padding_mask = src_key_padding_mask)

        return contextual_features

    @torch.inference_mode()
    def get_audio_representation(self, audio : torch.Tensor, padding_mask : torch.tensor):
        # Get the audio representatin of waveform x.
        self.eval()
        local_features = self.extract_audio(audio)
        local_features = self.feature_norms(local_features)
        if self.post_extraction_mapper:
            local_features = self.post_extraction_mapper(local_features)
        local_features = local_features + self.pos_encoding_encoder
        # Encoder and decoder forward
        contextual_features = self.encoder_forward(local_features, src_key_padding_mask = padding_mask)
        return contextual_features
