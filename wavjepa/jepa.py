import copy
import transformers 
import numpy as np 

from typing import List, Any, Optional

import torch
from torch import nn
from einops import repeat, rearrange
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl

from wavjepa.pos_embed import get_1d_sincos_pos_embed_from_grid, get_2d_sincos_pos_embed, get_binaural_pos_embed

from wavjepa.functions import trunc_normal_
from wavjepa.extractors.audio_extractor import Extractor
from wavjepa.types import ForwardReturn, TransformerLayerCFG, TransformerEncoderCFG

torch._dynamo.config.capture_dynamic_output_shape_ops = True

def collate_fn(batch : List[torch.Tensor]) -> torch.Tensor:
    return batch.flatten(start_dim = 0, end_dim = 1)


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
        size : str = "base",
        **kwargs : dict[str, Any],
    ):
        super().__init__(**kwargs)
        self.sr = resample_sr
        self.target_audio_length = self.TARGET_SECONDS * self.sr

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
            self.collate_fn = torch.compile(collate_fn)
        else:
            self.collate_fn = collate_fn


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
        elif self.in_channels == 2:
            # We use 1D sincos embeddings with channel number indicated on the embedding.
            # We assume total_patches is (time_steps * channels).
            if self.total_patches % self.in_channels != 0:
                raise ValueError(
                    f"total_patches ({self.total_patches}) must be divisible by "
                    f"in_channels ({self.in_channels}) for binaural embeddings."
                )
            
            print(f"Using Binaural Positional Embeddings for {self.in_channels} channels")
            pos_embed_data = get_binaural_pos_embed(
                embedding_dim, 
                time_steps=self.total_patches // self.in_channels
            )
        elif self.in_channels == 1:
            # IF it is plain audio, we used 1d sincos embeddings
            pos_embed_data = get_1d_sincos_pos_embed_from_grid(
                embedding_dim,
                positions,
            )
        else:
            raise Exception(f"Not supported for audio channels more than 2, you got {self.in_channels}")
        
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
        # Unpack batch (Audio is all 480,000)
        (
            audio_batch,
            ctx_masks,
            target_indices,
            ctx_and_target_masks,
        ) = batch
        
        B, C, L_full = audio_batch.shape
        # Generate all random start indices at once
        rand_starts = torch.randint(
            0, L_full - self.target_length + 1,
            (B, self.nr_samples_per_audio),
            device=self.device
        )

        # Create indices for gathering
        # Shape: (B, nr_samples, target_length)
        indices = rand_starts.unsqueeze(-1) + torch.arange(self.target_length, device=self.device)
        indices_expanded = indices.unsqueeze(2).expand(-1, -1, C, -1)

        clean_scene_expanded = audio_batch.unsqueeze(1).expand(-1, self.nr_samples_per_audio, -1, -1)

        return_clean_audios = torch.gather(clean_scene_expanded, 3, indices_expanded)

        mean = return_clean_audios.mean(dim=(-2, -1), keepdim=True)
        std = return_clean_audios.std(dim=(-2, -1), keepdim=True)
        normalized_clean_audios = (return_clean_audios - mean) / (std + 1e-5) # Add epsilon for stability

        # Cast to bfloat16 and flatten batch and samples dimensions
        flattened_clean = self.collate_fn(normalized_clean_audios.to(torch.bfloat16))

        # Shuffle the samples
        idx = torch.randperm(flattened_clean.size(0))

        return flattened_clean[idx, ...], self.collate_fn(ctx_masks), self.collate_fn(target_indices), self.collate_fn(ctx_and_target_masks)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> ForwardReturn:
        audio_input, ctx_masks, target_indices, ctx_and_target_masks = self.prepare_batch(batch)
        out = self(audio_input,ctx_masks, target_indices, ctx_and_target_masks)

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
    

    def forward(self, audio : torch.Tensor, ctx_masks, target_indices, ctx_and_target_masks) -> ForwardReturn:
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
        local_features = self.extract_audio(audio)
        local_features = self.feature_norms(local_features)
        if self.post_extraction_mapper is not None:
            local_features = self.post_extraction_mapper(local_features)
        
        local_features = local_features + self.pos_encoding_encoder
        contextual_features = self.encoder_forward(local_features, src_key_padding_mask=ctx_masks)
        # Accumulate contextual features on the batch dimensions
        contextual_features = contextual_features[~ctx_masks]
        contextual_features = self.encoder_to_decoder_mapper(contextual_features)
        
        preds = self.decoder_forward(contextual_features, ctx_masks, nr_targets = target_indices.shape[1], src_key_padding_mask=ctx_and_target_masks)
        
        # Compute the training targets using the teacher.
        x_targets = local_features.detach()
        targets = self._forward_teacher(x_targets)

        loss = self.masked_loss(preds, targets, target_indices)
        
        return ForwardReturn(
            local_features=local_features,
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