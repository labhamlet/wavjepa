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

from wavjepa.pos_embed import get_1d_sincos_pos_embed_from_grid
from wavjepa.functions import trunc_normal_
from wavjepa.extractors.audio_extractor import Extractor
from wavjepa.types import ForwardReturn, TransformerLayerCFG, TransformerEncoderCFG
from data_modules.dataset_functions import pad_or_truncate_batch

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
        average_top_k_layers: int = 8,
        resample_sr : int = 16000,
        process_audio_seconds: float = 2.01,
        nr_samples_per_audio = 16,
        use_gradient_checkpointing: bool = False,
        compile_modules : bool = False,
        size : str = "base",
        **kwargs : dict[str, Any],
    ):
        super().__init__(**kwargs)
        self.sr = resample_sr 
        self.nr_samples_per_audio = nr_samples_per_audio
        self.ema_end_step = ema_anneal_end_step
        self.target_length = int(resample_sr * process_audio_seconds)
        self.total_patches = feature_extractor.total_patches(self.target_length)
        self.use_compiled_forward = compile_modules
        self.use_gradient_checkpointing = use_gradient_checkpointing
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
        self.post_extraction_mapper : Optional[nn.Module] = nn.Linear(feature_extractor.embedding_dim, self.encoder_embedding_dim)
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
        
        self.collate_fn = collate_fn
        self.pad_or_truncate_batch = pad_or_truncate_batch


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
        pos_embed_data = get_1d_sincos_pos_embed_from_grid(
            embedding_dim,
            positions,
        )
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
        for student, teacher in zip(self.encoder.parameters(), 
                                    self.teacher_encoder.parameters()):
            teacher.data.mul_(r).add_((1 - r) * student.detach().data)

    def _compile_operations(self):
            """
            Compile inner modules to avoid PyTorch Lightning hook graph breaks.
            """
            _compile_opts = dict(
                fullgraph=True,
                mode="max-autotune",
            )
            self.encoder = torch.compile(self.encoder, **_compile_opts)
            self.decoder = torch.compile(self.decoder, **_compile_opts)
            self.teacher_encoder = torch.compile(self.teacher_encoder, **_compile_opts)

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
    def _forward_teacher(self, x: torch.Tensor) -> torch.Tensor:
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

        (
            audio,
            ctx_masks,
            target_indices,
            ctx_and_target_masks
        ) = batch


        if audio.ndim != 3:
            audio = audio.unsqueeze(1)

        B, C, L_full = audio.shape

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
        normalized_clean_audios = index_select_and_normalize(audio, indices)
        # Cast to bfloat16 and flatten batch and samples dimensions
        flattened_clean = self.collate_fn(normalized_clean_audios.to(torch.bfloat16))

        # Shuffle the samples
        idx = torch.randperm(flattened_clean.size(0))
        return flattened_clean[idx, ...], self.collate_fn(ctx_masks), self.collate_fn(target_indices), self.collate_fn(ctx_and_target_masks)


    def training_step(self, batch: torch.Tensor, batch_idx: int) -> ForwardReturn:
        audio, ctx_masks, target_indices, ctx_and_target_masks = batch
        out = self(audio, ctx_masks, target_indices, ctx_and_target_masks)

        log_data = {
            "train/loss": out['loss'],
            "train/loss_clean": out["loss_clean"],
            "train/loss_noisy": out["loss_consistency"],
            "ema" : self._get_ema_decay(),
            "lambda" : self._get_current_lambda(),
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

        pred_reshaped = pred.view(B, N, -1, D)           # (B, N, T, D)
        target = target.unsqueeze(1).expand(B, N, -1, D)         # (B, N, T, D) — view, no copy

        loss = self.loss_fn(pred_reshaped, target)        # (B, N, T, D)
        loss_per_timestep = loss.mean(dim=-1)             # (B, N, T)
        masked = loss_per_timestep * target_indices       # (B, N, T)
        return masked.sum() / (target_indices.sum() + 1e-8)
    
    def _student_head_forward(self, 
                                local_features, 
                                ctx_masks, 
                                ctx_tgt_masks, 
                                nr_targets):
        """
        Helper method to allow torch.compile compatible checkpointing.
        """
        local_features = local_features.contiguous()
        ctx_masks = ctx_masks.contiguous()
        if ctx_tgt_masks is not None:
            ctx_tgt_masks = ctx_tgt_masks.contiguous()

        contextual_features = self.encoder_forward(
            local_features, 
            src_key_padding_mask=ctx_masks
        )
        
        contextual_features = self.encoder_to_decoder_mapper(contextual_features)
        
        return self.decoder_forward(
            contextual_features, 
            ctx_masks, 
            nr_targets=nr_targets, 
            src_key_padding_mask=ctx_tgt_masks
        )
    
    def _extract_audio_embeddings(self, audio):
        feats = self.extract_audio(audio)
        feats = self.feature_norms(feats)
        feats = self.post_extraction_mapper(feats)
        return feats 
    
    def forward(self, audio: torch.Tensor, ctx_masks, target_indices, ctx_and_target_masks) -> ForwardReturn:
        with torch.no_grad():
            audio_embeddings_teacher = self._extract_audio_embeddings(audio) + self.pos_encoding_encoder
            targets = self._forward_teacher(audio_embeddings_teacher)
            del audio_embeddings_teacher 

        nr_targets = target_indices.shape[1]
        audio_embeddings = self._extract_audio_embeddings(audio) + self.pos_encoding_encoder
        

        if self.use_gradient_checkpointing:
            preds = checkpoint(
                self._student_head_forward,
                audio_embeddings, 
                ctx_masks, 
                ctx_and_target_masks, 
                nr_targets,
                use_reentrant=True
            )
        else:
            preds = self._student_head_forward(
                audio_embeddings, 
                ctx_masks, 
                ctx_and_target_masks, 
                nr_targets, 
            )

        loss = self.masked_loss(preds, targets, target_indices)
        
        return ForwardReturn(
            loss=loss,
        )

    def decoder_forward(self, contextual_features, ctx_mask, nr_targets, src_key_padding_mask=None):
        B, seq_len = ctx_mask.shape
        E = self.decoder_embedding_dim

        # Start from all mask tokens
        tgt = self.mask_token.expand(B, seq_len, E)                # (B, T, E)

        # Blend context in via masking — no boolean indexing
        ctx_mask_f = (~ctx_mask).unsqueeze(-1).to(contextual_features.dtype)  # (B, T, 1)
        tgt = tgt * ctx_mask.unsqueeze(-1).to(tgt.dtype) + contextual_features * ctx_mask_f

        # Repeat for each target block
        tgt = repeat(tgt, 'B S E -> (B N) S E', N=nr_targets)
        src_key_padding_mask = rearrange(src_key_padding_mask, 'B N S -> (B N) S')

        tgt = tgt + self.pos_encoding_decoder
        tgt = self.decoder(tgt, src_key_padding_mask=src_key_padding_mask)
        return self.decoder_to_encoder_mapper(tgt)


    #TODO use flex attention
    def encoder_forward(self, 
    x_contexts: torch.Tensor, 
    src_key_padding_mask : Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        contextual_features = self.encoder(x_contexts, 
                                           src_key_padding_mask = src_key_padding_mask)
        return contextual_features

    @torch.inference_mode()
    def get_audio_representation(self, audio : torch.Tensor, padding_mask : torch.tensor):
        self.eval()
        local_features = self.extract_audio(audio)
        local_features = self.feature_norms(local_features)
        if self.post_extraction_mapper:
            local_features = self.post_extraction_mapper(local_features)
        local_features = local_features + self.pos_encoding_encoder
        # Encoder and decoder forward
        contextual_features = self.encoder_forward(local_features, src_key_padding_mask = padding_mask)
        return contextual_features