def get_identity_from_cfg(cfg):
    identity = "Data={}_".format(
        cfg.data.get("name", None),
    )
    identity += "Extractor={}_InSeconds={}_".format(
        cfg.extractor.name,
        cfg.data.process_seconds,
    )
    identity += "BatchSize={}_NrSamples={}_NrGPUs={}_LR={}_".format(
        cfg.trainer.get("batch_size"),
        cfg.data.get("samples_per_audio"),
        cfg.trainer.get("num_gpus"),
        cfg.optimizer.get("lr"),
    )
    masker_name = str(cfg.masker.get("name", "time-inverse"))
    if masker_name == "time-inverse":
        # configs/masker/AudioSet.yaml uses ``context_mask_prob`` / ``context_mask_length``;
        # older configs used ``context_prob`` / ``context_len``. Accept both.
        identity += "TargetProb={}_TargetLen={}_ContextProb={}_ContextLen={}_MinContextBlock={}_ContextRatio={}_".format(
            cfg.masker.get("target_prob", 0.25),
            cfg.masker.get("target_length", 10),
            cfg.masker.get("context_mask_prob", cfg.masker.get("context_prob", 0.65)),
            cfg.masker.get("context_mask_length", cfg.masker.get("context_len", 10)),
            cfg.masker.get("min_context_len", 1),
            cfg.masker.get("ratio_cutoff", 0.1),
        )
    else:
        # data2vec 2.0 block masking (configs/masker/D2v2Block.yaml): its own parameters instead of the paper masker's,
        # so the run directory can never collide with a time-inverse run of the same objective / batch.
        identity += "Masker={}_B={}_R={}_A={}_Inv={}_".format(
            masker_name.replace("-", ""),
            cfg.masker.get("mask_length", 5),
            cfg.masker.get("mask_prob", 0.5),
            cfg.masker.get("mask_prob_adjust", 0.05),
            bool(cfg.masker.get("inverse_mask", False)),
        )
    identity += "Packed={}".format(
        cfg.trainer.get("use_packing", True),
    )
    # Revision ablations (E1): appended only when they differ from the paper setting, so the
    # paper / revision run directories keep their names.
    objective = cfg.trainer.get("objective", "jepa")
    if objective != "jepa":
        identity += "_Objective={}".format(objective)
    if objective == "d2v2":
        # the decoder decides what the arm tests (and, for the conv one, how far it can reach), so it is in the name
        if str(cfg.trainer.get("d2v2_decoder_type", "transformer")) == "transformer":
            identity += "_D2v2Dec=tf{}".format(cfg.trainer.get("transformer_decoder_layers", 12))
        else:
            identity += "_D2v2Dec={}x{}".format(
                cfg.trainer.get("d2v2_decoder_layers", 20), cfg.trainer.get("d2v2_decoder_kernel", 7)
            )
        # masks per clip (data2vec 2.0's clone_batch); the existing D runs used 4 and keep their names
        masks_per_clip = int(cfg.trainer.get("d2v2_masks_per_clip", 4))
        if masks_per_clip != 4:
            identity += "_D2v2M={}".format(masks_per_clip)
    top_k = cfg.trainer.get("average_top_k_layers", 8)
    if int(top_k) != 8:
        identity += "_TopK={}".format(top_k)
    # E6 ablations: predictor depth (trainer.predictor_layers) and number of target blocks (masker.target_masks_per_context)
    pred_layers = cfg.trainer.get("predictor_layers", 12)
    if int(pred_layers) != 12:
        identity += "_PredDepth={}".format(pred_layers)
    n_targets = cfg.masker.get("target_masks_per_context", 4)
    if int(n_targets) != 4:
        identity += "_NrTargets={}".format(n_targets)
    steps = cfg.trainer.get("steps", 375000)
    if int(steps) != 375000:
        identity += "_Steps={}".format(steps)
    return identity


def get_identity_from_cfg_denoise(cfg):
    identity = "Data={}_".format(
        cfg.data.get("name", None),
    )
    identity += "Extractor={}_InSeconds={}_".format(
        cfg.extractor.name,
        cfg.data.process_seconds,
    )
    identity += "BatchSize={}_NrSamples={}_NrGPUs={}_LR={}_".format(
        cfg.trainer.get("batch_size"),
        cfg.data.get("samples_per_audio"),
        cfg.trainer.get("num_gpus"),
        cfg.optimizer.get("lr"),
    )
    identity += "Alpha={}".format(
        cfg.trainer.get("alpha", 0.0)
    )
    return identity
