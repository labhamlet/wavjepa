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
    identity += "TargetProb={}_TargetLen={}_ContextLen={}_ContextRatio={}".format(
        cfg.masker.get("target_prob", 0.25),
        cfg.masker.get("target_length", 10),
        cfg.masker.get("min_context_len", 5),
        cfg.masker.get("ratio_cutoff", 0.15),
    )
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
