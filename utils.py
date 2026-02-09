def get_identity_from_cfg(cfg):
    identity = f"SR={cfg.data.sr}_"
    identity += "BatchSize={}_NrSamples={}_NrGPUs={}_ModelSize={}_LR={}_".format(
        cfg.trainer.get("batch_size"),
        cfg.data.get("samples_per_audio"),
        cfg.trainer.get("num_gpus"),
        cfg.trainer.get("size"),
        cfg.optimizer.get("lr"),
    )
    identity += "Masking={}_TargetProb={}_TargetLen={}_ContextLen={}_TopK={}".format(
        cfg.masker.name,
        cfg.masker.get("target_prob", 0.25),
        cfg.masker.get("target_length", 0),
        cfg.masker.get("context_mask_length", 0),
        cfg.trainer.get("average_top_k_layers", 1),
    )
    return identity
