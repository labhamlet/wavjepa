def get_identity_from_cfg(cfg):

    mixing_weight = 0.0 
    mix = cfg.data.get("mixing_weights", []) 
    if len(mix) == 2:
        mixing_weight = mix[1]
    identity = f"SR={cfg.data.sr}_"
    identity += "LibriRatio={}_BatchSize={}_NrSamples={}_NrGPUs={}_ModelSize={}_LR={}_".format(
        mixing_weight,
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
