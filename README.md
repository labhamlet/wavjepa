# Signal-based Joint-Embenning Predictive Architecture (S-JEPA)

## Installation
```bash
poetry install
```

## Usage
### Data pre-processing
```
python scripts/preprocess_moabb_dataset.py config/eeg_jepa_Lee2019_preprocessing.yaml
```

### SSL pre-training 
saves the runs to W&B
```
python sjepa/eeg_jepa.py fit -c scripts/config/eeg_jepa_Lee2019_spat-mask.yaml --trainer.accelerator gpu --trainer.devices='[0]' --data.length=1.2   --model.mask_maker_kwargs.chs_radius_blocks=0.4
```

### Fine-tuning and evaluation
Fine-tune the pre-trained model and evaluate it:
```
python sjepa/evaluation_eeg_jepa.py --config=scripts/config/eeg_jepa_Lee2019_eval.yaml --device=cuda:0 --run_id=jxrubd7p
```

Or, run the evaluation script without pre-training :
```
python sjepa/evaluation_eeg_jepa.py --config=scripts/config/eeg_jepa_Lee2019_eval.yaml --config_model=scripts/config/eeg_jepa_Lee2019_spat-mask.yaml --device=cuda:0
```

### Export runs history and summary to CSV
```
python sjepa/export_runs.py --config=scripts/config/eeg_jepa_Lee2019_export.yaml
```

### Plot results
Use the notebooks:
- Training curves: `notebooks/viz_history.ipynb`
- Downstream ranking and scores: `notebooks/viz_summary.ipynb`

## Cite
```bibtex
@inproceedings{sjepa2024,
  title = {S-JEPA: towards seamless cross-dataset transfer through dynamic spatial attention},
  author={Pierre Guetschel and Thomas Moreau and Michael Tangermann},
  booktitle={9th Graz Brain-Computer Interface Conference},
  address = {Graz, Austria},
  url = {https://arxiv.org/abs/2403.11772},
  year = {2024},
  month = {September},
  grant = {DCC}
}
```