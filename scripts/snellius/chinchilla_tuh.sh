#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --gpus=1
#SBATCH --partition=gpu_mig
#SBATCH --time=00:01:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=pierre.guetschel@gmail.com

# doc: https://slurm.schedmd.com/srun.html

# model sizes:
ms_name="i-2"
ms_args="--model.transformer_kwargs.num_encoder_layers=2 --model.transformer_kwargs.num_decoder_layers=2 --model.transformer_kwargs.nhead=2 --model.transformer_kwargs.dim_feedforward=64  --model.average_top_k_layers=2 --data.batch_size=512"
#ms_name="i-4"
#ms_args="--model.transformer_kwargs.num_encoder_layers=4 --model.transformer_kwargs.num_decoder_layers=4 --model.transformer_kwargs.nhead=4 --model.transformer_kwargs.dim_feedforward=128 --model.average_top_k_layers=3 --data.batch_size=512"
#ms_name="i-8"
#ms_args="--model.transformer_kwargs.num_encoder_layers=8 --model.transformer_kwargs.num_decoder_layers=8 --model.transformer_kwargs.nhead=8 --model.transformer_kwargs.dim_feedforward=256 --model.average_top_k_layers=4 --data.batch_size=512"
#ms_name="i-8_j-2"
#ms_args="--model.transformer_kwargs.num_encoder_layers=8 --model.transformer_kwargs.num_decoder_layers=2 --model.transformer_kwargs.nhead=8 --model.transformer_kwargs.dim_feedforward=256 --model.average_top_k_layers=4 --data.batch_size=512"
#ms_name="i-8_j-4"
#ms_args="--model.transformer_kwargs.num_encoder_layers=8 --model.transformer_kwargs.num_decoder_layers=4 --model.transformer_kwargs.nhead=8 --model.transformer_kwargs.dim_feedforward=256 --model.average_top_k_layers=4 --data.batch_size=512"
#ms_name="i-8_j-8"
#ms_args="--model.transformer_kwargs.num_encoder_layers=8 --model.transformer_kwargs.num_decoder_layers=8 --model.transformer_kwargs.nhead=8 --model.transformer_kwargs.dim_feedforward=256 --model.average_top_k_layers=4 --data.batch_size=512"

# num steps:
#ns_name="50k"
#ns_args="--trainer.max_steps=50000"
#ns_name="100k"
#ns_args="--trainer.max_steps=100000"
ns_name="200k"
ns_args="--trainer.max_steps=200000"

name="${ms_name}_${ns_name}"
args="${ms_args} ${ns_args} --trainer.logger.init_args.name=$name --trainer.logger.dict_kwargs.group=chinchilla"

# make output directory
out_file=$HOME/SLURM/${SLURM_JOBID}-${name}.out

# Execute program located in $HOME passing as argument the $SLURM_ARRAY_ID
#args=`python $HOME/signal-jepa/scripts/product_args.py "[('--a', [1, 2, 3]), ('--b', [4, 5, 6])]" $SLURM_ARRAY_TASK_ID`

# env
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/pguetschel/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/pguetschel/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/pguetschel/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/pguetschel/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate signal-jepa

srun --output="${out_file}" --job-name="$name" python $HOME/signal-jepa/sjepa/eeg_jepa.py --config=$HOME/signal-jepa/scripts/config/eeg_jepa_tuh_spat-mask.yaml $args