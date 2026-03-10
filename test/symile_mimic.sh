#!/bin/bash

#SBATCH --job-name=CV-S-MIMIC
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=300G
#SBATCH --time 48:00:00
#SBATCH --array=0-14

# Map SLURM array id -> (cv split, seed)
# Example: 5 CV splits x 3 seeds = 15 jobs, array=0-14
SEEDS=(420 1337 2024)
N_SEEDS=${#SEEDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
split_idx=$(( TASK_ID / N_SEEDS ))  # 0..4
seed_idx=$(( TASK_ID % N_SEEDS ))  # 0..2

# MIMIC splits appear to be numbered split1..split5 in this setup
SPLIT_BASE=1
split_nr=$(( SPLIT_BASE + split_idx ))
seed=${SEEDS[$seed_idx]}

# --config-name=config modelname="symile" modelname.negative_sampling="pair" dataset_name="symile_mimic" encoders="symile_mimic" wandb.group="None" batch_size=280 devices=1 modelname.emb_dim=1024 modelname.embedding_norm=True modelname.logit_scale_init=-0.992742892459706 modelname.use_gate=False optimizer.lr=0.008623049906813396 optimizer.warmup_steps=500 optimizer.weight_decay=0

max_epochs=50
batch_size=280
negative_sampling="pair"
modelname_emb_dim=1024
modelname_embedding_norm=True
modelname_logit_scale_init=-0.992742892459706
modelname_optimizer_lr=0.008623049906813396
modelname_optimizer_warmup_steps=500
modelname_optimizer_weight_decay=0
use_gate=False

python3 /sc-projects/sc-proj-ukb-cvd/projects/sigmile-paper/code_cleanup/main_inference.py \
        split_nr=${split_nr} \
        seed=${seed} \
        modelname="symile" \
        dataset_name="symile_mimic" \
        encoders="symile_mimic" \
        wandb.group="CV-${split_nr}-Symile-${negative_sampling}-MIMIC" \
        batch_size=${batch_size} \
        max_epochs=${max_epochs} \
        modelname.negative_sampling=${negative_sampling} \
        modelname.emb_dim=${modelname_emb_dim} \
        modelname.embedding_norm=${modelname_embedding_norm} \
        modelname.logit_scale_init=${modelname_logit_scale_init} \
        optimizer.lr=${modelname_optimizer_lr} \
        optimizer.warmup_steps=${modelname_optimizer_warmup_steps} \
        optimizer.weight_decay=${modelname_optimizer_weight_decay} \
        modelname.use_gate=${use_gate}
