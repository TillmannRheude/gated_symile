#!/bin/bash

#SBATCH --job-name=CV-SG-MIMIC
#SBATCH -p pgpu
#SBATCH --gres=gpu:2
#SBATCH --mem=500G
#SBATCH --time 48:00:00
#SBATCH --array=0-14
#SBATCH --exclude=s-sc-dgx01,s-sc-dgx02,s-sc-pgpu11,s-sc-pgpu12,s-sc-pgpu13,s-sc-pgpu14,s-sc-pgpu15

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

# --config-name=config modelname="symile" modelname.negative_sampling="pair" dataset_name="symile_mimic" encoders="symile_mimic" wandb.group="None" batch_size=280 devices=1 modelname.emb_dim=1024 modelname.embedding_norm=True modelname.gate_d_k=256 modelname.gate_mode=attention modelname.gate_strength_init=-0.38618370163036264 modelname.gate_temp=0.581935019249276 modelname.gate_type=sigmoid modelname.logit_scale_init=-2.315408727082122 modelname.neutral_type=random_trainable modelname.use_gate=True optimizer.lr=0.008800157072498408 optimizer.lr_gate_mul=13.78569410600449 optimizer.warmup_steps=500 optimizer.weight_decay=0

max_epochs=50
batch_size=280
negative_sampling="pair"
modelname_emb_dim=1024
modelname_embedding_norm=True
modelname_gate_d_k=256
modelname_gate_mode=attention
modelname_gate_strength_init=-0.38618370163036264
modelname_gate_temp=0.581935019249276
modelname_gate_type=sigmoid
modelname_neutral_type=random_trainable
modelname_logit_scale_init=-2.315408727082122
modelname_optimizer_lr=0.008800157072498408
modelname_optimizer_lr_gate_mul=13.78569410600449
modelname_optimizer_warmup_steps=500
modelname_optimizer_weight_decay=0
use_gate=True

CUDA_VISIBLE_DEVICES=0 python3 /sc-projects/sc-proj-ukb-cvd/projects/sigmile-paper/code_cleanup/main_inference.py \
        split_nr=${split_nr} \
        seed=${seed} \
        modelname="symile" \
        dataset_name="symile_mimic" \
        encoders="symile_mimic" \
        wandb.group="CV-${split_nr}-GatedSymile-${negative_sampling}-MIMIC" \
        batch_size=${batch_size} \
        max_epochs=${max_epochs} \
        modelname.negative_sampling=${negative_sampling} \
        modelname.emb_dim=${modelname_emb_dim} \
        modelname.embedding_norm=${modelname_embedding_norm} \
        modelname.gate_d_k=${modelname_gate_d_k} \
        modelname.gate_mode=${modelname_gate_mode} \
        modelname.gate_strength_init=${modelname_gate_strength_init} \
        modelname.gate_temp=${modelname_gate_temp} \
        modelname.gate_type=${modelname_gate_type} \
        modelname.neutral_type=${modelname_neutral_type} \
        modelname.logit_scale_init=${modelname_logit_scale_init} \
        optimizer.lr=${modelname_optimizer_lr} \
        optimizer.lr_gate_mul=${modelname_optimizer_lr_gate_mul} \
        optimizer.warmup_steps=${modelname_optimizer_warmup_steps} \
        optimizer.weight_decay=${modelname_optimizer_weight_decay} \
        modelname.use_gate=${use_gate}
