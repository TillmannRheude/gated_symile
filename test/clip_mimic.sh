#!/bin/bash

#SBATCH --job-name=CV-Symile-MIMIC
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=1-5

split_nr=${SLURM_ARRAY_TASK_ID}

# --config-name=config modelname="clip" dataset_name="symile_mimic" encoders="symile_mimic" wandb.group="None" batch_size=280 devices=1 modelname.emb_dim=1024 modelname.embedding_norm=True modelname.logit_scale_init=-0.023815122026025648 optimizer.lr=0.00773139782148669 optimizer.warmup_steps=200 optimizer.weight_decay=0

max_epochs=50
batch_size=280
modelname_emb_dim=1024
modelname_logit_scale_init=-0.023815122026025648
modelname_optimizer_lr=0.00773139782148669
modelname_optimizer_warmup_steps=200
modelname_optimizer_weight_decay=0

python3 /sc-projects/sc-proj-ukb-cvd/projects/sigmile-paper/code_cleanup/main_inference.py \
        split_nr=${split_nr} \
        modelname="clip" \
        dataset_name="symile_mimic" \
        encoders="symile_mimic" \
        wandb.group="CV-Clip-MIMIC" \
        batch_size=${batch_size} \
        max_epochs=${max_epochs} \
        modelname.emb_dim=${modelname_emb_dim} \
        modelname.logit_scale_init=${modelname_logit_scale_init} \
        optimizer.lr=${modelname_optimizer_lr} \
        optimizer.warmup_steps=${modelname_optimizer_warmup_steps} \
        optimizer.weight_decay=${modelname_optimizer_weight_decay}