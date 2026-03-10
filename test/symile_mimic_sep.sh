#!/bin/bash

#SBATCH --job-name=CV-Symile-MIMIC
#SBATCH -p pgpu
#SBATCH --gres=gpu:4
#SBATCH --mem=200G
#SBATCH --time 48:00:00

# batch_size=280 modelname.emb_dim=2048 optimizer.lr=0.0036511733001158434 optimizer.warmup_steps=500 optimizer.weight_decay=0

max_epochs=50
batch_size=280
modelname_emb_dim=2048
modelname_optimizer_lr=0.0036511733001158434
modelname_optimizer_warmup_steps=500
modelname_optimizer_weight_decay=0

for split_nr in 1 2 3 4
do
    device=$((split_nr - 1))
    CUDA_VISIBLE_DEVICES=${device} python3 /sc-projects/sc-proj-ukb-cvd/projects/sigmile-paper/code_cleanup/main_inference.py \
            split_nr=${split_nr} \
            modelname="symile" \
            dataset_name="symile_mimic" \
            encoders="symile_mimic" \
            wandb.group="CV-Symile-MIMIC" \
            batch_size=${batch_size} \
            max_epochs=${max_epochs} \
            modelname.emb_dim=${modelname_emb_dim} \
            optimizer.lr=${modelname_optimizer_lr} \
            optimizer.warmup_steps=${modelname_optimizer_warmup_steps} \
            optimizer.weight_decay=${modelname_optimizer_weight_decay} &
done
wait