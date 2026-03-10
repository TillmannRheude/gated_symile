#!/bin/bash

#SBATCH --job-name=CV-Clip-UKB-512
#SBATCH -p gpu
#SBATCH --gres=shard:1
#SBATCH --mem=300G
#SBATCH --time 48:00:00
#SBATCH --array=0-14

# Map SLURM array id -> (cv split, seed)
# Example: 5 CV splits x 3 seeds = 15 jobs, array=0-14
SEEDS=(420 1337 2024)
N_SEEDS=${#SEEDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
split_nr=$(( TASK_ID / N_SEEDS ))
seed_idx=$(( TASK_ID % N_SEEDS ))
seed=${SEEDS[$seed_idx]}

# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_0.yaml"
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_1.yaml"
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_2.yaml"
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_3.yaml"
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_4.yaml"

# --config-name=config modelname="clip" dataset_name="ukb" encoders="ukb" wandb.group="None" batch_size=512 datamodule.batch_size=512 datamodule.combine_eids_as="intersect" devices=1 "encoders.ehr.mlp.hidden_dims=[1024, 2048, 4096]" "encoders.ehr.mlp.hidden_dropouts=[0.6, 0.6, 0.6]" "encoders.nmr.mlp.hidden_dims=[1024, 2048, 4096]" "encoders.nmr.mlp.hidden_dropouts=[0.2, 0.2, 0.2]" "encoders.olink.mlp.hidden_dims=[1024, 2048, 4096]" "encoders.olink.mlp.hidden_dropouts=[0.4, 0.4, 0.4]" modelname.emb_dim=6144 modelname.logit_scale_init=-0.010926499587005978 optimizer.lr=0.008470750568449963 optimizer.warmup_steps=50 optimizer.weight_decay=0.01

max_epochs=75
devices=1
patience=5
batch_size=512
modelname_emb_dim=6144
modelname_logit_scale_init=-0.010926499587005978

# gate 
modelname_use_gate=False

# optim
modelname_optimizer_lr=0.008470750568449963
modelname_optimizer_warmup_steps=50
modelname_optimizer_weight_decay=0.01

# encoders 
encoders_ehr_mlp_hidden_dims="[1024,2048,4096]"
encoders_ehr_mlp_hidden_dropouts="[0.6,0.6,0.6]"
encoders_nmr_mlp_hidden_dims="[1024,2048,4096]"
encoders_nmr_mlp_hidden_dropouts="[0.2,0.2,0.2]"
encoders_olink_mlp_hidden_dims="[1024,2048,4096]"
encoders_olink_mlp_hidden_dropouts="[0.4,0.4,0.4]"

python3 /sc-projects/sc-proj-ukb-cvd/projects/sigmile-paper/code_cleanup/main_inference.py \
        split_nr=${split_nr} \
        seed=${seed} \
        datamodule.splits=/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_${split_nr}.yaml \
        modelname="clip" \
        dataset_name="ukb" \
        encoders="ukb" \
        wandb.group="CV-${split_nr}-Clip-UKB-Proteomics-512" \
        datamodule.combine_eids_as="intersect" \
        devices=${devices} \
        datamodule.batch_size=${batch_size} \
        batch_size=${batch_size} \
        max_epochs=${max_epochs} \
        patience=${patience} \
        encoders.ehr.mlp.hidden_dims=${encoders_ehr_mlp_hidden_dims} \
        encoders.ehr.mlp.hidden_dropouts=${encoders_ehr_mlp_hidden_dropouts} \
        encoders.nmr.mlp.hidden_dims=${encoders_nmr_mlp_hidden_dims} \
        encoders.nmr.mlp.hidden_dropouts=${encoders_nmr_mlp_hidden_dropouts} \
        encoders.olink.mlp.hidden_dims=${encoders_olink_mlp_hidden_dims} \
        encoders.olink.mlp.hidden_dropouts=${encoders_olink_mlp_hidden_dropouts} \
        modelname.emb_dim=${modelname_emb_dim} \
        modelname.logit_scale_init=${modelname_logit_scale_init} \
        optimizer.lr=${modelname_optimizer_lr} \
        optimizer.warmup_steps=${modelname_optimizer_warmup_steps} \
        optimizer.weight_decay=${modelname_optimizer_weight_decay} \
        modelname.use_gate=${modelname_use_gate}
