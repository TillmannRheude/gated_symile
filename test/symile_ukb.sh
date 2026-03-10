#!/bin/bash

#SBATCH --job-name=SG-UKB-NoRenorm
#SBATCH --time=48:00:00
#SBATCH -p pgpu
#SBATCH --gres=gpu:2
#SBATCH --mem=300G
#SBATCH --array=0-0

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

# datamodule.combine_eids_as=intersect devices=4 datamodule.batch_size=128 batch_size=128 max_epochs=75 patience=5 modelname.negative_sampling=pair encoders.ehr.mlp.hidden_dims=[1024,2048,4096] encoders.ehr.mlp.hidden_dropouts=[0.6,0.6,0.6] encoders.nmr.mlp.hidden_dims=[1024,2048,4096] encoders.nmr.mlp.hidden_dropouts=[0.2,0.2,0.2] encoders.olink.mlp.hidden_dims=[1024,2048,4096] encoders.olink.mlp.hidden_dropouts=[0.4,0.4,0.4] modelname.emb_dim=6144 modelname.logit_scale_init=-0.06765877877672555 optimizer.lr=0.0033153970428031264 optimizer.warmup_steps=1000 optimizer.weight_decay=0.01 modelname.use_gate=True modelname.gate_d_k=6144 modelname.gate_mode=attention modelname.gate_strength_init=5.112983913246319 modelname.gate_temp=1.0882940326322543 modelname.gate_type=sigmoid modelname.neutral_type=random_trainable modelname.renormalize=False modelname.use_null=True optimizer.lr_gate_mul=1.1935684117553809

max_epochs=2
devices=4
patience=5
batch_size=128
negative_sampling="pair"
modelname_emb_dim=6144
modelname_logit_scale_init=-0.06765877877672555

# gate
modelname_use_gate=True
modelname_gate_d_k=6144
modelname_gate_mode=attention
modelname_gate_strength_init=5.112983913246319
modelname_gate_temp=1.0882940326322543
modelname_gate_type=sigmoid
modelname_neutral_type=random_trainable
modelname_renormalize=False
modelname_use_null=True

# optim
modelname_optimizer_lr=0.0033153970428031264
modelname_optimizer_warmup_steps=1000
modelname_optimizer_weight_decay=0.01
modelname_optimizer_lr_gate_mul=1.1935684117553809

# encoders 
encoders_ehr_mlp_hidden_dims="[1024,2048,4096]"
encoders_ehr_mlp_hidden_dropouts="[0.6,0.6,0.6]"
encoders_nmr_mlp_hidden_dims="[1024,2048,4096]"
encoders_nmr_mlp_hidden_dropouts="[0.2,0.2,0.2]"
encoders_olink_mlp_hidden_dims="[1024,2048,4096]"
encoders_olink_mlp_hidden_dropouts="[0.4,0.4,0.4]"

#wandb.group="Ablation-CV-${split_nr}-GatedSymile-NoNull-UKB-Proteomics-512" \

python3 /sc-projects/sc-proj-ukb-cvd/projects/sigmile-paper/code_cleanup/main_inference.py \
        split_nr=${split_nr} \
        seed=${seed} \
        datamodule.splits=/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_${split_nr}.yaml \
        modelname="symile" \
        dataset_name="ukb" \
        encoders="ukb" \
        wandb.group="Ablation-CV-${split_nr}-GatedSymile-NoRenorm-Params-UKB-Proteomics-512" \
        datamodule.combine_eids_as="intersect" \
        devices=${devices} \
        datamodule.batch_size=${batch_size} \
        batch_size=${batch_size} \
        max_epochs=${max_epochs} \
        patience=${patience} \
        modelname.negative_sampling=${negative_sampling} \
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
        modelname.use_gate=${modelname_use_gate} \
        modelname.gate_d_k=${modelname_gate_d_k} \
        modelname.gate_mode=${modelname_gate_mode} \
        modelname.gate_strength_init=${modelname_gate_strength_init} \
        modelname.gate_temp=${modelname_gate_temp} \
        modelname.gate_type=${modelname_gate_type} \
        modelname.neutral_type=${modelname_neutral_type} \
        modelname.renormalize=${modelname_renormalize} \
        modelname.use_null=${modelname_use_null} \
        optimizer.lr_gate_mul=${modelname_optimizer_lr_gate_mul}