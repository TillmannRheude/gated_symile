#!/bin/bash

#SBATCH --job-name=CV-Clip-UKB-512
#SBATCH -p gpu
#SBATCH --gres=shard:1
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=0-4

split_nr=${SLURM_ARRAY_TASK_ID}
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_0.yaml"
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_1.yaml"
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_2.yaml"
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_3.yaml"
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_4.yaml"

# --config-name=config_nmr_retrieval modelname="clip" dataset_name="ukb" wandb.group="None" batch_size=512 datamodule.batch_size=512 datamodule.combine_eids_as="union" devices=1 "encoders.bloodbio.mlp.hidden_dims=[512, 1024, 2048]" "encoders.bloodbio.mlp.hidden_dropouts=[0.2, 0.2, 0.2]" "encoders.ehr.mlp.hidden_dims=[1024, 2048, 1024]" "encoders.ehr.mlp.hidden_dropouts=[0.6, 0.6, 0.6]" "encoders.nmr.mlp.hidden_dims=[512, 1024, 2048]" "encoders.nmr.mlp.hidden_dropouts=[0.2, 0.2, 0.2]" modelname.emb_dim=3072 modelname.logit_scale_init=-0.05122457589586027 optimizer.lr=0.004438067735433193 optimizer.warmup_steps=1800 optimizer.weight_decay=0.1

max_epochs=70
batch_size=512
modelname_emb_dim=3072
modelname_logit_scale_init=-0.05122457589586027
modelname_optimizer_lr=0.004438067735433193
modelname_optimizer_warmup_steps=1800
modelname_optimizer_weight_decay=0.1
encoders_bloodbio_mlp_hidden_dims="[512,1024,2048]"
encoders_bloodbio_mlp_hidden_dropouts="[0.2,0.2,0.2]"
encoders_ehr_mlp_hidden_dims="[1024,2048,1024]"
encoders_ehr_mlp_hidden_dropouts="[0.6,0.6,0.6]"
encoders_nmr_mlp_hidden_dims="[512,1024,2048]"
encoders_nmr_mlp_hidden_dropouts="[0.2,0.2,0.2]"

python3 /sc-projects/sc-proj-ukb-cvd/projects/sigmile-paper/code_cleanup/main_inference.py --config-name=config_nmr_retrieval \
        split_nr=${split_nr} \
        datamodule.splits=/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_${split_nr}.yaml \
        modelname="clip" \
        wandb.group="CV-CLIP-UKB-NMR-512" \
        datamodule.combine_eids_as="union" \
        datamodule.batch_size=${batch_size} \
        batch_size=${batch_size} \
        max_epochs=${max_epochs} \
        encoders.bloodbio.mlp.hidden_dims=${encoders_bloodbio_mlp_hidden_dims} \
        encoders.bloodbio.mlp.hidden_dropouts=${encoders_bloodbio_mlp_hidden_dropouts} \
        encoders.ehr.mlp.hidden_dims=${encoders_ehr_mlp_hidden_dims} \
        encoders.ehr.mlp.hidden_dropouts=${encoders_ehr_mlp_hidden_dropouts} \
        encoders.nmr.mlp.hidden_dims=${encoders_nmr_mlp_hidden_dims} \
        encoders.nmr.mlp.hidden_dropouts=${encoders_nmr_mlp_hidden_dropouts} \
        modelname.emb_dim=${modelname_emb_dim} \
        modelname.logit_scale_init=${modelname_logit_scale_init} \
        optimizer.lr=${modelname_optimizer_lr} \
        optimizer.warmup_steps=${modelname_optimizer_warmup_steps} \
        optimizer.weight_decay=${modelname_optimizer_weight_decay}