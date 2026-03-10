#!/bin/bash

#SBATCH --job-name=CV-Symile-UKB-NMR-512
#SBATCH -p gpu
#SBATCH --gpus=shard:2
#SBATCH --mem=200G
#SBATCH --time 48:00:00
#SBATCH --array=0-0

split_nr=${SLURM_ARRAY_TASK_ID}
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_0.yaml"
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_1.yaml"
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_2.yaml"
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_3.yaml"
# "/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_4.yaml"

# --config-name=config dataset_name="ukb" encoders="ukb" max_epochs=50 modelname="symile" modelname.negative_sampling="pair" wandb.group="None" batch_size=512 datamodule.batch_size=512 datamodule.combine_eids_as="intersect" devices=1 "encoders.ehr.mlp.hidden_dims=[1024, 2048, 4096]" "encoders.ehr.mlp.hidden_dropouts=[0.6, 0.6, 0.6]" "encoders.nmr.mlp.hidden_dims=[1024, 2048, 4096]" "encoders.nmr.mlp.hidden_dropouts=[0.2, 0.2, 0.2]" "encoders.olink.mlp.hidden_dims=[1024, 2048, 4096]" "encoders.olink.mlp.hidden_dropouts=[0.4, 0.4, 0.4]" modelname.emb_dim=6144 modelname.logit_scale_init=-0.21558322938721552 modelname.use_gate=False optimizer.lr=0.00525796298529248 optimizer.warmup_steps=1000 optimizer.weight_decay=0.001

max_epochs=50
batch_size=512
modelname_emb_dim=6144
modelname_logit_scale_init=-0.21558322938721552
modelname_optimizer_lr=0.00525796298529248
modelname_optimizer_warmup_steps=1000
modelname_optimizer_weight_decay=0.001
encoders_ehr_mlp_hidden_dims="[1024,2048,4096]"
encoders_ehr_mlp_hidden_dropouts="[0.6,0.6,0.6]"
encoders_olink_mlp_hidden_dims="[1024,2048,4096]"
encoders_olink_mlp_hidden_dropouts="[0.4,0.4,0.4]"
encoders_nmr_mlp_hidden_dims="[1024,2048,4096]"
encoders_nmr_mlp_hidden_dropouts="[0.2,0.2,0.2]"

ckpt_name="symile_ukb_${split_nr}_proteomics_probe"

python3 /sc-projects/sc-proj-ukb-cvd/projects/sigmile-paper/code_cleanup/main_inference.py --config-name=config \
        split_nr=${split_nr} \
        datamodule.splits=/sc-projects/sc-proj-ukb-cvd/projects/data/ukb/cvsplits/splits_${split_nr}.yaml \
        modelname="symile" \
        wandb.group="CV-Symile-UKB-NMR-512" \
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