#!/bin/bash

#SBATCH --job-name=Probe-UKB
#SBATCH --time=48:00:00
#SBATCH -p pgpu
#SBATCH --gres=gpu:4
#SBATCH --mem=300G
#SBATCH --nodes=1
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=32
#SBATCH --array=0-0
# --nodelist=s-sc-dgx[01-02],s-sc-pgpu[03-08],s-sc-pgpu[11-15]

split_nr=${SLURM_ARRAY_TASK_ID}

python3 /sc-projects/sc-proj-ukb-cvd/projects/rhti10/gated_symile/linear_probe_ukb.py \
        --checkpoint="/sc-projects/sc-proj-ukb-cvd/projects/rhti10/gated_symile/checkpoints/transformer_symile_ukb.ckpt" \
        --config-name="config" \
        --metrics-out="/sc-projects/sc-proj-ukb-cvd/projects/rhti10/gated_symile/probe/ukb_probe.json" \
        --batch-size 128 \
        --override modelname.emb_dim=512 \
        --override modelname.transformer_params.nhead=2 \
        --override modelname.transformer_params.num_layers=2 \
        --override modelname.transformer_params.dropout=0 \
        --override encoders.ehr.mlp.hidden_dims="[1024, 2048, 4096]" \
        --override encoders.ehr.mlp.hidden_dropouts="[0.6, 0.6, 0.6]" \
        --override encoders.nmr.mlp.hidden_dims="[1024, 2048, 4096]" \
        --override encoders.nmr.mlp.hidden_dropouts="[0.2, 0.2, 0.2]" \
        --override encoders.olink.mlp.hidden_dims="[1024, 2048, 4096]" \
        --override encoders.olink.mlp.hidden_dropouts="[0.4, 0.4, 0.4]" 

# modelname="clip" dataset_name="ukb" encoders="ukb" wandb.group="None" batch_size=512 datamodule.batch_size=512 datamodule.combine_eids_as="intersect" devices=1 "encoders.ehr.mlp.hidden_dims=[1024, 2048, 4096]" "encoders.ehr.mlp.hidden_dropouts=[0.6, 0.6, 0.6]" "encoders.nmr.mlp.hidden_dims=[1024, 2048, 4096]" "encoders.nmr.mlp.hidden_dropouts=[0.2, 0.2, 0.2]" "encoders.olink.mlp.hidden_dims=[1024, 2048, 4096]" "encoders.olink.mlp.hidden_dropouts=[0.4, 0.4, 0.4]" modelname.emb_dim=6144 modelname.logit_scale_init=-0.010926499587005978 optimizer.lr=0.008470750568449963 optimizer.warmup_steps=50 optimizer.weight_decay=0.01
python3 /sc-projects/sc-proj-ukb-cvd/projects/rhti10/gated_symile/linear_probe_ukb.py \
        --checkpoint="/sc-projects/sc-proj-ukb-cvd/projects/rhti10/gated_symile/checkpoints/clip_ukb.ckpt" \
        --config-name="config" \
        --metrics-out="/sc-projects/sc-proj-ukb-cvd/projects/rhti10/gated_symile/probe/ukb_probe.json" \
        --batch-size 128 \
        --override modelname="clip" \
        --override modelname.emb_dim=6144 \
        --override modelname.logit_scale_init=-0.010926499587005978 \
        --override optimizer.lr=0.008470750568449963 \
        --override optimizer.warmup_steps=50 \
        --override optimizer.weight_decay=0.01 \
        --override encoders.ehr.mlp.hidden_dims="[1024, 2048, 4096]" \
        --override encoders.ehr.mlp.hidden_dropouts="[0.6, 0.6, 0.6]" \
        --override encoders.nmr.mlp.hidden_dims="[1024, 2048, 4096]" \
        --override encoders.nmr.mlp.hidden_dropouts="[0.4, 0.4, 0.4]" \
        --override encoders.olink.mlp.hidden_dims="[1024, 2048, 4096]" \
        --override encoders.olink.mlp.hidden_dropouts="[0.4, 0.4, 0.4]" \
        --override datamodule.combine_eids_as="intersect"