#!/bin/bash

#SBATCH --job-name=SG-Mimic
#SBATCH --time=48:00:00
#SBATCH -p pgpu
#SBATCH --gres=gpu:2
#SBATCH --mem=500G
#SBATCH --exclude=s-sc-dgx01,s-sc-dgx02,s-sc-pgpu11,s-sc-pgpu12,s-sc-pgpu13,s-sc-pgpu14,s-sc-pgpu15


SWEEP_ID="cardiors/sigmile/esed73lj"


ENV_PATH="/sc-projects/sc-proj-ukb-cvd/environments/mml"
export HF_HOME="/sc-projects/sc-proj-ukb-cvd/projects/data/tmp_hf_cache"

AGENT_CMD="conda run -p $ENV_PATH wandb agent --count 15 $SWEEP_ID"

CUDA_VISIBLE_DEVICES=0 $AGENT_CMD &

wait