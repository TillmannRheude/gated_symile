#!/bin/bash

#SBATCH --job-name=CLIP-UKB
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --gres=shard:2
#SBATCH --mem=400G

SWEEP_ID="cardiors/sigmile/c8yr2vb6"


ENV_PATH="/sc-projects/sc-proj-ukb-cvd/environments/mml"
export HF_HOME="/sc-projects/sc-proj-ukb-cvd/projects/data/tmp_hf_cache"

AGENT_CMD="conda run -p $ENV_PATH wandb agent --count 10 $SWEEP_ID"

$AGENT_CMD &

wait