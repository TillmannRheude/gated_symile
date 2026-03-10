#!/bin/bash

#SBATCH --job-name=Clip-Mimic
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --gres=shard:1
#SBATCH --mem=300G


SWEEP_ID="cardiors/sigmile/a0bu4lst"


ENV_PATH="/sc-projects/sc-proj-ukb-cvd/environments/mml"
export HF_HOME="/sc-projects/sc-proj-ukb-cvd/projects/data/tmp_hf_cache"

AGENT_CMD="conda run -p $ENV_PATH wandb agent --count 25 $SWEEP_ID"

$AGENT_CMD &

wait