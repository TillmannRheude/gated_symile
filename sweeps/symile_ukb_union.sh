#!/bin/bash

#SBATCH --job-name=SG-UKB-Union
#SBATCH --time=48:00:00
#SBATCH -p pgpu
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
#SBATCH --output=/sc-projects/sc-proj-ukb-cvd/projects/sigmile-paper/code_cleanup/sweeps/outputs/symilegated_ukb_union.out


SWEEP_ID="cardiors/sigmile/vui6szor"


ENV_PATH="/sc-projects/sc-proj-ukb-cvd/environments/mml"
export HF_HOME="/sc-projects/sc-proj-ukb-cvd/projects/data/tmp_hf_cache"

AGENT_CMD="srun conda run -p $ENV_PATH wandb agent --count 10 $SWEEP_ID"

$AGENT_CMD &

wait