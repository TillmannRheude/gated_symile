#!/bin/bash

#SBATCH --job-name=SG-XNOR
#SBATCH --time=48:00:00
#SBATCH -p pgpu
#SBATCH --gres=gpu:2
#SBATCH --mem=200G

SWEEP_ID="cardiors/sigmile/he9bz874"
#SWEEP_ID="cardiors/sigmile/vnd2szab"

ENV_PATH="/sc-projects/sc-proj-ukb-cvd/environments/mml"
export HF_HOME="/sc-projects/sc-proj-ukb-cvd/projects/data/tmp_hf_cache"

AGENT_CMD="conda run -p $ENV_PATH wandb agent --count 15 $SWEEP_ID"

CUDA_VISIBLE_DEVICES=0 $AGENT_CMD &
CUDA_VISIBLE_DEVICES=1 $AGENT_CMD &

wait