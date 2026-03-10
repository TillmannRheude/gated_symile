#!/bin/bash

#SBATCH --job-name=Symile-Imaging
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --gpus=nvidia_a100_80gb_pcie:1
#SBATCH --mem=400G

SWEEP_ID="cardiors/sigmile/4lzpy3kc"


ENV_PATH="/sc-projects/sc-proj-ukb-cvd/environments/mml"
export HF_HOME="/sc-projects/sc-proj-ukb-cvd/projects/data/tmp_hf_cache"

export CUDA_VISIBLE_DEVICES=0
AGENT_CMD="conda run -p $ENV_PATH wandb agent --count 25 $SWEEP_ID"

$AGENT_CMD &

wait