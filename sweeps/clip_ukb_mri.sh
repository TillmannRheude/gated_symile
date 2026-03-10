#!/bin/bash

#SBATCH --job-name=CLIP-Imaging
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=400G


SWEEP_ID="cardiors/sigmile/rsqvytzy"


ENV_PATH="/sc-projects/sc-proj-ukb-cvd/environments/mml"
export HF_HOME="/sc-projects/sc-proj-ukb-cvd/projects/data/tmp_hf_cache"

AGENT_CMD="conda run -p $ENV_PATH wandb agent --count 25 $SWEEP_ID"

CUDA_VISIBLE_DEVICES=0 $AGENT_CMD &

wait