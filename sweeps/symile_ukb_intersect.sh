#!/bin/bash

#SBATCH --job-name=Symile-UKB-Ablation-Ones
#SBATCH --time=48:00:00
#SBATCH -p pgpu
#SBATCH --nodelist=s-sc-dgx[01-02],s-sc-pgpu[03-08],s-sc-pgpu[11-15]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --mem=300G

SWEEP_ID="cardiors/sigmile/l4ie8szd"

ENV_PATH="/sc-projects/sc-proj-ukb-cvd/environments/mml"
export HF_HOME="/sc-projects/sc-proj-ukb-cvd/projects/data/tmp_hf_cache"

AGENT_CMD="conda run -p $ENV_PATH wandb agent --count 25 $SWEEP_ID"

$AGENT_CMD &

wait