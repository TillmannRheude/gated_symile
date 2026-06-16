#!/bin/bash

#SBATCH --job-name=Probe-UKB
#SBATCH --time=48:00:00
#SBATCH -p gpu
#SBATCH --gres=shard:1
#SBATCH --mem=300G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --array=0-7
# --exclude=s-sc-pgpu[03]
# --dependency=afterok:8978865_0:8978866_0:8978862_0:8978864_0
# --nodelist=s-sc-dgx[01-02],s-sc-pgpu[03-08],s-sc-pgpu[11-15]

set -euo pipefail

PROJECT_DIR="/sc-projects/sc-proj-ukb-cvd/projects/rhti10/gated_symile"
SPLITS="${SPLITS:-0 1 2 3 4}"
EMBEDDINGS_DIR="${EMBEDDINGS_DIR:-}"
RUN_CONFIG="${RUN_CONFIG:-${PROJECT_DIR}/probe/ukb_probe.yaml}"
EXPANDED_RUN_CONFIG="$(mktemp "${PROJECT_DIR}/probe/ukb_probe_expanded_XXXXXX.yaml")"
MODEL_INDEX="${MODEL_INDEX:-${SLURM_ARRAY_TASK_ID:-0}}"

cleanup() {
    rm -f "${EXPANDED_RUN_CONFIG}"
}
trap cleanup EXIT

python3 - "${RUN_CONFIG}" "${EXPANDED_RUN_CONFIG}" "${MODEL_INDEX}" ${SPLITS} <<'PY'
import copy
import sys
from omegaconf import OmegaConf

run_config_path = sys.argv[1]
expanded_path = sys.argv[2]
model_index = int(sys.argv[3])
splits = [int(x) for x in sys.argv[4:]]

cfg = OmegaConf.to_container(OmegaConf.load(run_config_path), resolve=True)
runs = cfg.get("runs")
if not isinstance(runs, list) or len(runs) == 0:
    raise ValueError(f"Run config must define a non-empty 'runs' list: {run_config_path}")

if model_index < 0 or model_index >= len(runs):
    raise ValueError(
        f"MODEL_INDEX={model_index} out of range for {len(runs)} runs. "
        f"Set SLURM array bounds or MODEL_INDEX accordingly."
    )

base_run = copy.deepcopy(runs[model_index])
model_name = str(base_run.get("name", f"run_{model_index}"))

expanded_runs = []
for split_nr in splits:
    new_run = copy.deepcopy(base_run)
    new_run["split_nr"] = split_nr
    expanded_runs.append(new_run)

cfg["runs"] = expanded_runs
cfg.pop("split_nr", None)
OmegaConf.save(config=OmegaConf.create(cfg), f=expanded_path)
print(f"Wrote expanded run config to: {expanded_path}")
print(f"Total expanded runs: {len(expanded_runs)}")
print(f"Selected model: {model_name}")
PY

MODEL_NAME="$(python3 - "${RUN_CONFIG}" "${MODEL_INDEX}" <<'PY'
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True)
runs = cfg.get("runs", [])
idx = int(sys.argv[2])
if idx < 0 or idx >= len(runs):
    raise ValueError(f"MODEL_INDEX={idx} out of range for {len(runs)} runs.")
print(str(runs[idx].get("name", f"run_{idx}")))
PY
)"

METRICS_OUT="${METRICS_OUT:-${PROJECT_DIR}/probe/ukb_probe_${MODEL_NAME}_aggregated.json}"

cmd=(
    env CUDA_VISIBLE_DEVICES=0 python3 "${PROJECT_DIR}/linear_probe_ukb.py"
    --config-name "config"
    --metrics-out "${METRICS_OUT}"
    --batch-size 512
    --num-workers 32
    --probe-type sgd_logreg
    --alpha-values 1e-5 1e-4 1e-3
    --class-weight balanced
    --probe-max-iter 1000
    --probe-tol 1e-3
    --probe-n-jobs 32
)

if [[ -n "${RUN_CONFIG}" ]]; then
    cmd+=(--run-config "${EXPANDED_RUN_CONFIG}")
else
    echo "ERROR: RUN_CONFIG must be set for split aggregation mode." >&2
    exit 1
fi

if [[ -n "${EMBEDDINGS_DIR}" ]]; then
    cmd+=(--save-embeddings-dir "${EMBEDDINGS_DIR}/${MODEL_NAME}")
fi

"${cmd[@]}"
