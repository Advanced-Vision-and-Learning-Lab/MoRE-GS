#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "usage: $0 PREPARED_SCENE MODEL_DIR SCENE_NAME [FORECAST_SEED]" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source_dir="$(realpath "$1")"
model_dir="$(realpath -m "$2")"
scene_name="$3"
forecast_seed="${4:-42}"
python_bin="${PYTHON_BIN:-python}"
backbone="${GAUSSIAN_PREDICTION_ROOT:-${repo_root}/external/gaussian_prediction}"
manifest="${source_dir}/MOREGS_SPLIT_MANIFEST.json"

[[ -s "${manifest}" ]] || {
    echo "missing ${manifest}; run scripts/prepare_hypernerf.py first" >&2
    exit 3
}
[[ -f "${backbone}/train.py" ]] || {
    echo "GaussianPrediction is not installed; run scripts/setup_gaussianprediction.sh" >&2
    exit 3
}

gpu_lock="${MOREGS_GPU_LOCK:-${repo_root}/.locks/gpu0.lock}"
mkdir -p "${model_dir}" "$(dirname "${gpu_lock}")"
exec 9>"${gpu_lock}"
flock -n 9 || { echo "MoRE-GS GPU 0 lock is already held" >&2; exit 4; }

exec > >(tee -a "${model_dir}/moregs_run.log") 2>&1

export CUDA_VISIBLE_DEVICES=0
export PYTHONHASHSEED="${forecast_seed}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export MPLCONFIGDIR="${model_dir}/.matplotlib"
mkdir -p "${MPLCONFIGDIR}"

date --iso-8601=seconds
nvidia-smi --query-gpu=index,name,memory.total,memory.free,utilization.gpu,compute_mode --format=csv,noheader

environment_prefix="$(dirname "$(dirname "$(realpath "${python_bin}")")")"
export CUDA_HOME="${CUDA_HOME:-${environment_prefix}}"
export PATH="${CUDA_HOME}/bin:${PATH}"
torch_library="$(${python_bin} -c 'import os, torch; print(os.path.dirname(torch.__file__) + "/lib")')"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${torch_library}:${LD_LIBRARY_PATH:-}"

read -r observed future fit_frames calibration_time < <(
    "${python_bin}" - "${manifest}" "${source_dir}" <<'PY'
import json, math, sys
from pathlib import Path
m = json.load(open(sys.argv[1]))
p = Path(sys.argv[2])
metadata = json.load(open(p / "metadata.json"))
dataset = json.load(open(p / "dataset.json"))
observed, future = int(m["observed"]), int(m["future"])
fit = int(math.floor(0.75 * observed))
names = dataset["train_ids"]
if len(names) != observed or not (10 <= fit < observed):
    raise RuntimeError((len(names), observed, future, fit))
maximum = float(dataset["time_normalization_max_warp"])
boundary = (float(metadata[names[fit - 1]]["warp_id"]) +
            float(metadata[names[fit]]["warp_id"])) / 2.0
print(observed, future, fit, boundary / maximum)
PY
)

echo "scene=${scene_name} observed=${observed} future=${future} fit=${fit_frames} seed=${forecast_seed}"

if [[ ! -s "${model_dir}/chkpnt70000.pth" ]]; then
    (
        cd "${backbone}"
        "${python_bin}" train.py -s "${source_dir}" -m "${model_dir}" \
            --max_points 150 --adaptive_points_num 100 --time_freq 8 \
            --densify_from_iter 5000 --densify_until_iter 15000 \
            --opacity_reset_interval 3000000 --max_time 0.9 --norm_rotation \
            --position_lr_max_steps 40000 --data_device cpu --eval \
            --use_time_decay --nearest_num 6 --adaptive_interval 1000 \
            --feature_amplify 5 --jointly_iteration 1000 --seed 42 \
            --iterations 70000 --checkpoint_iterations 70000 \
            --save_iterations 70000 --test_iterations 70001
    )
fi

full_gcn_name="moregs_full_gcn_seed${forecast_seed}"
fit_gcn_name="moregs_fit_gcn_seed${forecast_seed}"
full_gcn="${model_dir}/${full_gcn_name}/ckpt.pth"
fit_gcn="${model_dir}/${fit_gcn_name}/ckpt.pth"

train_gcn() {
    local maximum_time="$1" name="$2" checkpoint="${model_dir}/$2/ckpt.pth"
    [[ -s "${checkpoint}" ]] && return
    (
        cd "${backbone}"
        "${python_bin}" train_GCN.py -m "${model_dir}" \
            --max_points 150 --adaptive_points_num 100 --ckpt_iteration 70000 \
            --time_freq 8 --max_time "${maximum_time}" --nearest_num 6 \
            --feature_amplify 5 --num_stage 6 --noise_step 100 --noise_init 0 \
            --input_size 10 --epoch 2001 --linear_size 128 --batch_size 32 \
            --exp_name "${name}" --norm_rotation --seed "${forecast_seed}" --quiet \
            --strict_fit_no_future_views
    )
}

train_gcn 0.9 "${full_gcn_name}"
train_gcn "${calibration_time}" "${fit_gcn_name}"

coefficient="${model_dir}/moregs_coefficient_seed${forecast_seed}.pt"
structured="${model_dir}/moregs_structured_seed${forecast_seed}.pt"
base_checkpoint="${model_dir}/moregs_base_g8_seed${forecast_seed}.pt"
base_summary="${model_dir}/moregs_base_g8_summary.json"
final_checkpoint="${model_dir}/moregs_g16_seed${forecast_seed}.pt"
final_summary="${model_dir}/moregs_g16_summary.json"

common=(
    -m "${model_dir}" --ckpt_iteration 70000 --max_points 150
    --adaptive_points_num 100 --time_freq 8 --max_time 0.9
    --nearest_num 6 --feature_amplify 5 --norm_rotation
    --jointly_iteration 1000 --sequence_length 10 --seed "${forecast_seed}"
    --scene_name "HyperNeRF ${scene_name} strict 90/10"
    --expected_observed_frames "${observed}"
    --expected_future_frames "${future}"
)

if [[ ! -s "${coefficient}" ]]; then
    "${python_bin}" -m moregs.coefficient_expert "${common[@]}" \
        --mode train --basis_variant random --max_train_horizon "${future}" \
        --epochs 2001 --learning_rate 0.001 --weight_decay 0.0001 \
        --hidden 32 --heads 8 --feedforward 64 --dropout 0.2 \
        --checkpoint "${coefficient}" --experiment_name moregs_coefficient
fi

if [[ ! -s "${structured}" ]]; then
    "${python_bin}" -m moregs.structured_expert "${common[@]}" \
        --mode train --routing structured --groups 8 --epochs 2001 \
        --learning_rate 0.001 --checkpoint "${structured}" \
        --experiment_name moregs_structured
fi

if [[ ! -s "${base_summary}" ]]; then
    "${python_bin}" -m moregs.train "${common[@]}" \
        --mode run --max_train_horizon "${future}" --epochs 2001 \
        --learning_rate 0.001 --structured_learning_rate 0.001 \
        --weight_decay 0.0001 --hidden 32 --heads 8 --feedforward 64 \
        --dropout 0.2 --groups 8 --group_temperature 0.18 \
        --fit_fraction 0.75 --calibration_bins 4 \
        --minimum_relative_gain 0.005 --rendered_region_weight 1.0 \
        --router_steps 200 --router_learning_rate 0.05 \
        --router_smoothness_weight 0.002 --router_residual_weight 0.0001 \
        --router_entropy_weight 0.0001 --gcn_linear_size 128 \
        --gcn_num_stage 6 --nonfinite_policy persistence \
        --calibration_gcn_checkpoint "${fit_gcn}" \
        --gcn_checkpoint "${full_gcn}" --random_checkpoint "${coefficient}" \
        --structured_checkpoint "${structured}" --checkpoint "${base_checkpoint}" \
        --summary "${base_summary}" --experiment_name moregs_base_g8
fi

if [[ ! -s "${final_summary}" ]]; then
    "${python_bin}" -m moregs.ablation "${common[@]}" \
        --mode run --max_train_horizon "${future}" --epochs 2001 \
        --learning_rate 0.001 --structured_learning_rate 0.001 \
        --weight_decay 0.0001 --hidden 32 --heads 8 --feedforward 64 \
        --dropout 0.2 --groups 8 --group_temperature 0.18 \
        --fit_fraction 0.75 --calibration_bins 4 \
        --minimum_relative_gain 0.005 --rendered_region_weight 1.0 \
        --router_steps 200 --router_learning_rate 0.05 \
        --router_smoothness_weight 0.002 --router_residual_weight 0.0001 \
        --router_entropy_weight 0.0001 --gcn_linear_size 128 \
        --gcn_num_stage 6 --nonfinite_policy persistence \
        --calibration_gcn_checkpoint "${fit_gcn}" \
        --gcn_checkpoint "${full_gcn}" --random_checkpoint "${coefficient}" \
        --structured_checkpoint "${structured}" \
        --base_checkpoint "${base_checkpoint}" --base_summary "${base_summary}" \
        --evaluation_variants motion_groups_16 --checkpoint "${final_checkpoint}" \
        --summary "${final_summary}" --experiment_name moregs_g16
fi

"${python_bin}" - "${final_summary}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
assert d["cold_reload"] is True
assert d["protocol"]["future_frames_used_for_training_grouping_routing_or_selection"] == 0
print(json.dumps(d["results"]["motion_groups_16"]["metrics"], indent=2))
PY
