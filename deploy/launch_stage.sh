#!/usr/bin/env bash
set -eo pipefail
# (deliberately no -u: this script expands possibly-empty EXTRA_ARGS/
# ENTRY_CMD arrays, and `set -u` + empty-array expansion is only safe on
# bash >=4.4 -- required vars are still validated explicitly below via
# "${VAR:?msg}", which errors regardless of -u.)

# Usage: deploy/launch_stage.sh STAGE [CONFIG_DIR] [install_root]
#
# Launches every *.env config in CONFIG_DIR concurrently for the given stage
# (default CONFIG_DIR: deploy/configs_stage1_T06 for stage 1,
# deploy/configs_stage<STAGE> otherwise). Each run gets its own CPU range
# (taskset) and its own runs/<RUN_NAME> directory. Runs resume from their
# newest checkpoint, so re-running this script after an interruption
# continues every run in place. Training uses the CPU only.
#
# Each *.env file is a shell snippet sourced by this script. It may set:
#   RUN_NAME              run directory name under runs/ (required)
#   SEED                  PPO seed (required)
#   TERMINATION_VARIANT   T06 or T02 (required)
#   PREV_POLICY_PATH      previous stage's final_policy_state_dict.pt, relative
#                         to runs/ or absolute (required for stages 2-4)
#   TAU_MAX_OVERRIDE      device torque cap override in N*m (control conditions)
#   ZERO_EXO_REWARD       1 removes the exo reward terms (reward-neutral control)
#   CORES_PER_RUN         CPU count for this run (default: all CPUs split
#                         evenly across the configs)
#   NUM_ENVS              parallel environments (default 32, Table S4)
#   TOTAL_TIMESTEPS       absolute step target (default: the stage length in
#                         stage_config.TOTAL_TIMESTEPS_BY_STAGE)
#   MODEL_PATH            model XML override (default: train.py's choice, the
#                         bare-human model at stage 1 and the device model
#                         otherwise)
#   ENTRY_CMD             array: entry point (default: smat3/train.py)
#   EXTRA_ARGS            array: extra train.py flags (e.g. smoke-test sizes)
#
# install_root defaults to the repository root (the parent of deploy/).

BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:?usage: launch_stage.sh STAGE [CONFIG_DIR] [install_root]}"
case "${STAGE}" in 1|2|3|4) ;; *) echo "[launch_stage] FATAL: STAGE must be 1-4" >&2; exit 1;; esac
ROOT="${3:-${BUNDLE_ROOT}}"

if [ "${ROOT}" != "${BUNDLE_ROOT}" ]; then
  echo "[launch_stage] copying bundle contents into ${ROOT}"
  mkdir -p "${ROOT}"
  cp -r "${BUNDLE_ROOT}/smat_r3" "${BUNDLE_ROOT}/deploy" "${ROOT}/"
fi

SMAT3_DIR="${ROOT}/smat_r3/smat3"
if [ "${STAGE}" = 1 ]; then DEFAULT_CFG="${ROOT}/deploy/configs_stage1_T06"; else DEFAULT_CFG="${ROOT}/deploy/configs_stage${STAGE}"; fi
CONFIGS_DIR="${2:-${DEFAULT_CFG}}"
RUNS_DIR="${ROOT}/runs"
PYTHON_BIN="${ROOT}/venv-smat3/bin/python"

[ -x "${PYTHON_BIN}" ] || { echo "[launch_stage] FATAL: ${PYTHON_BIN} not found -- run setup_smat3.sh first" >&2; exit 1; }

TOTAL_CORES="$(nproc)"
# Throughput settings (2026-10-27, D-PERF-01): pin each env worker to one CPU of its run's taskset range; workers are
# single-threaded MuJoCo, so their BLAS/OpenMP pools are 1 thread; the PPO process keeps 4 threads for its update.
export SMAT_PIN_WORKERS="${SMAT_PIN_WORKERS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
mkdir -p "${RUNS_DIR}"

shopt -s nullglob
config_files=("${CONFIGS_DIR}"/*.env)
shopt -u nullglob

n_configs="${#config_files[@]}"
if [ "${n_configs}" -eq 0 ]; then
  echo "[launch_stage] FATAL: no *.env files found in ${CONFIGS_DIR}" >&2
  exit 1
fi

echo "[launch_stage] host has ${TOTAL_CORES} cores; launching ${n_configs} config(s)"

base_cores=$(( TOTAL_CORES / n_configs ))
extra_cores=$(( TOTAL_CORES % n_configs ))
if [ "${base_cores}" -lt 1 ]; then
  echo "[launch_stage] WARNING: ${TOTAL_CORES} cores / ${n_configs} configs < 1 core each;" \
       "pinning every run to 1 core (heavy oversubscription expected)." >&2
  base_cores=1
  extra_cores=0
fi

core_cursor=0
idx=0
pids=()
names=()

for cfg in "${config_files[@]}"; do
  # Reset per-config variables so one file cannot leak its overrides into
  # the next iteration.
  CORES_PER_RUN="${base_cores}"
  if [ "${idx}" -lt "${extra_cores}" ]; then
    CORES_PER_RUN=$(( base_cores + 1 ))
  fi
  NUM_ENVS=32
  TOTAL_TIMESTEPS=""
  MODEL_PATH=""
  PREV_POLICY_PATH=""
  TAU_MAX_OVERRIDE=""
  ZERO_EXO_REWARD=""
  ENTRY_CMD=("${SMAT3_DIR}/train.py")
  EXTRA_ARGS=()
  RUN_NAME=""
  SEED=""
  TERMINATION_VARIANT=""

  # shellcheck disable=SC1090
  source "${cfg}"

  : "${RUN_NAME:?${cfg} must set RUN_NAME}"
  : "${SEED:?${cfg} must set SEED}"
  : "${TERMINATION_VARIANT:?${cfg} must set TERMINATION_VARIANT}"

  core_lo="${core_cursor}"
  core_hi=$(( core_lo + CORES_PER_RUN - 1 ))
  if [ "${core_hi}" -ge "${TOTAL_CORES}" ]; then
    echo "[launch_stage] WARNING: ${RUN_NAME} wants cores ${core_lo}-${core_hi}," \
         "but only ${TOTAL_CORES} exist -- wrapping to start at core 0 instead." \
         "This run will share physical cores with another run; expect reduced" \
         "throughput versus the Table S4 reference rate." >&2
    core_lo=0
    core_hi=$(( CORES_PER_RUN - 1 ))
  fi
  if [ "${NUM_ENVS}" -gt "${CORES_PER_RUN}" ]; then
    echo "[launch_stage] NOTE: ${RUN_NAME} has NUM_ENVS=${NUM_ENVS} > CORES_PER_RUN=${CORES_PER_RUN} --" \
         "SubprocVecEnv will oversubscribe these cores. Lower NUM_ENVS in this" \
         "run's .env, or give it more cores, if that matters for this run." >&2
  fi

  run_dir="${RUNS_DIR}/${RUN_NAME}"
  mkdir -p "${run_dir}"
  log_file="${run_dir}/launch.log"

  model_path_args=()
  if [ -n "${MODEL_PATH}" ]; then
    model_path_args=(--model-path "${MODEL_PATH}")
  fi
  stage_args=()
  if [ -n "${TOTAL_TIMESTEPS}" ]; then stage_args+=(--total-timesteps "${TOTAL_TIMESTEPS}"); fi
  if [ -n "${TAU_MAX_OVERRIDE}" ]; then stage_args+=(--tau-max-override "${TAU_MAX_OVERRIDE}"); fi
  if [ "${ZERO_EXO_REWARD}" = 1 ]; then stage_args+=(--zero-exo-reward); fi
  if [ "${STAGE}" != 1 ]; then
    # A fresh stage-2/3/4 start needs the previous stage's final state dict; a resume (checkpoints present)
    # does not read it, but the file must still exist so a run can never start from the wrong parent.
    case "${PREV_POLICY_PATH}" in /*) ;; "") ;; *) PREV_POLICY_PATH="${RUNS_DIR}/${PREV_POLICY_PATH}";; esac
    if [ -z "${PREV_POLICY_PATH}" ] || [ ! -s "${PREV_POLICY_PATH}" ]; then
      echo "[launch_stage] FATAL: ${RUN_NAME}: PREV_POLICY_PATH missing or empty: '${PREV_POLICY_PATH}'" >&2
      exit 1
    fi
    stage_args+=(--prev-policy-path "${PREV_POLICY_PATH}")
  fi

  echo "[launch_stage] ${RUN_NAME}: cores ${core_lo}-${core_hi}, seed=${SEED}," \
       "termination=${TERMINATION_VARIANT}, num_envs=${NUM_ENVS}, stage=${STAGE}, stage_args=${stage_args[*]}," \
       "run_dir=${run_dir}"

  {
    echo "[launch_stage] $(date -u +%Y-%m-%dT%H:%M:%SZ) starting ${RUN_NAME}"
    echo "[launch_stage] cmd: PYTHONPATH=${SMAT3_DIR} taskset -c ${core_lo}-${core_hi} ${PYTHON_BIN} ${ENTRY_CMD[*]} --stage ${STAGE} --run-dir ${run_dir} --seed ${SEED} --termination-variant ${TERMINATION_VARIANT} --num-envs ${NUM_ENVS} ${stage_args[*]} ${model_path_args[*]} ${EXTRA_ARGS[*]}"
  } >> "${log_file}"

  PYTHONPATH="${SMAT3_DIR}${PYTHONPATH:+:${PYTHONPATH}}" \
    taskset -c "${core_lo}-${core_hi}" "${PYTHON_BIN}" "${ENTRY_CMD[@]}" \
      --stage "${STAGE}" \
      --run-dir "${run_dir}" \
      --seed "${SEED}" \
      --termination-variant "${TERMINATION_VARIANT}" \
      --num-envs "${NUM_ENVS}" \
      "${stage_args[@]}" \
      "${model_path_args[@]}" \
      "${EXTRA_ARGS[@]}" \
      >> "${log_file}" 2>&1 &
  pids+=("$!")
  names+=("${RUN_NAME}")

  core_cursor=$(( core_hi + 1 ))
  idx=$(( idx + 1 ))
done

echo "[launch_stage] launched ${#pids[@]} run(s); pids: ${pids[*]}"
echo "[launch_stage] tail -f ${RUNS_DIR}/<run_name>/launch.log and .../checkpoints/ to monitor."
echo "[launch_stage] waiting for all runs to exit..."

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "[launch_stage] ${names[$i]} (pid ${pids[$i]}) exited 0"
  else
    rc=$?
    echo "[launch_stage] ${names[$i]} (pid ${pids[$i]}) exited ${rc}" >&2
    status=1
  fi
done
exit "${status}"
