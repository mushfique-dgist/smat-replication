#!/usr/bin/env bash
set -uo pipefail
# NOT set -e: each smoke job's pass/fail is checked and reported explicitly
# below so one failure does not abort the whole matrix silently.

# Smoke-test matrix at toy size: Stage 1 (T06 and T02), a Stage 2 -> 3 -> 4
# chain seeded from the Stage-1 T06 run, and both Stage-4 control conditions,
# each for one or two PPO updates (4 envs, n_steps 64, batch 64,
# --total-timesteps = updates x 256 so that every job reaches its target and
# writes final_policy_state_dict.pt). A job passes when train.py exits 0,
# writes final_policy_state_dict.pt and logs no NaN.
# Results: smat_r3/smoke_runs/SUMMARY.txt.

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${REPO}/smat_r3"
SMAT3="${ROOT}/smat3"
PY="${REPO}/venv-smat3/bin/python"
RUNS="${ROOT}/smoke_runs"

mkdir -p "${RUNS}"
rm -rf "${RUNS:?}"/* 2>/dev/null || true

NUM_ENVS=4
N_STEPS=64
BATCH_SIZE=64
BUFFER_SIZE=$((NUM_ENVS * N_STEPS))

SUMMARY="${RUNS}/SUMMARY.txt"
: > "${SUMMARY}"

# Appends to SUMMARY only -- never prints to stdout. run_job() below is
# called as `VAR=$(run_job ...)` by its caller to capture the ONE path it
# deliberately echoes on its last line; anything log() put on stdout would
# corrupt that capture into a multi-line string.
log() { echo "$@" >> "${SUMMARY}"; }

# run_job <name> <updates> <extra train.py args...>
# Returns 0/1 via $? and prints PASS/FAIL to SUMMARY. On success, prints the
# run's final_policy_state_dict.pt path to stdout (ONLY that path, on its own
# final line) so callers can capture it with $(...).
run_job() {
  local name="$1" updates="$2"; shift 2
  local run_dir="${RUNS}/${name}"
  local total=$((updates * BUFFER_SIZE))
  mkdir -p "${run_dir}"
  local t0 t1
  t0=$(date +%s)
  ( cd "${SMAT3}" && \
    nice -n 10 timeout 540 "${PY}" train.py \
        --run-dir "${run_dir}" --num-envs "${NUM_ENVS}" --n-steps "${N_STEPS}" \
        --batch-size "${BATCH_SIZE}" --total-timesteps "${total}" --verbose 1 \
        "$@" \
  ) > "${run_dir}/stdout.log" 2> "${run_dir}/stderr.log"
  local rc=$?
  t1=$(date +%s)

  local nan_hits=0
  if [ -f "${run_dir}/progress.jsonl" ]; then
    nan_hits=$(grep -ic 'nan' "${run_dir}/progress.jsonl" || true)
  fi
  local final_ok="no"
  [ -f "${run_dir}/final_policy_state_dict.pt" ] && final_ok="yes"

  if [ "${rc}" -eq 0 ] && [ "${nan_hits}" -eq 0 ] && [ "${final_ok}" = "yes" ]; then
    log "[PASS] ${name}: $((t1 - t0))s, rc=${rc}, final_state_dict=${final_ok}, nan_hits=${nan_hits}"
  else
    log "[FAIL] ${name}: $((t1 - t0))s, rc=${rc}, final_state_dict=${final_ok}, nan_hits=${nan_hits} -- see ${run_dir}/stderr.log"
    tail -n 20 "${run_dir}/stderr.log" | sed 's/^/    /' | tee -a "${SUMMARY}" > /dev/null
  fi
  echo "${run_dir}/final_policy_state_dict.pt"
}

log "=== SMAT_R3 smoke matrix: $(date -u +%FT%TZ) ==="
log "buffer_size=${BUFFER_SIZE} (num_envs=${NUM_ENVS} x n_steps=${N_STEPS}), batch_size=${BATCH_SIZE}"

# --- Stage 1, T06 and T02: 4 envs, 2 PPO updates ------------------------------
S1_T06_FINAL=$(run_job "stage1_T06" 2 --stage 1 --termination-variant T06 --seed 1)
run_job "stage1_T02" 2 --stage 1 --termination-variant T02 --seed 1 > /dev/null

# --- Stage 2 -> 3 -> 4 chain, seeded from the stage1_T06 run -----------------
if [ -f "${S1_T06_FINAL}" ]; then
  S2_FINAL=$(run_job "stage2_chain" 1 --stage 2 --termination-variant T06 --seed 1 --prev-policy-path "${S1_T06_FINAL}")
  if [ -f "${S2_FINAL}" ]; then
    S3_FINAL=$(run_job "stage3_chain" 1 --stage 3 --termination-variant T06 --seed 1 --prev-policy-path "${S2_FINAL}")
    if [ -f "${S3_FINAL}" ]; then
      S4_FINAL=$(run_job "stage4_chain" 1 --stage 4 --termination-variant T06 --seed 1 --prev-policy-path "${S3_FINAL}")
      echo "${S4_FINAL}" > "${RUNS}/_stage4_chain_final_path.txt"

      # --- both Stage-4 control conditions, 1 update each, branching off Stage 3
      run_job "stage4_s4z_on" 1 --stage 4 --termination-variant T06 --seed 1 \
        --prev-policy-path "${S3_FINAL}" --tau-max-override 0.0 > /dev/null
      run_job "stage4_s4z_neutral" 1 --stage 4 --termination-variant T06 --seed 1 \
        --prev-policy-path "${S3_FINAL}" --tau-max-override 0.0 --zero-exo-reward > /dev/null
    else
      log "[SKIP] stage4_chain + both S4z controls: stage3_chain produced no final_policy_state_dict.pt"
    fi
  else
    log "[SKIP] stage3_chain onward: stage2_chain produced no final_policy_state_dict.pt"
  fi
else
  log "[SKIP] full chain + controls: stage1_T06 produced no final_policy_state_dict.pt"
fi

log "=== smoke matrix done: $(date -u +%FT%TZ) ==="
cat "${SUMMARY}"
