#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

# Launch two smoke gates followed by six independent full-order waiters.
# Existing CopyFlatCV launchers are treated as blockers so no GPU is stolen
# during their internal wave transitions.

seed=${1:-1993}
run_id=${GAMFS_LLAMA_QUEUE_ID:-$(date +%Y%m%d_%H%M%S)}
queue_root=${GAMFS_LLAMA_QUEUE_ROOT:-${repo_root}/experiment_queue/gamfs_llama_${run_id}}
mkdir -p "${queue_root}/logs" "${queue_root}/states" "${queue_root}/cache"

if [[ -f ${queue_root}/launch_manifest.tsv ]]; then
  echo "Queue already initialized: ${queue_root}" >&2
  exit 2
fi
printf 'job\tpid\tmodel\torder\tmode\tcandidates\tblockers\tstate\tlog\n' > "${queue_root}/launch_manifest.tsv"
waiter_pids=()

launch_waiter() {
  local job=$1 model=$2 order=$3 mode=$4 candidates=$5 blockers=$6 marker=$7
  local state=${queue_root}/states/${job}.json
  local log=${queue_root}/logs/${job}.log
  local train_log=${queue_root}/logs/${job}_train.log
  nohup env \
    GAMFS_LLAMA_QUEUE_ROOT="${queue_root}" \
    GAMFS_LLAMA_QUEUE_STATE="${state}" \
    GAMFS_LLAMA_TRAIN_LOG="${train_log}" \
    GAMFS_LLAMA_SMOKE_MARKER="${marker}" \
    GAMFS_LLAMA_CANDIDATE_GPUS="${candidates}" \
    GAMFS_LLAMA_BLOCKER_PIDS="${blockers}" \
    bash scripts_llama/wait_and_run_gamfs_llama.sh \
      "${model}" "${seed}" "${order}" "${mode}" \
      > "${log}" 2>&1 &
  local pid=$!
  waiter_pids+=("${pid}")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${job}" "${pid}" "${model}" "${order}" "${mode}" "${candidates}" \
    "${blockers}" "${state}" "${log}" >> "${queue_root}/launch_manifest.tsv"
}

marker_1b=${queue_root}/smoke_1b.pass
marker_3b=${queue_root}/smoke_3b.pass

launch_waiter smoke_1b 1B 1 smoke 0,1,2 1737541 "${marker_1b}"
launch_waiter smoke_3b 3B 1 smoke 3,4,5,6 1737540 "${marker_3b}"
for order in 1 2 3; do
  launch_waiter full_1b_order${order} 1B "${order}" full 0,1,2 1737541 "${marker_1b}"
  launch_waiter full_3b_order${order} 3B "${order}" full 3,4,5,6 1737540 "${marker_3b}"
done

printf 'queue_root=%s\nseed=%s\nlaunched_at=%s\n' \
  "${queue_root}" "${seed}" "$(date -Iseconds)" > "${queue_root}/run_meta.txt"
echo "Launched GAMFisherShape Llama queue: ${queue_root}"
column -t -s $'\t' "${queue_root}/launch_manifest.tsv" || cat "${queue_root}/launch_manifest.tsv"

# Keep the manager alive. Some local execution environments reap detached
# descendants as soon as their parent command exits, even when nohup is used.
printf '{"status":"running","pid":%s,"updated_at":"%s"}\n' \
  "$$" "$(date -Iseconds)" > "${queue_root}/manager_state.json"
manager_exit=0
set +e
for waiter_pid in "${waiter_pids[@]}"; do
  wait "${waiter_pid}"
  waiter_exit=$?
  if (( waiter_exit != 0 )); then
    manager_exit=1
  fi
done
set -e
printf '{"status":"completed","exit_code":%s,"pid":%s,"updated_at":"%s"}\n' \
  "${manager_exit}" "$$" "$(date -Iseconds)" > "${queue_root}/manager_state.json"
exit "${manager_exit}"
