#!/bin/bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${script_dir}/.." && pwd)
cd "${repo_root}"

# Wait for a truly idle physical GPU, claim it with flock, run the seeded CUDA
# witness, then keep the claim while all requested orders run sequentially.
# Usage:
#   bash scripts_llama/wait_and_run_gamfs_llama.sh \
#     <1B|3B> [seed] [order:1|2|3|all] [mode:smoke|full]

model_size=${1:?model size 1B or 3B is required}
seed=${2:-1993}
task_order=${3:-all}
run_mode=${4:-full}
candidate_gpus=${GAMFS_LLAMA_CANDIDATE_GPUS:-0,1,2,3,4,5,6}
blocker_pids=${GAMFS_LLAMA_BLOCKER_PIDS:-}
threshold_mib=${GAMFS_LLAMA_GPU_FREE_THRESHOLD_MIB:-500}
poll_seconds=${GAMFS_LLAMA_GPU_POLL_SECONDS:-60}
settle_seconds=${GAMFS_LLAMA_GPU_SETTLE_SECONDS:-15}
python_executable=${GAMFS_LLAMA_PYTHON:-/data/140-1/users/liying/conda_envs/nlora_req/bin/python}
queue_root=${GAMFS_LLAMA_QUEUE_ROOT:-${repo_root}/experiment_queue/gamfs_llama_20260825}
state_file=${GAMFS_LLAMA_QUEUE_STATE:-${queue_root}/${model_size,,}_order${task_order}_${run_mode}_state.json}
smoke_marker=${GAMFS_LLAMA_SMOKE_MARKER:-${queue_root}/${model_size,,}_smoke.pass}
train_log=${GAMFS_LLAMA_TRAIN_LOG:-${queue_root}/${model_size,,}_order${task_order}_${run_mode}_train.log}
lock_root=${GAMFS_LLAMA_LOCK_ROOT:-/tmp/olora_gamfs_gpu_locks}
mkdir -p "${queue_root}" "${lock_root}"

case "${model_size,,}" in
  1b) run_all=scripts_order1_llama1B/run_all_gamfishershape_sam_olora_ewc.sh ;;
  3b) run_all=scripts_order1_llama3B/run_all_gamfishershape_sam_olora_ewc.sh ;;
  *) echo "model size must be 1B or 3B; got ${model_size}." >&2; exit 2 ;;
esac

case "${task_order}" in
  1|2|3)
    if [[ ${model_size,,} == 1b ]]; then
      run_target="scripts_order${task_order}_llama1B/EWCLoRA-GAMFisherShape-SAM-OLoRA-EWC_LLaMA3-1b.sh"
    else
      run_target="scripts_order${task_order}_llama3B/EWCLoRA-GAMFisherShape-SAM-OLoRA-EWC_LLaMA3-3b.sh"
    fi
    ;;
  all)
    run_target=${run_all}
    ;;
  *)
    echo "order must be 1, 2, 3, or all; got ${task_order}." >&2
    exit 2
    ;;
esac
if [[ ${run_mode} != smoke && ${run_mode} != full ]]; then
  echo "mode must be smoke or full; got ${run_mode}." >&2
  exit 2
fi
if [[ ${run_mode} == smoke && ${task_order} == all ]]; then
  echo "smoke mode requires a single order." >&2
  exit 2
fi

write_state() {
  local status=$1
  local gpu=${2:-null}
  local message=${3:-}
  local tmp=${state_file}.tmp.$$
  printf '{"model_size":"%s","seed":%s,"status":"%s","gpu":%s,"updated_at":"%s","message":"%s"}\n' \
    "${model_size}" "${seed}" "${status}" "${gpu}" "$(date -Iseconds)" "${message}" > "${tmp}"
  mv "${tmp}" "${state_file}"
}

gpu_memory_used() {
  local wanted=$1
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v wanted="${wanted}" '
        {gsub(/[[:space:]]/, "", $1); gsub(/[[:space:]]/, "", $2)}
        $1 == wanted {print $2; exit}
      '
}

write_state waiting null "waiting for GPU memory below ${threshold_mib} MiB"
echo ">>> ${model_size} Order ${task_order} (${run_mode}): waiting for a free GPU among ${candidate_gpus}"
IFS=',' read -ra candidates <<< "${candidate_gpus}"

while true; do
  blockers_alive=()
  if [[ -n ${blocker_pids} ]]; then
    IFS=',' read -ra blockers <<< "${blocker_pids}"
    for blocker in "${blockers[@]}"; do
      if [[ ${blocker} =~ ^[0-9]+$ ]] && kill -0 "${blocker}" 2>/dev/null; then
        blockers_alive+=("${blocker}")
      fi
    done
  fi
  if (( ${#blockers_alive[@]} > 0 )); then
    write_state waiting_for_blockers null "waiting for launcher PID(s): ${blockers_alive[*]}"
    sleep "${poll_seconds}"
    continue
  fi
  if [[ ${run_mode} == full && ! -f ${smoke_marker} ]]; then
    if [[ -f ${smoke_marker}.failed ]]; then
      write_state blocked_by_smoke null "smoke gate failed: ${smoke_marker}.failed"
      exit 4
    fi
    write_state waiting_for_smoke null "waiting for smoke marker ${smoke_marker}"
    sleep "${poll_seconds}"
    continue
  fi

  for gpu in "${candidates[@]}"; do
    if [[ ! ${gpu} =~ ^[0-9]+$ ]]; then
      echo "Invalid candidate GPU: ${gpu}" >&2
      exit 2
    fi

    exec {lock_fd}>"${lock_root}/gpu${gpu}.lock"
    if ! flock -n "${lock_fd}"; then
      exec {lock_fd}>&-
      continue
    fi

    used=$(gpu_memory_used "${gpu}" || true)
    if [[ -z ${used} || ${used} -ge ${threshold_mib} ]]; then
      exec {lock_fd}>&-
      continue
    fi

    echo ">>> Candidate GPU ${gpu} is below threshold; waiting ${settle_seconds}s to rule out a wave-transition race."
    sleep "${settle_seconds}"
    used=$(gpu_memory_used "${gpu}" || true)
    if [[ -z ${used} || ${used} -ge ${threshold_mib} ]]; then
      echo ">>> GPU ${gpu} became busy again; returning to wait loop."
      exec {lock_fd}>&-
      continue
    fi

    write_state validating "${gpu}" "running seeded CUDA kernel witness"
    CUDA_VISIBLE_DEVICES="${gpu}" "${python_executable}" -c \
      'import torch; torch.manual_seed(0); x=torch.randn(8,8,device="cuda"); y=x@x; print("WITNESS", tuple(y.shape), torch.cuda.get_device_name(), float(y.sum()))'

    write_state running "${gpu}" "running ${run_target}"
    echo ">>> Claimed GPU ${gpu} for ${model_size} Order ${task_order} (${run_mode}); starting ${run_target}"
    set +e
    if [[ ${task_order} == all ]]; then
      GAMFS_LLAMA_REQUIRE_FREE_GPU=false \
        bash "${run_target}" "${seed}" "${gpu}" 1.0 1.0 0.5 0.0 \
        2>&1 | tee "${train_log}"
    elif [[ ${run_mode} == smoke ]]; then
      GAMFS_LLAMA_REQUIRE_FREE_GPU=false \
      GAMFS_LLAMA_RUN_TAG="SMOKE-GAMFisherShape-SAM-OLoRA-EWC-${model_size}-order${task_order}-seed${seed}" \
      GAMFS_LLAMA_OUTPUT_ROOT="${queue_root}/smoke_outputs_${model_size,,}" \
      GAMFS_LLAMA_CACHE_ROOT="${queue_root}/cache/${model_size,,}_smoke" \
      GAMFS_LLAMA_MAX_STAGES=2 \
      GAMFS_LLAMA_MAX_STEPS=2 \
      GAMFS_LLAMA_MAX_TRAIN_SAMPLES=16 \
      GAMFS_LLAMA_DO_PREDICT=false \
      GAMFS_LLAMA_FISHER_MAX_BATCHES=1 \
      GAMFS_LLAMA_GLOBAL_BATCH_SIZE=1 \
      GAMFS_LLAMA_GRADIENT_ACCUMULATION_STEPS=1 \
      GAMFS_LLAMA_USE_TORCHRUN=true \
        bash "${run_target}" "${gpu}" "${seed}" 1.0 1.0 0.5 0.0 \
        2>&1 | tee "${train_log}"
    else
      GAMFS_LLAMA_REQUIRE_FREE_GPU=false \
      GAMFS_LLAMA_CACHE_ROOT="${queue_root}/cache/${model_size,,}_order${task_order}" \
      GAMFS_LLAMA_USE_TORCHRUN=true \
        bash "${run_target}" "${gpu}" "${seed}" 1.0 1.0 0.5 0.0 \
        2>&1 | tee "${train_log}"
    fi
    exit_code=${PIPESTATUS[0]}
    set -e
    if (( exit_code == 0 )) && [[ ${run_mode} == smoke ]]; then
      set +e
      "${python_executable}" -c '
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text(errors="replace")
rows = re.findall(r"GAMFisherShape-SAM-OLoRA step:.*?coeff=([0-9.eE+-]+).*?ewc_loss=([0-9.eE+-]+).*?olora_loss=([0-9.eE+-]+).*?gam_probes=([0-9.eE+-]+)", text)
ok = any(abs(float(c)-1.0) < 1e-6 and float(e) > 0.0 and float(o) > 0.0 and float(g) == 0.0 for c,e,o,g in rows)
print("SMOKE_METRIC_WITNESS", {"rows": len(rows), "joint_nonzero": ok})
raise SystemExit(0 if ok else 1)
' "${train_log}"
      metric_status=$?
      set -e
      if (( metric_status != 0 )); then
        echo "Smoke finished but did not show a stage-2 step with coeff=1, EWC>0, OLoRA>0, gam_probes=0." >&2
        exit_code=3
        printf 'failed_at=%s\nlog=%s\n' "$(date -Iseconds)" "${train_log}" > "${smoke_marker}.failed"
      else
        printf 'passed_at=%s\nlog=%s\n' "$(date -Iseconds)" "${train_log}" > "${smoke_marker}"
      fi
    fi
    if (( exit_code == 0 )); then
      write_state completed "${gpu}" "${run_mode} completed"
    else
      write_state failed "${gpu}" "${run_mode} exited with code ${exit_code}"
      if [[ ${run_mode} == smoke && ! -f ${smoke_marker}.failed ]]; then
        printf 'failed_at=%s\nlog=%s\nexit_code=%s\n' \
          "$(date -Iseconds)" "${train_log}" "${exit_code}" > "${smoke_marker}.failed"
      fi
    fi
    exec {lock_fd}>&-
    exit "${exit_code}"
  done
  sleep "${poll_seconds}"
done
