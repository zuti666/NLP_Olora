#!/bin/bash
set -euo pipefail

seed=${1:-1993}

# 训练第一批：IncLoRA-SAM (GPU 0,1,2,3) + OLoRA-SAM (GPU 4,5,6,7) 并行
bash scripts_order3/IncLoRA-SAM_E1_Le3.sh 0,1,2,3 "${seed}" &
bash scripts_order3/OLoRA_SAM.sh 4,5,6,7 "${seed}" &
wait

# 训练第二批：IncLoRA-Base (GPU 0,1,2,3) + OLoRA-Base (GPU 4,5,6,7) 并行
bash scripts_order3/IncLoRA_E1_Le3.sh 0,1,2,3 "${seed}" &
bash scripts_order3/OLoRA.sh 4,5,6,7 "${seed}" &
wait

# 同时评估
bash scripts_order3/Olora_Eval3/run_evalFlat_order3_inc_sam_sgd_gpu01.sh "${seed}" &
bash scripts_order3/Olora_Eval3/run_evalFlat_order3_olora_sam_sgd_gpu23.sh "${seed}" &
wait
