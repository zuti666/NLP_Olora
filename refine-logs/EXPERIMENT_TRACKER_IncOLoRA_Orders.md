# Experiment Tracker: IncLoRA + OLoRA Multi-Order

| Run ID | Milestone | Purpose | System / Variant | Order | GPU | Priority | Status | Notes |
|--------|-----------|---------|------------------|-------|-----|----------|--------|-------|
| R100 | M0 | sanity | IncLoRA-Base (1 stage only) | Order1 | 0,1 | MUST | TODO | 仅跑 1-dbpedia，验证路径 |
| R101 | M0 | sanity | OLoRA-Base (1 stage only) | Order1 | 2,3 | MUST | TODO | 同上 |
| R102 | M1 | train | IncLoRA-SAM | Order1 | 0,1 | MUST | TODO | batch 1 |
| R103 | M1 | train | OLoRA-SAM | Order1 | 2,3 | MUST | TODO | batch 1 |
| R104 | M1 | train | IncLoRA-GAM | Order1 | 0,1 | MUST | TODO | batch 2, 已修复端口 |
| R105 | M1 | train | OLoRA-GAM | Order1 | 2,3 | MUST | TODO | batch 2 |
| R106 | M1 | train | IncLoRA-RWP | Order1 | 0,1 | MUST | TODO | batch 3 |
| R107 | M1 | train | OLoRA-RWP | Order1 | 2,3 | MUST | TODO | batch 3 |
| R108 | M1 | train | IncLoRA-Base | Order1 | 0,1 | MUST | TODO | batch 4 |
| R109 | M1 | train | OLoRA-Base | Order1 | 2,3 | MUST | TODO | batch 4 |
| R110 | M2 | eval | IncLoRA all 4 variants | Order1 | 0,1 | MUST | TODO | run_evalFlat_order1_inc_gpu01.sh |
| R111 | M2 | eval | OLoRA all 4 variants | Order1 | 2,3 | MUST | TODO | run_evalFlat_order1_olora_gpu23.sh |
| R112 | M3 | train | IncLoRA-SAM | Order2 | 0,1 | MUST | TODO | |
| R113 | M3 | train | OLoRA-SAM | Order2 | 2,3 | MUST | TODO | |
| R114 | M3 | train | IncLoRA-GAM | Order2 | 0,1 | MUST | TODO | |
| R115 | M3 | train | OLoRA-GAM | Order2 | 2,3 | MUST | TODO | |
| R116 | M3 | train | IncLoRA-RWP | Order2 | 0,1 | MUST | TODO | |
| R117 | M3 | train | OLoRA-RWP | Order2 | 2,3 | MUST | TODO | |
| R118 | M3 | train | IncLoRA-Base | Order2 | 0,1 | MUST | TODO | |
| R119 | M3 | train | OLoRA-Base | Order2 | 2,3 | MUST | TODO | |
| R120 | M3 | eval | IncLoRA all 4 variants | Order2 | 0,1 | MUST | TODO | run_evalFlat_order2_inc_gpu01.sh |
| R121 | M3 | eval | OLoRA all 4 variants | Order2 | 2,3 | MUST | TODO | run_evalFlat_order2_olora_gpu23.sh |
| R122 | M3 | train | IncLoRA-SAM | Order3 | 0,1 | MUST | TODO | |
| R123 | M3 | train | OLoRA-SAM | Order3 | 2,3 | MUST | TODO | |
| R124 | M3 | train | IncLoRA-GAM | Order3 | 0,1 | MUST | TODO | |
| R125 | M3 | train | OLoRA-GAM | Order3 | 2,3 | MUST | TODO | |
| R126 | M3 | train | IncLoRA-RWP | Order3 | 0,1 | MUST | TODO | |
| R127 | M3 | train | OLoRA-RWP | Order3 | 2,3 | MUST | TODO | |
| R128 | M3 | train | IncLoRA-Base | Order3 | 0,1 | MUST | TODO | |
| R129 | M3 | train | OLoRA-Base | Order3 | 2,3 | MUST | TODO | |
| R130 | M3 | eval | IncLoRA all 4 variants | Order3 | 0,1 | MUST | TODO | run_evalFlat_order3_inc_gpu01.sh |
| R131 | M3 | eval | OLoRA all 4 variants | Order3 | 2,3 | MUST | TODO | run_evalFlat_order3_olora_gpu23.sh |
| R132 | M4 | analysis | Order robustness heatmap | All | - | NICE | TODO | post-process only |
| R133 | M4 | analysis | Training wall-time log | All | - | NICE | TODO | read from train logs |
