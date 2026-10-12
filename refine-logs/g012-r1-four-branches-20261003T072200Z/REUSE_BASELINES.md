# Four baseline reuse audit

Audited at 2026-10-03T07:28:47.819560+00:00. Historical evidence passed; reuse remains conditional on the new CPU radius-invariance gate. `passed=true` describes this historical audit, while `reuse_ready=false` until that gate is recorded.

The original Stage1 core contains 16 conditions: CE/S+/S− natural; O_P/D/C/R natural; A_P/D/C/R natural; A_P/D/C/R matched; CE matched. It does not contain O matched. Reuse these four baselines and run the 12 g2-dependent conditions at r=1.

| Historical baseline | AAA | FAA | LA | BWT | Forgetting |
|---|---:|---:|---:|---:|---:|
| S1-01_CE_natural_s2 | 60.336565 | 40.140000 | 51.005000 | -12.072222 | 12.240000 |
| S1-02_Splus_natural_s2 | 63.233744 | 44.578000 | 53.703000 | -10.138889 | 10.268889 |
| S1-03_Sminus_natural_s2 | 51.090460 | 25.368000 | 45.184000 | -22.017778 | 22.017778 |
| S1-16_CE_matched_s2 | 59.954966 | 39.565000 | 51.690000 | -13.472222 | 13.516667 |

Each baseline passed the original read-only `validate_job(job)` over 10 tasks, 500 epochs and 2550 actual SGD updates. Completion receipts bind the current summary, trace and task matrix hashes; summaries bind the original input config and trainer hash. All 190 source hashes and all 78 historical RUN_PINS files match. Every recorded step passed trainable/frozen/gradient/buffer/mode/control-cache/RNG restoration. CE matched completed 2550 matched updates with maximum rounded FP32 relative norm error 6.521230334638729e-7.

Radius-independent update reasoning: g0 and g1± are evaluated before the outer point. CE/S+/S− choose g0/g1+/g1−; CE matched uses g0 and the same candidate-state g1+ reference. After g2, the transaction restores parameters, buffers, gradients, modes and caches; final RNG is R1 from g0. The pending update contains only base, g1+, U and nominal LR. Thus the outer radius does not enter these four submitted updates or their matching target.

The required gate is `gates/R1_CPU_ACCEPTANCE.json` with `status="passed"` and `baseline_radius_invariance_passed=true`. The new CPU gate must compare r=.25 and r=1 for all four baselines across multiple actual FP32 updates, with exact parameters, buffers, controls/caches and final RNG; matched target/actual delta must agree. Outer-point/g2 diagnostics must demonstrably differ so an ignored radius cannot produce a false pass. New resolved settings must remain identical except radius and campaign paths. If the gate fails, disable reuse and train four new baselines.

No independent completed S reference is required. Matching is to the positive-g1 update at each candidate’s current state and batch. Do not substitute independent-S trace matching from the separate LR-controls campaign.

Historical artifacts retain r=.25. Their final-update baseline metrics may be reused conditionally; their g2/outer-point diagnostics are not r=1 evidence. No new completed receipts or new r1 full-training traces have been created. Fresh full execution of all 16 would provide 16 r1 traces at the cost of four extra full runs; the old four baseline runtimes total 8.335 GPU hours (historical cost only).

Native r1 g2/A results are excluded from same-protocol reuse: they use g0→g1+→g2→g3 plus clean CE, global loader/RNG, mechanism eval/stats on and eval workers8. Stage1 uses g0→g1+→g1−→g2, per-epoch isolated data RNG, R0 replay, observational evaluation, mechanism eval/stats off and eval workers4.

AAA = mean of the ten seen-task stage means; FAA = final mean of ten task accuracies; LA = diagonal mean; BWT and forgetting use old tasks0–8. Results are single seed2 observations.

Exact evidence paths and SHA256 values:

## S1-01_CE_natural_s2
Branch=P; direction=g0; mode=natural; output=/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-01_CE_natural_s2.
- config: [S1-01_CE_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/configs/S1-01_CE_natural_s2.json) — `63176f6629ed1bae053c958924a4f83eaf5e4ff4a9372c3c8135b5503335aa9c`
- effective_config: [effective_config.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-01_CE_natural_s2/effective_config.json) — `9aba4846ae57e2178d02ee1f551a365a027093f01119d6688f7f2e56f1c5c357`
- completed: [completed.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/runtime/jobs/S1-01_CE_natural_s2/completed.json) — `bd226158a5e3f54f5cc2a802e4f6104d02b1f4c9de53f3bd075e1b23ad9e26f0`
- summary: [final_summary.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-01_CE_natural_s2/final_summary.json) — `acf5c01b72e83b451d659565bddf56f9e672cdb3bd963462c9e46a51fd4b2164`
- trace: [step_trace.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-01_CE_natural_s2/step_trace.jsonl) — `8e057aed1e589cd08ee6c6fe160fa18e46a871371000c00aaa139794c95ef0ba`
- matrix: [task_matrix.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-01_CE_natural_s2/task_matrix.json) — `ec93d1b3f011d59321770535950d3ae4e4e74c00ca2037dd1dded2e5c25287a6`

## S1-02_Splus_natural_s2
Branch=P; direction=g1_plus; mode=natural; output=/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-02_Splus_natural_s2.
- config: [S1-02_Splus_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/configs/S1-02_Splus_natural_s2.json) — `002901917009ca10c213bee4c62f0c13e91a7f7c02a4697dfb28f4b9edafe916`
- effective_config: [effective_config.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-02_Splus_natural_s2/effective_config.json) — `c66d2de70bf45115d975e7b7ec8a0655af8fd1591430a6e70df674a448b23710`
- completed: [completed.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/runtime/jobs/S1-02_Splus_natural_s2/completed.json) — `2a5b65a2b74f327c461775ca6f19718bcba8f1a0a1699ed4382836aa34f7c7b7`
- summary: [final_summary.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-02_Splus_natural_s2/final_summary.json) — `a1fbbcb3b96e18708dc225d07ed8483b96c72f933d712c0c18afb7dadc9581c9`
- trace: [step_trace.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-02_Splus_natural_s2/step_trace.jsonl) — `53085de072a22af57d65ba7afae6428cacb82b935f6ea8b5c0ce308140f33011`
- matrix: [task_matrix.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-02_Splus_natural_s2/task_matrix.json) — `7f16ebca39d755cd85fbe98cf1e8844e2b42d491286b0abecfd8b2dbe71a4402`

## S1-03_Sminus_natural_s2
Branch=P; direction=g1_minus; mode=natural; output=/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-03_Sminus_natural_s2.
- config: [S1-03_Sminus_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/configs/S1-03_Sminus_natural_s2.json) — `283a065468112b89f66fbc2982ab66accc9d12dec2919ac22936ef895b3cec51`
- effective_config: [effective_config.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-03_Sminus_natural_s2/effective_config.json) — `cd36370de7e1ff1b936f236d4074758aeb4f35a0975bb05b8c23beb4012121bd`
- completed: [completed.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/runtime/jobs/S1-03_Sminus_natural_s2/completed.json) — `c0bf2bc6f87f21d93ce2befde2d0ed450c8c99e83d89fe51c8d25c07966f70dc`
- summary: [final_summary.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-03_Sminus_natural_s2/final_summary.json) — `abea6f6fe304b83c90af2ed508a4b0d3c84e8f0ba2277fe39c5b741666b4dcba`
- trace: [step_trace.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-03_Sminus_natural_s2/step_trace.jsonl) — `74145674d770ebdaf8796450d52a7fff47bf1f3244c70a4e3f487a068cdcb9af`
- matrix: [task_matrix.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-03_Sminus_natural_s2/task_matrix.json) — `84d4660599fbe6fdf64db9fb233783a528d86cbe47ea849008f3249569cbf36c`

## S1-16_CE_matched_s2
Branch=P; direction=g0; mode=matched; output=/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-16_CE_matched_s2.
- config: [S1-16_CE_matched_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/configs/S1-16_CE_matched_s2.json) — `76888fe6595db754d53598a77ce6a123291cb46ebab8c1d6ffcb28fce6ffe9b3`
- effective_config: [effective_config.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-16_CE_matched_s2/effective_config.json) — `ae7fe7c63b7902ae9821602559c07b088cca5d2705dc3f2829196ff0a1d38502`
- completed: [completed.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/runtime/jobs/S1-16_CE_matched_s2/completed.json) — `8a642e62ac92fa3acb466186bd1d5ec81ba889cb3fa35f21fe75f799977a564f`
- summary: [final_summary.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-16_CE_matched_s2/final_summary.json) — `1ecce49f4ce8a61118204c2ee3dee806bb5dab9c1f71e385d8cc2ba1131ab743`
- trace: [step_trace.jsonl](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-16_CE_matched_s2/step_trace.jsonl) — `787e7d77a2b9ba0c07b57ea07c7060f5140dee99cf59c15380b9338bf5b94652`
- matrix: [task_matrix.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/outputs/S1-16_CE_matched_s2/task_matrix.json) — `7850787f9171186ad6f12416cfe3a5183aa674feb9c6c0915e831da873a3a5e1`
