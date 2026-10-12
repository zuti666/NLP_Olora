# 配置与主要代码映射

源代码路径用于定位历史实现；运行时应创建独立的新campaign薄包装，不能直接调用旧目录队列。

共同native入口：`/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/src/main.py` → `src/trainer.py` → `utils/factory.py` → `models_LoRAbasedCL/as1_normfisher_gam_intask_damp_weightmergethenfisher.py::Learner` → 原生GAM优化器。配置model_name虽然含ewc/fisher字样，本轮关闭这些执行开关，并检查实际计数。

原生GAM四探针与组合：source/optimer_PerturabtionType/gam.py。原生核心后三项：s3-puregam-r1-complete7-20261002T074400Z/legacy_direction.py。原生A/B：s3-r1-c21-ab-20261002T115100Z/legacy_direction.py与residual_selection.py。LR/Gmatch：s3-r1-lr-controls-20261002T191527Z/legacy_direction.py、step_reference.py、effective_step.py。四探针：g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py、direction_step.py、native_lifecycle.py、effective_step.py。

|条件|历史source config|实际runner模板|选择器|
|---|---|---|---|
|N01 original_GAM|[s3_puregam_original_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/configs/s3_puregam_original_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/run_job.py)|原生beta/γ|
|N02 Sum|[s3_puregam_gamma0_sum_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/configs/s3_puregam_gamma0_sum_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/run_job.py)|原生beta/γ|
|N03 g1 / S（审计基准）|[formal_S_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/configs/formal_S_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/run_job.py)|g1_control|
|N04 g3|[s3_puregam_gamma0_g3_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/configs/s3_puregam_gamma0_g3_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-direction-run-20260930T180800Z/run_job.py)|原生beta/γ|
|N05 g0|[formal_g0_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-r1-complete7-20261002T074400Z/configs/formal_g0_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-r1-complete7-20261002T074400Z/run_job.py)|g0|
|N06 g2|[formal_g2_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-r1-complete7-20261002T074400Z/configs/formal_g2_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-r1-complete7-20261002T074400Z/run_job.py)|g2|
|N07 ug|[formal_ug_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-r1-complete7-20261002T074400Z/configs/formal_ug_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-puregam-r1-complete7-20261002T074400Z/run_job.py)|ug|
|N08 A|[formal_c21_base_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-c21-ab-20261002T115100Z/configs/formal_c21_base_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-c21-ab-20261002T115100Z/run_job.py)|c21_base|
|N09 B|[formal_c21_perp_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-c21-ab-20261002T115100Z/configs/formal_c21_perp_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-c21-ab-20261002T115100Z/run_job.py)|c21_perp|
|N10 Mean|[s3_puregam_meansum_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/configs/s3_puregam_meansum_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-g1g3-mechanism-run-20260930T210711Z/training/run_job.py)|原生beta/γ|
|N11 S1.5|[formal_S15_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/configs/formal_S15_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/run_job.py)|g1_control|
|N12 S2|[formal_S2_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/configs/formal_S2_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/run_job.py)|g1_control|
|N13 Mean2|[formal_Mean2_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/configs/formal_Mean2_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/run_job.py)|mean|
|N14 Gmatch|[formal_Gmatch_r1_k10_s2.json](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/configs/formal_Gmatch_r1_k10_s2.json)|[run_job.py](/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/run_job.py)|gmatch|
|P01 S1-01_CE_natural|[S1-01_CE_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/configs/S1-01_CE_natural_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|g0|
|P02 S1-02_Splus_natural|[S1-02_Splus_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/configs/S1-02_Splus_natural_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|g1_plus|
|P03 S1-03_Sminus_natural|[S1-03_Sminus_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/configs/S1-03_Sminus_natural_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|g1_minus|
|P04 S1-04_O_P_natural|[S1-04_O_P_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/configs/S1-04_O_P_natural_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|g2|
|P05 S1-05_O_D_natural|[S1-05_O_D_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/configs/S1-05_O_D_natural_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|g2|
|P06 S1-06_O_C_natural|[S1-06_O_C_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/configs/S1-06_O_C_natural_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|g2|
|P07 S1-07_A_P_natural|[S1-07_A_P_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/configs/S1-07_A_P_natural_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|A|
|P08 S1-08_A_D_natural|[S1-08_A_D_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/configs/S1-08_A_D_natural_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|A|
|P09 S1-09_A_C_natural|[S1-09_A_C_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/configs/S1-09_A_C_natural_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|A|
|P10 S1-10_O_R_natural|[S1-10_O_R_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/configs/S1-10_O_R_natural_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|g2|
|P11 S1-11_A_R_natural|[S1-11_A_R_natural_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/configs/S1-11_A_R_natural_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|A|
|P12 S1-12_A_P_matched|[S1-12_A_P_matched_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/configs/S1-12_A_P_matched_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|A|
|P13 S1-13_A_D_matched|[S1-13_A_D_matched_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/configs/S1-13_A_D_matched_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|A|
|P14 S1-14_A_C_matched|[S1-14_A_C_matched_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/configs/S1-14_A_C_matched_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|A|
|P15 S1-15_A_R_matched|[S1-15_A_R_matched_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/configs/S1-15_A_R_matched_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|A|
|P16 S1-16_CE_matched|[S1-16_CE_matched_s2.json](/data/140-1/users/liying/O-LoRA/refine-logs/g012-two-stage-execution-20261002T122014Z/configs/S1-16_CE_matched_s2.json)|[four_probe_train.py](/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py)|g0|

## seed参数化与协议定位

N族/LR适配：
```json
[
  {
    "file": "/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/common.py",
    "lines": [
      230,
      237,
      241
    ],
    "change": "GPU allowlist[4,5,6], parallel3; seed0/1 manifest+shared-config equality; add Mean arm spec mean/LR1/r1"
  },
  {
    "file": "/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/step_reference.py",
    "lines": [
      52,
      88
    ],
    "change": "replace seed2 assertions with explicit candidate/reference same requested seed0/1; keep budget/hash/batch/RNG alignment strict"
  },
  {
    "file": "/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/validate_smokes.py",
    "lines": [
      197,
      207,
      208
    ],
    "change": "group fixed s2 smoke identifiers by seed; each Gmatch binds same-seed completed S"
  },
  {
    "file": "/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/summarize_controls.py",
    "lines": [
      33,
      36,
      37
    ],
    "change": "seed-aware reports; normal-LR Mean primary repeat arm, preserve old/new S provenance"
  },
  {
    "file": "/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-r1-lr-controls-20261002T191527Z/suite.json",
    "change": "fresh immutable per-seed arm rows; same-seed S dependency before Gmatch, r1 only; do not reuse resource_wrapper with old GPU2/3/job/hash hardcodes"
  }
]
```

P族适配：
```json
{
  "trainer": "native_lifecycle.resolve_config already supports seeds 0..4 and restores actual seed after its deliberate seed2 surrogate. Change four_probe_train summary/dry-run seed literals to resolved run seed; retain numeric helpers.",
  "builder_contract": "Create fresh isolated suite containing all16 formals for EACH requested seed; do not inherit skip-four-baseline logic or seed2 REUSE_BASELINES. Add job.seed and compare configs to seed2 templates allowing only actual requested seed and provenance.",
  "validator": "Compare summary/trace seeds to job.seed, not literal2; preserve all budgets, matching tolerance and scientific checks.",
  "smoke_gate": "Pair initial parameters/data within seed; separate baseline identities keyed seed. All32 smokes must pass before32 formals.",
  "queue": "No seed-specific algorithm change; use fresh allowed_gpus=[4,5,6]/max_parallel=3 and original shared index/UUID leases, namespace checks and fail-closed state/receipts.",
  "collector": "Record actual job seed. Preserve per-seed metrics; compute mean and sample SD only over completed seed0,seed1,seed2 rows in same protocol/arm.",
  "environment_witness": "Canonical tiny CUDA witness may retain fixed seed2 as ENVIRONMENT evidence; it is not a training-seed assertion."
}
```

code_locations
```json
[
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/native_lifecycle.py",
    "line": 132,
    "finding": "Already accepts one integer/list seed 0..4; line141 deliberately uses seed2 only for the base development resolver and line143 restores actual run seed. Keep numerical helper bytes unchanged."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/native_lifecycle.py",
    "line": 362,
    "finding": "Controller consumes actual scalar cfg seed after src.trainer.train iterates its seed list."
  },
  {
    "path": "/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/src/trainer.py",
    "line": 823,
    "finding": "Native train supports seed list; converts each run to scalar seed at827 and calls set_random at910."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py",
    "line": 271,
    "finding": "Dry-run prints literal seed2; plan replacement by resolved run seed."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py",
    "line": 298,
    "finding": "Summary writes literal seed2; plan replacement by resolved run seed."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/validate_job.py",
    "line": 110,
    "finding": "Training summary seed is compared to literal2; compare to explicit job.seed."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/validate_job.py",
    "line": 141,
    "finding": "Every actual trace row seed compared to literal2; compare to job.seed. Dead diagnostic/HVP checks at19/62 remain out of this Stage1 scope."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/build_suite.py",
    "line": 43,
    "finding": "Skips all four baseline formals; new seed campaign must instantiate them rather than importing seed2 reuse."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/build_suite.py",
    "line": 45,
    "finding": "IDs inherit suffix_s2; new seed manifest needs IDs and config seed matching0/1 plus explicit job.seed."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/experiment_contract.py",
    "line": 41,
    "finding": "Scope fixed12formal/16smokes and seed2 historical config equality at50; new contract must enforce32formal/32smoke, exact16conditions per seed, and allow only requested seed/provenance changes against templates."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/experiment_contract.py",
    "line": 10,
    "finding": "Reuse audit contains four seed2 baselines; not valid for new0/1 runs. Preserve historical evidence, do not mark it as completed0/1."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/validate_smokes.py",
    "line": 11,
    "finding": "Requires16smokes; new pair gate must have16per seed and32total."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/validate_smokes.py",
    "line": 13,
    "finding": "One global baseline identity; split by seed so expected seed0/1 initialization/data differences are not falsely rejected."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/collect_results.py",
    "line": 34,
    "finding": "New row seed literal2; record explicit job.seed. Seed2 reuse row literal at50 is appropriate only for source seed2 context."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/collect_results.py",
    "line": 64,
    "finding": "Report wording fixes seed2 and12new+4reuse; update isolated campaign to32fresh0/1, keep protocol and provenance explicit."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/common.py",
    "line": 19,
    "finding": "ALLOWED_GPUS fixed2/3; new campaign should declare only user requested4/5/6 withmax3."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/experiment_contract.py",
    "line": 28,
    "finding": "Contract fixed GPU2/3 max2; new isolated contract should strictly require4/5/6 max3."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/run_queue.py",
    "line": 24,
    "finding": "Queue has no training-seed restriction. Preserve verify_pins/contract, process identity, same shared GPU index+UUID leases, failure policy, and dependencies."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/cuda_witness.py",
    "line": 12,
    "finding": "Canonical tiny witness seed2 is environment evidence; not a scientific training-seed override."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/native_lifecycle.py",
    "line": 299,
    "finding": "Four-probe data sampler/augmentation stream seed=runseed*100000+task*1000+epoch and model/data RNG switching at303."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/direction_step.py",
    "line": 152,
    "finding": "Four CE gradients g0,g1+,g1-,g2; q signs at160-164, selected A at166-168; every probe replaysR0 and finalRNG fromg0 at195-204."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/direction_step.py",
    "line": 275,
    "finding": "Matched target is actual same-candidate-state rawg1+ SGD BA+FC delta, not independent completed-S trajectory."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/four_probe_train.py",
    "line": 25,
    "finding": "Observational evaluation restores full model/cache/RNG; used aroundeval_task at209 and periodic accuracy at222."
  },
  {
    "path": "/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/models_LoRAbasedCL/seqlora.py",
    "line": 398,
    "finding": "Native family constructs original global-RNG loader with no explicitper-epoch generator; shuffleTrue means sample shuffle, while class order stays fixed."
  },
  {
    "path": "/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/optimer_PerturabtionType/gam.py",
    "line": 248,
    "finding": "Native fourclosure chain isg0->g1+->g2->g3; no R0 replay and subtractive unperturb at258/277, unlike exact restoration."
  },
  {
    "path": "/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/models_LoRAbasedCL/as1_normfisher_gam_intask.py",
    "line": 1211,
    "finding": "Native additionally computes clean task gradient before GAM and then applies original clean FP32 direction projector at1234: fiveCE forwards/update total."
  },
  {
    "path": "/data/140-1/users/liying/O-LoRA/refine-logs/g012-r1-four-branches-20261003T072200Z/trainer/direction_step.py",
    "line": 24,
    "finding": "Four-probe normalizer usesFP64 global inner products; native gam._grad_norm at168-200 accumulates source gradient dtypeFP32."
  },
  {
    "path": "/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/utils/data.py",
    "line": 719,
    "finding": "Both share ImageNetA train/test transform recipe from frozen source; different sampler/worker seeding changes actual augmentation realizations, not a claim of different transform operators."
  }
]
```
