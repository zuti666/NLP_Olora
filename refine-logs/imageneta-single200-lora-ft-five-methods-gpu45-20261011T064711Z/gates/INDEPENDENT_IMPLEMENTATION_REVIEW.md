# Independent single200 implementation review

`status: passed`

Read-only review of the final frozen campaign. Both commands in RUN.md were executed verbatim with empty CUDA_VISIBLE_DEVICES in the actual host namespace and exited 0. No GPU lease was acquired and no training output was created.

Full FT deep-copies the untouched pretrained ViT origin and the same FC200 head, consumes no new random initialization, rejects any retained LoRA or frozen parameter, and requires actual nonzero first-step backbone and FC motion. LoRA retains only Q/V rank10 factors and the same classifier as trainable parameters.

The implementation uses SGD g0, SAM g1+, mirror g0+(g0-g1-), g2 at theta0+r*N(g1+-g0), and complete four-read GAM. GAM retains the source-native sequential norm, cosine epsilon placements, and gamma=.1 decomposition. All probes restore the entry parameters, gradients, buffers, modes, controls/caches, and R0; a single ordinary SGD commit occurs from theta0 and only base R1 advances. Runtime hooks measure actual model forwards.

The complete fixed 200-class 5981/1519 split, 50 epochs, batch128, 47 batches/epoch and 2350 updates are enforced. The seven new formal jobs total 16450 SGD updates and 42300 training gradient reads. Ten smokes total 20 updates and 48 reads. Formal jobs wait for all ten accepted smokes and CPU pairing; the three reused LoRA rows must match their first two realized inputs and update vectors.

GPU admission is limited to physical 4/5, at most two active jobs, no compute PID, memory below 500 MiB and utilization below 5%, with shared physical-index and UUID leases rechecked after locking. Inherited leases survive controller restarts through workers/trainers. Pairing imports only the standard library and is launched with no GPU assignment.

All 44 final execution hashes were verified. Queue dry-run verifies all 190 source files. This reviewer independently verified all 121 predecessor execution pins and 22 reuse receipts. The earlier cache output validation, measured-forward/FT-motion acceptance, and result-documentation findings are resolved; no unresolved issue remains.

Real GPU smokes and worker CUDA witnesses remain pending and are enforced before formal admission. This review does not claim completed GPU validation or experiment results. Root's CPU_ACCEPTANCE records 19 passed CPU tests; this reviewer reviewed that artifact and did not rerun those suites.

Checks (each true):

- `documented_queue_dry_run_passed_actual_host`: true
- `documented_trainer_dry_run_passed_actual_host`: true
- `all17_configs_and_command_arguments_match_registered_protocol`: true
- `full_ft_plain_vit_without_lora_all_backbone_and_fc_trainable`: true
- `runtime_full_ft_backbone_and_fc_displacement_is_required`: true
- `same_pretrained_backbone_and_fc_initialization_is_enforced`: true
- `first_two_inputs_and_three_reused_lora_update_vectors_are_gated`: true
- `g2_outer_point_is_theta0_plus_r_N_g1_minus_g0`: true
- `complete_gam_four_reads_and_exact_source_gamma_formula`: true
- `actual_native_gradient_budgets_are_1_2_2_3_4`: true
- `exact_state_R0_restore_base_R1_advance_and_one_real_sgd_update`: true
- `measured_forward_counts_are_required_by_acceptance`: true
- `supervised200_full5981_train1519_test_50epochs2350updates`: true
- `no_fisher_weight_merge_replay_history_HVP_or_final_matching`: true
- `only_physical_gpu4_gpu5_max2_idle_thresholds_and_shared_leases`: true
- `cpu_pairing_does_not_claim_gpu`: true
- `all7_formal_runs_wait_for_all10_smokes_and_pairing`: true
- `all44_frozen_execution_pins_verified`: true
- `all190_source_files_verified_by_queue_dry_run`: true
- `all121_historical_execution_pins_and22_reuse_receipts_verified`: true
- `dry_runs_created_no_training_outputs`: true
- `corrected_results_paths_and_cost_reporting_match_implementation`: true

Documented dry-run evidence:

```bash
CUDA_VISIBLE_DEVICES='' /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/run_queue.py --suite /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/suite.json --dry-run
```

Exit 0; namespace: actual host; CUDA_VISIBLE_DEVICES empty.

```text
{'status': 'valid', 'jobs': 18, 'smokes': 10, 'formal': 7, 'physical_gpus': [4, 5], 'max_parallel': 2}
```

```bash
CUDA_VISIBLE_DEVICES='' /home-local/liying/.conda/envs/Pilot_new_local/bin/python /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/trainer/train_single200.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/formal_full_ft_g2_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/outputs/formal_full_ft_g2_s2 --dry-run
```

Exit 0; namespace: actual host; CUDA_VISIBLE_DEVICES empty.

```text
{"status": "valid", "protocol": "imageneta_single200_lora_fullft_five_methods_native_fp32_exact_restore_v1", "train_scope": "full_ft", "method": "g2", "epochs": 50, "batches": 47, "ordinary_gradient_calls": 3}
```

Evidence hashes are SHA256 of the exact reviewed files. This report and FRESH_DOC_CHECK.json are intentionally excluded from execution pins to avoid a reference cycle.

| File | SHA256 |
|---|---|
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/RUN.md` | `6bf26d57eed79f14c750c105c50df0fefe01d5d2e40e63f9b504678bd22cdb55` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/RUN_PINS.json` | `9982fb5916ab2903361033779fd0717ec005409a40cc483982ccbe013465018f` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/suite.json` | `abcfe74d856917ec212adbf1a9be3c3897b1628234c20fee7c8200c9dc92b714` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/source_manifest.json` | `e8a827ae67768a73a59106914477905d051da0646fcb0ffb7c3a13b23d337ab2` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/ENVIRONMENT_REUSE.json` | `729ff5a30496d105ea46851994d395df77e2d30104ea2d7317b87eeab7a16de6` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/common.py` | `1083d3f3b1b976809bb94feab73ede97155b7bc84400a9de37e2327a741509e1` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/run_queue.py` | `832ea97dbe5be0e6ab30da7b4946135be915782ecec15c1ff2be77eca5e72d79` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/job_worker.py` | `b872c9bd73d439d4e466bfb7af4c3f39349ad0d80726c3ebe79b04a4d9a7bdec` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/validate_job.py` | `95b2fba3f0becd38d3668a2be5e4d4d5f53af5a6499bf6c1b0afd0e1c440a53f` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/collect_results.py` | `ffa80d5e906de70891027236b24715f7aaf74a60509bb4149460c3e676df61db` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/test_queue_cpu.py` | `2db2e7eb26b4994bb7124708c79274ce96bf1c25d34f611a5c01351315b32e2d` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/gates/pairing_gate.py` | `356ea6a30d28202dd0493d8ba7fd03f622c1a6a78ff29a6dd2f968e432eb517e` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/gates/CPU_ACCEPTANCE.json` | `c7f28e2023b73d6df68d01515d5ba23cb58f8326d207258041dcc2e836eb53dc` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/gates/REUSED_RESULTS.json` | `28de555cc96efb2aeae643b2ffe5f9365711dfdee3947201d083b900f68a2d0a` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/trainer/train_single200.py` | `0fce66cf4abc8189c25f514fd6e61fa46492dc217f508336a79c0059ab4cde25` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/trainer/scope_helper.py` | `4e46d2f6a9fe81932238e98230283316fff4b2e5be77d3fc98b457f7b6d45333` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/trainer/five_method_step.py` | `ac4575125b745d3daef8a105e5bc2dae5bf00c15de1ee4b15ba4f756cd92601a` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/trainer/state_transaction.py` | `2b23480cf296f3a20ef6d5f93e14af5498680661f1f8ef19d6196f83ce80cfee` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/trainer/cuda_witness.py` | `f4245c35969024d40842f767b86ed78d008fe6f60cfceee8eeef7e320fbe0f7a` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/trainer/test_five_method_step_cpu.py` | `944ec4d85a79cfe62015768da627fa9dcf9322897d7c296b8e9c9f04f593a2b3` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/trainer/test_single200_scope_cpu.py` | `bc216d251e7fbf9874048df1fea5d55fda7825f419684e2648f0f93a4fc58790` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/trainer/five_method_cpu_validation.json` | `eb1d76020f8dbafee867ccfb6ac4c7634480b1d8c269673e85e007e0e75d20e9` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/trainer/train_single200_cpu_validation.json` | `72c5bf59f85f4d37ee7d7465c6a45b9805145429bcbcd2b39a33573fd585a069` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/formal_full_ft_g2_s2.json` | `ddbfc029733ae641cee934ce84b21c11a43ecf613c4bd261ada5d1aa23e4cd70` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/formal_full_ft_gam_s2.json` | `e72a20efee400942df7ad0e7e1018ac8db48c27fbb65a3d4c99e22eef3ff18f1` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/formal_full_ft_mirror_s2.json` | `66f3b8b2d11a57e74bbf4c64a2a7dbca0915ce6fc0de9bccef8ac45e4a3f9220` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/formal_full_ft_sam_s2.json` | `781b58dfddd4121baf7cb878435bda04f89d68f59111445f618af7e9b2f42b3a` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/formal_full_ft_sgd_s2.json` | `99ce6c1b81fa437e6ebfbfb825ab0e5b4283e51103e21a59876fce87d443b156` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/formal_lora_gam_s2.json` | `ee356a1ebc308bb7a22f6e74b156495930a1207dad970371f349d7d52efae7f5` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/formal_lora_mirror_s2.json` | `6ea4975d89f5795080952970b9edfaa2a50b861a564e045e27bba7e30861e29b` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/smoke_full_ft_g2_s2.json` | `e295f8606f3fb8461cdc920ec823efcda6f841f760120afa31908685bac92bdb` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/smoke_full_ft_gam_s2.json` | `56a0e945bfc99ab2e80386a747227cafb2ee0e799e9d8680bd82f4b8f583b6e9` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/smoke_full_ft_mirror_s2.json` | `1a86802268c96e0389ed5d4df1a0d2c71c56caec54b12874f834c8fbe4d71805` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/smoke_full_ft_sam_s2.json` | `d2bf5a3dc1f471349e4d387927aa8f21e2ffdb001f4f64af759c722d444080ef` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/smoke_full_ft_sgd_s2.json` | `ef9bad353419dc0922667a82d72a8a5515af31b6e1fe4e83e497647ee9152fa4` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/smoke_lora_g2_s2.json` | `e760d7acbeb648b9f0f8836c34cc077862fc1b6e5887bd11bd3f967aeb0752a9` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/smoke_lora_gam_s2.json` | `1c407c451df002cf63b58ccba13a901570dfc187fdf0e7e610b72be5dcfa7bf8` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/smoke_lora_mirror_s2.json` | `c79eab51f259228a34e87a0fffee9c6af50c27f6093aa06738899d5f78d712f6` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/smoke_lora_sam_s2.json` | `b468a44b8eef0a85457636713a4421c452b4148ed7c9704eeb914b1a4aaa82f5` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-single200-lora-ft-five-methods-gpu45-20261011T064711Z/configs/smoke_lora_sgd_s2.json` | `39f699621283f8f73e8771930983785b7c5c44e4023b191e6d1914776a0a51f1` |
| `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/backbone/lora.py` | `303ed752ccf68b3a42106aedfc134e97e4305af0b013a45aea7f41ea7665dc86` |
| `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/utils/inc_net.py` | `173d1a92350771616368c30e4e0d9a451d69baa1f7058de0d2bf22fc5e21b600` |
| `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/utils/factory.py` | `dc7a7ef2972652d9c51c6ad060e005150bbc21c73bab7f974504886332746b64` |
| `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/utils/data_manager.py` | `8a1bc22828e80071990286b1eebc7c40ff478601f089a6c7316ad2eb19b119f6` |
| `/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source/optimer_PerturabtionType/gam.py` | `0b37fca57a31952d911eba4ab99534b7952a6271c1f2c37dbf7a45b72fa5e40e` |
| `/data/140-1/users/liying/O-LoRA/refine-logs/imageneta-upward-eight-fork-20261010T123703Z/RUN_PINS.json` | `bc6c458f4cb5aed53012a61e31829d5dcef4b7426992bc45f216723b7c81384e` |

