Independent review: PASS for the reflected-probe implementation and gated six-chain controller.

Final reread: 2026-10-12T10:46:51.289011+00:00. Reviewer: fresh independent subagent /root/reflected_probe_fresh_review.

The first documented dry-run command was executed verbatim and repeated after final launcher repairs. Both executions exited0; /tmp/reflected_probe_docs_20261012 was not created. The inspection code is stdlib-only: the independent six-profile plan witness did not import Torch or query CUDA. The canonical environment SHA256 matches nlora_req@0715917e; no package rebuild or installation was performed. Exact command and structured evidence are in DOC_CHECK.json.

The reviewed method computes clean preclip global-mean g0, uses the normalized descent point BASE−.05*g0/(norm(g0)+1e-12), computes one replayed CE gradient gminus, restores BASE, and submits 2*g0−gminus. SAM submits the corresponding ascending gradient; the clean mode submits g0. All three commit through one Transformers AdamW with eps1e-6, weight decay0, betas.9/.999 and one final clip to norm1. There is no outer probe, g2/g3 computation, Fisher/EWC, projection, history fusion or virtual AdamW-state prediction.

The independent nonlinear CPU oracle matched analytically computed directions and exact HF AdamW parameters for all three modes. It asserted BASE at the sole commit, observed1/2/2 CE phases and one optimizer commit per mode, and confirmed CUDA remained uninitialized. Trainer inspection confirms identical cached effective windows, per-microbatch Python/NumPy/CPU/selected-device RNG replay, post-clean RNG restoration, native DDP global mean for clean gradients, no_sync replay plus one explicit mean, and no re-reduction of the submitted direction. Entry/trainer guards require decoder_token_ids_v2 and actual FP32 LoRA trainables. The authored tiny-Llama/BF16 and CPU-Gloo receipts were inspected; broad tests were not rerun by this reviewer.

The six active profiles have identical fixed protocol options within each model except the selected mode and run paths. Scope is only1B/3B × clean/SAM/reflected × Order1 × seed0: six chains,24 formal stages,12 bounded smoke stages,3,084 real commits,5,140 CE-gradient phases and2,056 replay phases. Other Orders and seeds exist only in the inspection catalog. Fresh two-task smoke and same-model first-window trainable/batch/clean-gradient hashes gate formal admission.

Queue inspection confirms authorization is limited to physical0/1/2/3; all four must be idle, compatible shared index/UUID locks are held, and a second idle snapshot closes reservation races. Every fresh reservation performs seeded per-card CUDA/BF16 witnesses. STOP/PAUSE is rechecked after witnesses; PAUSE releases leases and resumption repeats reservation/witness admission. Frozen source file population, source hashes, local model/data/tokenizer/Python/environment inputs and manifest are rechecked before stages. Existing outputs are preserved and unexpected failures stop the queue. The10-test launcher receipt, including witness-arrival pause/stop tests, was inspected as PASS.

Review-reported defects were repaired and the patched files reread: sentinel arrival during witnesses; weak smoke prediction identities/gold labels; missing collector source/artifact pins and runtime/paired metric exports; resumed receipt reblessing; unpinned preceding adapter weights; unpinned adapter configuration; and formal receipt identity/budget mismatch. Accepted receipts now include adapter weight and adapter_config SHA256, refuse replacing differing receipts, compare resumed receipts to stored state, and check both prior weight/config hashes before continuation. Collector validates source population/hashes, full formal receipt identity/budget, accepted audit/adapter/config/prediction/trace pins and canonical pinned-gold coverage. Fresh same-campaign controls provide FAA/AAA/BWT/learning accuracy and paired deltas; smoke and incomplete chains receive no final metrics. Shared official normalized exact-match and task-equal metrics are correct; no old baseline is used.

No unresolved substantive defect was found in the final reviewed sources. Root remains responsible for its final positive collector regression and frozen preparation check before launching the controller. This review has performed no GPU reservation, CUDA witness, native1B/3B smoke, training, process signal, network call or external notification. GPU/model acceptance remains unperformed and mandatory; controller startup and CPU checks cannot certify it. BF16 cast proxies remain observational and explicitly do not prove forward visibility. Single-seed Order1 findings, if obtained, remain preliminary.

Final reviewed SHA256 pins:

- src/optimer_reverse_probe.py: a56921aeed9c159ca6e691acdbeca5eddd2e6c6ba51b3b39a9121c0fdbeb8133
- src/uie_trainer_lora.py: d362756d79596662c4ed6857e591887121a89361659f2534dd65c3ab7c128336
- src/run_uie_lora.py: b4f6cbbf951ac1b88b6f9a069a8b5ea29d6273ff12108d8cd204647036d3949b
- scripts_llama/run_two_point_probe_llama.py: e67a503d66d3e3f9d51f421eeb974d13b32ac3a3d3c21fcbbbb7426da52f05c6
- scripts_llama/run_two_point_probe_queue.py: 6471e177a9726fb2b93349ee40f5858192703e7b01c1f897fddaeb7395ab282a
- scripts_llama/collect_two_point_probe_results.py: 6a2fa06b6bcf5c84db7705f0742cf57d8c39ddeb69957079bde41d4ed78200d9
- scripts_llama/run_g2_llama.py: 045c73d8e1aa51292bad699862c899bcc9cd2574d4d95f14884b3f1a47ff2ebb
- scripts_llama/collect_g2_orders_results.py: 33e09eb23fb75feee9fad2c50d3b73d866f98a60db4c6e1e268a7c7bc23633b2
- configs/two_point_probe_llama/inclora_probe_reverse_llama1b_order1_tokv2_rho005.json: 8ca2410d0446ff3dac0fbcbd5a1260fbddca79d4082b7a0ae289622c1766f3e5
- configs/two_point_probe_llama/inclora_probe_reverse_llama3b_order1_tokv2_rho005.json: 8fad3c654f88443e5f8198c45f63a8a058acbf840dc762033f64322ac8dc705d
- configs/two_point_probe_llama/inclora_probe_sam_llama1b_order1_tokv2_rho005.json: 892c4f248d38edf2e148049a35d8f85d220431d73a7c1ba2de4bc2e309e0111a
- configs/two_point_probe_llama/inclora_probe_sam_llama3b_order1_tokv2_rho005.json: eb76cf11b8cca117ce11c04eb6dae3b4e5a5c26b8a4df9bf4b15c036919b8c90
- configs/two_point_probe_llama/inclora_probe_sgd_llama1b_order1_tokv2_rho005.json: 6810cc46eac42dd4aa54a210909ca9d43af30aec9330a712d39ee97a6a065893
- configs/two_point_probe_llama/inclora_probe_sgd_llama3b_order1_tokv2_rho005.json: 3cb6cb24d03bfed3848fe4c4d60af7e3bd37ed65219c819d2da158c6ea1f9cd7
