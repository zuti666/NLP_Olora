# Experiment tracker

Verified UTC: 2026-10-10T21:12:44.950562+00:00
Controller PID: 3710758; heartbeat confirmed; queue status `waiting_for_idle_gpu`.

Formal training:0/18 started at this verification. GPU4/5/6 occupied by existing jobs. Six native acceptance-smokes and the pairing gate will run automatically after idle admission; accepted gates then unlock all18 formal trainings. No job preempted.

| Task0 condition | Seed0 | Seed1 | Seed2 |
|---|---|---|---|
| sgd | pending | pending | pending |
| sam | pending | pending | pending |
| gam_residual | pending | pending | pending |
| sf_g2 | pending | pending | pending |
| sf_g2_then_sgd | pending | pending | pending |
| sgd_then_sf_g2 | pending | pending | pending |

Validated before controller launch:8trainer/controller CPUtests,5queue CPUtests,24literal training CLIchecks, independent implementation41checks, fresh document71checks. All57 sealed artifacts/source190 unchanged. No native GPU smoke or formal scientific score is claimed yet.

Liveprogress:runtime/queue_state.json. Actualhost launch/proof:runtime/LAUNCH_RECEIPT.json andLAUNCH_VERIFIED.json. Acceptedscientific results:FORMAL_RESULTS.csv,AGGREGATED_RESULTS.json,PAIRED_SEED_DIFFERENCES.csv,TASK1_RETENTION_CURVES.csv andRESULTS.md.

The six conditions and allTask1epoch0..20 metrics are specified inEXPERIMENT_PLAN.md/protocol.json. Only the independent20epoch cohort is pooled.


## Latest completion and GPU6 request

The initial waiting-state snapshot above is historical. Durable queue state now records all25 jobs accepted, including18/18 formal trainings acrossseed0/1/2; no pending/running/failed jobs. FinishedUTC2026-10-10T23:46:53.932836+00:00 (Europe/Paris2026-10-11 01:46:53).

User resource steering on2026-10-11: subsequent experiments useonlyphysicalGPU6, maxparallel1, idle admission. This completed campaign is not duplicated; its frozen GPU456manifest and execution receipts remainunchanged. New manifests must honor runtime/NEXT_RUN_GPU_POLICY.json. Results are inRESULTS.md andAGGREGATED_RESULTS.json. Independentcompletion audit is being written underresults/status_gpu6_*/.
