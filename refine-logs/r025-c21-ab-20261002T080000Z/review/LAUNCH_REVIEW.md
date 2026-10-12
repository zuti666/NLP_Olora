# A/B queue, configuration and deployment review

Reviewed at: 2026-10-02T08:06:14.589960+00:00

Verdict: **PASS for queue/configuration/deployment scope**. No concrete launch blocker found. This is a static and CPU read-only review; it does not claim that GPU smokes or formal jobs have run. Trainer numerical correctness and per-job validator are independently reviewed by their respective agents.

## Scope and configuration

- Exactly two full natural-step runs: `c21_base` and `c21_perp`, seed2, r=.25. Three two-task/one-batch smoke jobs: A/B plus old-US compatibility. No matched full run or C/D/E is queued.
- Independently compared all five JSON configs against the completed `screen_ug_natural_r025_s2` reference. Only `component_rule`, `prefix`, and `outputs_root` differ. Config SHA-256 values all match suite entries.
- Full commands include no task/epoch/batch truncation; smoke commands explicitly use 2 tasks, 1 epoch, 1 batch. Both formal jobs depend on all three smokes.
- All output and config paths belong to the new isolated deployed directory; outputs are distinct. No historical source, queue, state, checkpoint or output is mutated by the new campaign.

## Environment and provenance

- Suite pins `cudahpc40.cvc.uab.es`, the Pilot_new_local Python executable, physical GPU pool 0–6, and physical host PID namespace `pid:[4026531836]`.
- Independently verified all 190 frozen source-file hashes and the source-manifest hash.
- Independently verified canonical environment specification hash `574b252ec7b676947b1e431b88231959dcb69e79742d933f03cf354eb8c17798` and copied warm-witness hash. Read the actual environment ledger at `CopyFlatCV/.aris/compute/local.md`, which records this validated local environment. Deployment additionally checks installed package versions and witness provenance.
- Baseline release revalidates the nine completed training runs and old natural-US smoke with their frozen original validator. The proof pins completion receipts, complete summaries, traces, identities, configs, matrices, checkpoint maps and each validated checkpoint file. Unrelated failed HVP diagnostics correctly do not gate this independent A/B campaign.

## Queue lifecycle and failure behavior

- Fresh deployment refuses an existing target directory, freezes code/config/plan/review/environment hashes, copies isolated bytes, and verifies deployed hashes. Host and PID-namespace guards prevent launching through the container namespace.
- Pipeline validates references, runs only the three smokes, verifies all current smoke receipts plus old-US identity/trajectory compatibility, and only then starts the two formal jobs. A nonzero phase exit stops the pipeline.
- The formal scheduler checks the exact smoke-ID set, suite hash and receipt hashes and independently revalidates each dependency before launch. Run/source/config hashes are rechecked at launches and completion.
- Shared per-GPU `flock` leases, a second live idle/UUID check, UUID-based CUDA binding and inherited worker/trainer lease descriptors protect cooperating queues. Thresholds are memory <500 MiB and utilization <5%; no preemption occurs. Live host occupancy remains to be checked at actual launch.
- Single controller/pipeline locks and explicit launch receipts guard duplicate starts. A failed job stops new dispatch while letting already-active jobs finish. A PAUSE file stops dispatch. Completed outputs are validated before collection; partial scores are not presented as complete results.
- Result collection reuses the five natural baseline identities (including validated nominal-LR g1 identity) and includes A/B only after full validated completion. Comparisons retain the single-seed and natural-step scale-confounding limits.

## Checks executed

- Parsed every root Python script with `ast.parse`.
- Compared every configuration and independently checked each config hash.
- Asserted job counts/rules/scales, complete smoke dependency sets, command interpreter/trainer paths, output isolation and the physical GPU-pool/host/namespace manifest.
- Recomputed all source hashes and environment-spec/witness hashes.

No deployment, GPU launch, historical queue mutation or environment modification was performed by this reviewer.
