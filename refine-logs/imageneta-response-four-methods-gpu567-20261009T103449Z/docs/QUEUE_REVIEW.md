# GPU 5/6/7 formal queue review

Passed the independent prelaunch queue-design review. This is scheduler and configuration verification, not a formal accuracy result.

The seven commands request full 50-epoch training: four full-200 single-task runs (2,350 SGD updates each), followed only after all four have accepted completion receipts by three ten-task runs (2,550 updates each). The ten-task history-fusion result is reused separately and is not retrained here. Scientific configurations match their accepted plan drafts; only output paths changed. The eleven trainer files are byte-identical to the accepted eight real-GPU bounded smokes and pairing gate. All 190 frozen source files and the prior implementation evidence match their hashes.

CPU fixtures execute the actual scheduler with simulated GPU queries, workers and acceptance while retaining real index/UUID file locks. They verify absent GPU 7, GPU 5 occupied, a card becoming busy after lock acquisition, and an injected completion-validation failure. The scheduler stays on physical GPUs 5/6, retains both leases, respects all four phase dependencies, and stops adding work after failure. Eight invalid or unauthorized allowlists are rejected. These fixtures do not dispatch CUDA or training.

The worker forwards both lease descriptors through the CUDA kernel witness and training child. PID recovery includes process start time, and a zombie is treated as exited. Dependency summary hashes are checked before admission. Pin verification now rechecks the actual source files on every admission and completion check.

Before launch, seal the run pins, perform the real-host dry run and GPU check, then verify actual finite SGD updates. The results collector must report only fully accepted results and preserve reused HRF provenance. Runtime shared locks apply to cooperating queues and cannot prevent an unrelated user from allocating a card after the final idle check.
