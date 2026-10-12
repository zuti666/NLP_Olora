Independent fresh document check: **passed**. All six literal RUN.md CPU commands exited 0; 71 checks are true. No document/reality divergence was observed.

| Command | Result |
| --- | --- |
| Trainer unit tests | 8 tests, exit 0 |
| Queue unit tests | 5 tests, exit 0 |
| Formal GAM dry-run | valid; 2 tasks; 20 epochs/task; Task1 SGD |
| Formal early-G2 dry-run | valid; 2 tasks; 20 epochs/task; Task1 SGD |
| Short late-G2 dry-run | valid; 2 tasks; 20 epochs/task; Task1 SGD |
| Actual-host queue dry-run | valid; 25 jobs; 18 formal; 6 smokes; GPUs [4, 5, 6]; max parallel 3 |

Queue dry-run identity: `cudahpc40.cvc.uab.es`, `pid:[4026531836]`. The 57 sealed inputs and RUN.md remained unchanged.

Exact commands, arguments, documented environment, working directory, exit codes, stdout/stderr, and SHA256 evidence are retained in [FRESH_DOC_CHECK.json](FRESH_DOC_CHECK.json) and the six `FRESH_LITERAL_CPU_*.log` receipts with pure stdout companions. This check covers CPU invocations; CUDA witnesses and GPU smoke acceptance are separate execution gates.
