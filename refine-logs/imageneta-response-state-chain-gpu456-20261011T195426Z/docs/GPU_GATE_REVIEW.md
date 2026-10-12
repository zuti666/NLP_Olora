# Independent GPU launch and short-gate review

Status: **PASS**. Observed 2026-10-11T20:19:07.919202+00:00 UTC on cudahpc40.cvc.uab.es.

The three exact Task0 SGD parents were fully restored, the classifier was expanded once, and each paired branch inherited one common full model and RNG state. All 51 registered inputs and 190 frozen source files remain unchanged.

Both native short branches completed four real SGD commits. S used six CE gradient reads; R used eight. The switch preserved model, optimizer, scheduler, parameter references, and RNG, continuing the 50-epoch cosine schedule at LR 0.01998224056. All 13 pairing checks passed, including all four realized augmented inputs and the first base gradient. Fisher, model fusion, projection, final update rescaling, HVP, and history were absent from training.

The offline pilot submitted zero updates and restored the full state. Native/math gradient relative error was 6.92e-6. Hg finite-difference relative errors were 0.00194 / 0.00100 / 0.00629; HHg errors were 0.01076 / 0.04920 / 0.11719. The first two adjacent HHg points satisfy the registered 0.08 threshold; the third failure is retained. Thresholds were unchanged.

| GPU | Formal job | Actual trainer PID | Observed real commits | Latest observed epoch |
| --- | --- | --- | --- | --- |
| 4 | formal_S_s2 | 1654381 | 112 | 14 |
| 5 | formal_R_s2 | 1654412 | 84 | 11 |
| 6 | formal_S_s0 | 1654464 | 113 | 15 |

Queue snapshot: {'completed': 7, 'running': 3, 'pending': 6}. All seven prerequisite jobs completed before formal training. Every observed formal update was finite, positive, and an exact natural SGD commit on the registered GPU.

The pilot compares a single state and minibatch: native/half-scale correction cosines are 0.8444/0.9396, with relative errors 0.6655/0.4720. These values validate logging and local correspondence only; they are not complete mechanism or performance results. Six formal trajectories and the 36-case offline study still require final acceptance.

This reviewer performed read-only inspection and wrote only this report and its JSON receipt.
