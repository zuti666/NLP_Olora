# Independent g0+B code review

Result: passed. No executable formula, schema, scheduling, or historical-boundary defect was found that blocks the four authorized GPU smokes. The two formal P/g0B trainings remain behind validation of all four smokes and the pairing gate.

- P uses the positive first probe; g2 is evaluated from the original base at phi+r*N(g1plus-g0). The update is g0+g2-alpha*g1plus with a single global FP64 dot reduction over every trainable LoRA factor and seen FC parameter; denominator epsilon is 1e-12.
- C/A uses the negative first probe, q=g0-g1minus, and U=g0+g2-g1minus.
- The executable step path performs three CE forwards/autograd calls with create_graph=False and one ordinary zero-momentum/zero-decay SGD update. It restores entry state, replays R0 and advances once to g0's R1. g3 and the opposite first probe do not run.
- Native current-task loss, trainable coverage, effective radius, state/cache restoration, disabled Fisher/merge guards, and nominal SGD scale agree with the requested protocol.
- Independent CPU rerun: 7 acceptance tests passed. An additional fixture generated actual Controller step records for both radii and both directions through two expanded-head tasks: validate_job accepted all four. The real pairing script also accepted the fixture.
- Pairing checks realized inputs, targets, sampler indices and Task0 initial parameters. It does not require equal Task1 model weights, evolving gradients, trajectories, or cross-GPU bitwise behavior.
- The suite contains exactly four smokes, one acceptance job and two formal P/g0B jobs. Formal dependencies are behind the pairing gate. GPU0/1 are excluded; admission uses idle memory/utilization checks and inherited shared index/UUID locks.
- All 30 pinned execution files and the source manifest were verified. The four natural historical AP/AC endpoints were independently checked against receipts, raw trace/matrix hashes and recomputed task-macro metrics.

Historical AC remains a four-probe control with a different gradient budget; the new AC smoke implements the same mathematical update with three probes. P/g0B versus C/AC changes the first side and outer point as well as projection, so it is not a projection-only mechanism test. ImageNet-R g0B and sample-weighted metrics remain distinct from ImageNet-A task-macro results.

This review launches no GPU work and claims no ImageNet-A runtime result. The CPU schema fixture uses a toy model and a summary envelope matching the inspected writer. Actual CUDA/data/model/lifecycle validation belongs to the queued GPU smokes.

Evidence: [JSON review](INDEPENDENT_CODE_REVIEW.json), [CPU schema integration](INDEPENDENT_SCHEMA_CPU.json).
