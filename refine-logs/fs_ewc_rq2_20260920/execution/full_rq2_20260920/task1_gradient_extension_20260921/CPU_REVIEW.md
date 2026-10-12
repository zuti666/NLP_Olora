# Pre-launch Task1 implementation review

Final protocol: one fixed Task1 negative mean-CE gradient at each own Task0 anchor; 997 new-task training examples. Evaluate old Task0 loss/accuracy and raw Fisher with the original old20 classifier.

Worker CPU checks pass: six CLI configurations, exact initial-new-head reconstruction matching formal receipts, full997 membership, weighted batch gradients including the last5 examples, actual-vector sign-flip rejection, and nonfinite/scope rejection. Worker source/config/check receipt hashes were independently checked before launch.

Existing audit_radius50_protocol agent independently checked source code and a separate997-example dual-head CPU fixture: batched-vs-full gradient maximum discrepancy 3.95e-8; g1 dot u = -0.086088745; h=1e-4 decreases Task1 CE by8.82e-6; a flipped actual direction is rejected; both heads unchanged. This reviewer had project context and is independent from the implementers, not a fresh-context certification.

Publisher independent validation checks saved negative gradient direction, exact untrained new20 head and Task1 membership, Task1 finite-difference descent, unchanged old20 head, all93points and raw-to-summary consistency. No old-loss positivity or monotonicity condition is imposed. File-content hashes and canonical-data hashes are distinct.

These are CPU implementation checks; every real GPU job must still pass its own seeded environment witness and Task1 small-step descent witness. CPU fixture numbers are never reported as experimental results.
