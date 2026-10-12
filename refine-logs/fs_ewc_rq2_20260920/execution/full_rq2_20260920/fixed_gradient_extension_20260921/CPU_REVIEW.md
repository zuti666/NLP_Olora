# CPU implementation review before GPU execution

The final user intent is Task0 positive-gradient ascent. Each anchor uses u = +grad(mean Task0 CE) / norm(grad), computed once and fixed for every radius. No negative-gradient run is authorized by the final correction.

Worker and publisher were reviewed by the existing audit_radius50_protocol agent, independently from their implementers, with conversation context. This is not a fresh-context review.

- Six CLI preflights passed for seeds 0/1/2 and SGD/GAM.
- Worker check used 19 samples (8+8+3); independent reviewer used 27 samples (8+8+8+3). Both compare weighted batch accumulation with full-batch autograd.
- Reviewer observed positive gradient-direction inner product and positive small-step CE change on its CPU fixture. An actual sign-flipped unit vector was rejected.
- Publisher independently verifies saved full gradients, unit directions, sample membership, finite-difference witness, and raw measurement consistency.
- The real models have not yet passed these checks: each GPU job must obtain its own witness before scanning. Large-radius monotonicity is not required.
- Frozen worker/config/CPU-check hashes and publisher hash were rechecked before launch.

All data and figures will be generated from the six new jobs. No CPU fixture values are experimental results.
