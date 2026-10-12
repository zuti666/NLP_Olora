# Two-point reverse probe implementation and CPU validation

Status: implementation complete; 20 CPU tests passed. No GPU training was launched by this implementation task.

Source: `src/optimer_reverse_probe.py`; tests: `tests/test_optimer_reverse_probe.py`.

Command executed:

```bash
env CUDA_VISIBLE_DEVICES= TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
  /data/140-1/users/liying/conda_envs/nlora_req/bin/python tests/test_optimer_reverse_probe.py
```

Observed result: `Ran 20 tests in 0.286s`, `OK`; process exit code 0.

## Implemented directions

- `reverse`: `g0 = grad CE(BASE)`, `g1_minus = grad CE(BASE-rho*g0/(norm(g0)+eps))`; restore BASE and submit `2*g0-g1_minus`.
- `sam`: the same normalized ascent probe; restore BASE and submit `g1_plus`.
- `sgd`: submit the clean `g0` without any probe. This mode name describes the gradient; its Llama commit is still AdamW.

`rho` is independent of commit learning rate. The literal denominator is `norm(g0)+eps`. Probe construction uses the full global-mean pre-clipping gradient. The final submitted direction is clipped at most once, followed by exactly one ordinary base optimizer step. No optimizer moments, scheduler, weight decay, G2/G3, Fisher, EWC, projection, history, outer probe or merge is involved in constructing a direction.

`rho=0` and zero `g0` retain a real second CE phase at BASE, with no fallback, so same-point noise and execution budget are explicit. Ratios/cosines with a zero denominator are JSON null rather than NaN/Inf.

The API matches CurrentG2: `TwoPointProbe(params, base_optimizer, model, mode='reverse', radius=.05, ...)`, then `step(closure, clip_grad_norm=None, first_grad_ready=False)`. The native Trainer supplies effective-batch/per-microbatch RNG replay and globally mean-reduced gradients; standalone closures replay the BASE RNG. Submitted gradients are never re-reduced.

## Test coverage

Twenty test cases exercise independent eps-sensitive quadratic formulas across all three modes and both first-gradient paths; a nonquadratic analytic example showing reverse and SAM differ by a second-order radius term; exact HF AdamW state/parameter equality over four commits; restore-before-commit and single-clip counts; learning-rate-independent probe points; zero radius/gradient; absent-coordinate AdamW policy; union presence for the reverse residual; Python/NumPy/Torch/dropout RNG replay; per-microbatch RNG ownership; failed CE restoration; failed and nonfinite AdamW commit rollback; frozen/trainable mutation guards; resumable checkpoints and cross-mode rejection; actual BF16 CPU autocast with FP32 trainables; diagnostic cadence; diagnostics-on/off bitwise equality; and invalid configuration/pending-checkpoint rejection.

Frozen backbone parameters are immutable by contract and checked through the shared CurrentG2 transaction version guard. Intentional malicious frozen-weight writes are rejected before commit; they cannot be restored without cloning the complete backbone. Trainables, gradients, supported forward buffers/caches/flags and RNG are restored on failure. Only trainable optimizer moments are snapshotted immediately before the unique commit, so a partially failed optimizer commit is also rolled back.

The BF16 weight-cast proxy is observational only and explicitly records `proves_forward_visibility=false`. The BF16 CPU tests validate execution and metadata; they do not establish actual 1B/3B probe response quality.
