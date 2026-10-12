# Historical endpoint mismatch: independent investigation

Date: 2026-09-20. Initial failure is retained below; the subsequent controlled precision probe explains it and passes the original tolerance at four historical endpoints. Production measurements still require their independent hard gates.

The first sanity attempt failed its predeclared `3e-5` historical Task0 test CE tolerance: SGD +0.0001832504 and GAM +0.0001788967. SGD test accuracy also changed from 85.3420195% to 85.6677524% (one of 307 examples); this cannot be explained by changing only loss summation precision.

## Findings established locally

- The old evaluator sums per-example cross-entropy over every batch and divides by the total number of examples (`source/src/trainer.py::_evaluate_fixed_task_batches`). The new evaluator does the same, accumulating individual FP32 losses in FP64. This is not a batch-mean versus image-mean discrepancy.
- All 307 Task0 test image files under `/data/115-2/users/liying/datasets/imagenet-a` have identical SHA256 to the corresponding `/data/140-1/datasets/imagenet-a` files. These two data roots are not the source of the mismatch.
- Repository `requirements.txt` records timm 1.0.9, torchvision 0.19.1, torch 2.4.1 and Pillow 11.3.0. The current first three versions match; base Pillow is 12.1.1. A requirements file is not a historical runtime receipt.
- A controlled CPU test of Pillow 11.1.0 + libjpeg9 versus Pillow 12.1.1 + libjpeg-turbo3.1.3 found 283/307 decoded image byte arrays differ. Applying both Pillow versions' bicubic resize and crop to the **same** RGB arrays through lossless PNG gives identical output bytes for all 307 images (`pillow_resize_isolation.json`). Thus the tested pair's differences originate in JPEG decoding, not resize/crop.
- This does **not** identify the historical environment: GPU evaluation of the 11.1/libjpeg9 cache moved losses much farther from history (root's independent GPU probe). Do not use that cache as a fix.
- A local cached wheel for Pillow 11.3.0 was found, ABI cp311 manylinux x86_64, SHA256 `106064daa23a745510dabce1d84f29137a37224831d88eb4ce94bb187b1d7e5f`; see `local_pillow_wheel_search.json`. Root installed it in an isolated target, preserving the base environment.
- The isolated 11.3.0 candidate actually loads its own compiled extension and libjpeg-turbo3.1.1 (`pillow11_3_runtime.json`). Root's `pillow113_gpu.log` reports exactly the same Task0 losses as base Pillow12.1.1. Therefore the Pillow pin has **not** fixed the historical discrepancy and should not be advertised as its cause.

## Controlled precision probe: mismatch explained numerically

Independently inspected `execution/investigate_tf32.py`: it strictly loads a fresh checkpoint, rounds only patch-embedding convolution weights and inputs to a 10-bit mantissa with round-to-nearest-even, and evaluates FP32 convolution accumulation. It does not save modified checkpoints. The rounding expression clears 13 FP32 mantissa bits after the appropriate tie-breaking addition. Matmul and cuDNN TF32 flags are otherwise explicitly false.

Compared its four results directly to the original trajectory CSV (`tf32_independent_comparison.json`):

| Method / task | Historical CE | Emulated-convolution CE | Difference | Accuracy |
|---|---:|---:|---:|---|
| SGD / t0 | .4422401869 | .4422413063 | +1.1194e-6 | exact match |
| SGD / t9 | .7025827072 | .7025834431 | +0.7359e-6 | exact match |
| GAM / t0 | .4134413629 | .4134417125 | +0.3496e-6 | exact match |
| GAM / t9 | .4478546292 | .4478532525 | -1.3767e-6 | exact match |

All four satisfy the originally declared 3e-5 CE tolerance without widening it. This controlled intervention strongly supports a TF32 patch-embedding convolution precision difference as the historical discrepancy source. It does not establish bitwise reconstruction of the historical hardware or complete training. Pillow11.3 pinning is not needed and is not selected for production.

The revised acceptable protocol is: preserve the first failures; use an isolated temporary precision-reference check for legacy reproduction, then remove hooks and restore convolution weights exactly; keep the new raw Fisher, all task endpoints, path losses and A-table baselines uniformly full FP32. The old precision reference must never leak into the scientific gradient estimator. Keep all original current-function hard checks and reevaluate all 20 endpoints under the new explicitly recorded precision protocol.

## Earlier fallback considered before the precision probe

The following conditions were proposed only if no explanation could be established. The successful controlled precision probe above supersedes the unresolved-cause branch while retaining its requirements for coherent new measurements and transparent provenance.

Do not increase the historical tolerance retrospectively and mark the old test PASS. Keep a separate `legacy_reproduction=unresolved/failed` record with old/new CE and accuracy. An unsupported hardware or decoder explanation remains a hypothesis.

New offline checkpoint evaluation can nevertheless be scientifically interpretable if all of the following hold:

1. Keep strict full-state restoration and fixed old-head/backbone checks, source-LoRA versus effective-coordinate forward equivalence, current direct t0/t9 versus path endpoints, independent per-example autograd comparisons, and finite-difference checks as hard gates.
2. Evaluate **all 20 task endpoints** again under the same fixed current environment, deterministic transforms, class mapping and sample list. Use this internally consistent trajectory together with newly measured trace/path sensitivity. Do not combine legacy trajectory CE and new Fisher into a single same-protocol derivation.
3. Keep historical dense radius curves separately labeled as legacy results, without claiming newly verified paired input tensors or directions.
4. Use the current evaluated a11 and two freshly paired Task1 continuations for A; verify new-head, sample-order and augmentation receipts. Describe the nominal historical hyperparameters and the explicit new environment/RNG protocol accurately.
5. Report that numerical historical reproduction remains unresolved even if all current-function measurement gates pass. Preserve the first failed sanity artifacts and all candidate probes.

These conditions preserve an honest distinction between historical reproduction and a valid new measurement on restored tensors.
