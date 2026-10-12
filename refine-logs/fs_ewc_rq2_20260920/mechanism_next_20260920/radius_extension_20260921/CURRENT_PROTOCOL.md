# Current B protocol and historical training provenance

Audited 2026-09-21 using local configurations, historical logs, dataset directories, saved diagnostic manifests, and CPU-only checkpoint metadata reads. No training or GPU computation was performed for this audit.

## Identity of the existing B results

The current B data are `execution/results_seed0_recovery1/{sgd,gam}/neighborhood/`, configured by [diagnostic_config_recovery1.json](../../execution/diagnostic_config_recovery1.json). They use **ImageNet-A**, not ImageNet-R. A directory or figure carrying a legacy ImageNet-R name does not change the checkpoint/data identity.

Both Task0 anchors are the historical `sgd_gam_s1c` models also used by the old LoRA radius/FQA sweep:

```text
/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/
  stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/
    {sgd,gam}/seed0/checkpoints/task_00.pt
```

The original anchors were trained on 2026-06-14. Their checkpoint metadata explicitly report `cur_task=0`, `known_classes=20`, `total_classes=20`, `model_name=ewclora_normfisher_gam_intask`, and the respective optimizer. The diagnostic configuration points to these exact paths (lines 29–44); neighborhood manifests retain their SHA256 hashes.

## Model, data, and task partition

| Item | Exact protocol |
|---|---|
| Backbone | ViT-B/16, `vit_base_patch16_224`, 224-pixel input, 768-dimensional features, 12 transformer blocks |
| Pretrained weights | Historical logs identify `timm/vit_base_patch16_224.augreg2_in21k_ft_in1k` (ImageNet-21k pretraining followed by ImageNet-1k fine tuning). Diagnostic reconstruction sets `pretrained=False` and strictly loads the checkpoint's complete `network_state`; it does not fetch today's alias. |
| Trained adapter | Rank 10 LoRA on Q and V in every block; 24 pairs of A/B, shapes A=(10,768), B=(768,10). Scaling is identity, so effective adapter weight is exactly BA. The backbone is frozen during LoRA training. |
| Classifier | Task0 learned linear head, weight shape (20,768), bias shape (20,). B keeps this head fixed. |
| Dataset | `/data/115-2/users/liying/datasets/imagenet-a/{train,test}`; 200 class directories |
| Partition | 10 tasks × 20 classes, `class_shuffle=false`, identity class order 0…199. ImageFolder indexes sorted synset-directory names. Task t contains indices 20t…20t+19, t=0…9. |
| Task0 data | 1,259 train images and 307 test images; direct directory count agrees with both diagnostic manifests |
| Task0 synsets | `n01498041, n01531178, n01534433, n01558993, n01580077, n01614925, n01616318, n01631663, n01641577, n01669191, n01677366, n01687978, n01694178, n01698640, n01735189, n01770081, n01770393, n01774750, n01784675, n01819313` |

Evidence: [rq2_common.py](../../execution/rq2_common.py), lines 71–115 (rank, strict restoration, head), 204–264 (data); frozen [backbone/lora.py](../../execution/source/backbone/lora.py), lines 482–508 and 573–605 (layers, freezing, Q/V); [data.py](../../execution/source/utils/data.py), lines 719–744 (ImageNet-A loader); original logs described below, lines 6–17, 87–92.

## Original training versus B measurement

| Item | Historical training of anchors / sequence | Current B diagnostic |
|---|---|---|
| Optimizer | Task0 SGD versus GAM with SGD base optimizer | No optimizer or training updates |
| Learning rate / duration | Task0 LR .04, 50 epochs; later tasks LR .02, 50 epochs each; cosine schedule | No LR or epochs |
| Batch / momentum / decay | Batch 128; momentum 0; weight decay 0 | Loss/accuracy batch 64; per-example Fisher processing batch 8 |
| GAM | `lambda_flat=1`, rho .05, norm-rho .2, gamma .1, beta1/2/3=1, nonadaptive | Method label identifies the trained anchor, not a GAM perturbation algorithm |
| EWC | Configured lambda 2000, gamma .9, eta 0; endpoint/task and accumulated Fisher trace normalization enabled; no in-task Fisher, projection, dual ascent, or weight merge | No EWC penalty, lambda, trace normalization, damping/floor, or regularizer in measured CE/Fisher |
| Task0 EWC nuance | No past Fisher exists at Task0: original logs show `has_fisher=False`, EWC=0, D_F=0. Thus lambda2000 in the filename does not mean the first-task fit was constrained by past-task EWC. | Each model is simply the restored Task0 endpoint |
| Input transform | Source training transform: random resized crop to 224, scale .05…1, aspect .75…1.333; horizontal flip .5; tensor; no normalization | Both train and test measurements: RGB, bicubic resize256, center crop224, tensor, no normalization |
| Coordinate | Train A/B factors plus task classifier; 368,640 adapter scalars | Independently perturb all 24 effective Q/V matrices: 24×768×768 = **14,155,776** coordinates; fixed remaining backbone and fixed head |
| Perturbation | Not applicable | BA + r u; u is a unit-Frobenius Gaussian direction in the full effective-Q/V coordinate space. Perturbed matrices need not retain rank10. This is not whole-backbone full-model fine tuning. |
| Fisher | Training uses saved, trace-normalized EWC buffers | Raw per-example CE Fisher, F=(1/N)sum_i g_i g_i^T, g_i=dCE_i/d(effective Q/V). Trace and directional contractions are computed without materializing dense F. |
| Samples / randomness | Training seed0 | Same fixed 256-image proportional, class-stratified train subset for Fisher; subset seed20260920. Loss/accuracy use all 1,259 train and 307 test images. |

Existing radius grid: 0, .5, 1, 2, 5, 10, 20; 16 common directions with direction seed20260920. Radius0 is evaluated once per anchor, giving 1+6×16=97 states per method, **194 total**. These are the source states for the checked194states series. A future extension should inherit this identity and alter only explicitly recorded sampling parameters.

Implementation evidence: [rq2_common.py](../../execution/rq2_common.py), lines 119–136, 166–201, 231–257, 326–353; [diagnostic.py](../../execution/diagnostic.py), lines 317–340; [neighborhood manifest](../../execution/results_seed0_recovery1/sgd/neighborhood/manifest.json), lines 89–120. The manifest records full FP32 production computation with TF32 disabled; a historical TF32 witness is validation-only.

The Task1…Task9 paths in the configuration belong to the 2026-06-22 `stage2_T_compare/Ttraj_fisher_lam2000_lr0p02_e50` sequence. They provide actual drift directions/trajectory endpoints for other panels. That historical comparison is **SGD→SGD+EWC versus GAM→GAM+EWC**. It is not the later matched-future comparison where both anchors continue with SGD. Each checkpoint represents the same continually trained adapter; compute BA_t−BA_0, never sum adapters across task checkpoints. See [AUDIT_TASK_ASSETS.md](../../AUDIT_TASK_ASSETS.md), lines 24–43.

## Relation to older radius experiments and full-model results

The old LoRA `sgd_gam_s1c/stage2_r` / `stage2_r-2` FQA experiments reused the same Task0 anchors, model, rank, dataset, and task split. Their evaluation protocol is not interchangeable with current B: the current old-sweep launcher defaults to 100 directions and radii 0…20 in steps .25, and contains an earlier fine-grained 0…3 grid as a comment. It resumes saved Fisher state and runs evaluation only. Current B explicitly recomputes raw per-example Fisher on the pinned 256 examples and uses 16 shared directions. Exact old plotted values must be tied to their saved logs/partials, not inferred solely from today's launcher. See [run_s1c_stage2_r_fqa_from_task0.sh](/data/140-1/users/liying/CopyFlatCV/scripts/run_s1c_stage2_r_fqa_from_task0.sh), lines 4–5, 18–26, 83–118.

The recoverable historical **full-model** family is separate: `outputs_logs/sgd_gam_s1c_fullmodel/`, actual **ImageNet-R**, 10×20 classes, seed0, ViT-B/16 with the same pretrained tag, all-backbone training, LR .01 for every task, 50 epochs, batch128, cosine, momentum/decay0. Its GAM rho=.02, norm-rho=.2, gamma=.03; later-task EWC lambda2000 and gamma.9. Its recorded training Fisher is `ncm_empirical`, tau16, a batch-gradient-square proxy, with 150 backbone tensors / 85,798,656 coordinates and no FC. Stored Fisher traces are not normalized to one. It also compares SGD→SGD+EWC with GAM→GAM+EWC. These are material dataset, coordinate, learning-rate, GAM, and Fisher differences; it is not a controlled full-model counterpart to current ImageNet-A B. Evidence: [full-model config](/data/140-1/users/liying/CopyFlatCV/config_Expand/sgd_gam_s1c_fullmodel/base_s1c_fullmodel_imagenetr_e50_lr001.yaml), lines 19–75, and the original-log/checkpoint audit [NEW_FULL_RESULTS.md](../../../lora_full_two_task_drift_audit_20260920/NEW_FULL_RESULTS.md), lines 35–47. The latter is prior audit evidence; no new full-model tensor computation was performed here.

## Primary historical log locations and provenance limits

```text
/data/140-1/users/liying/CopyFlatCV/outputs_logs/sgd_gam_s1c/
stage1_task0/task0_shared_lam2000_initlr0p04_lr0p02_e50/trainer_outputs/
logs_inc_lora/ewclora_normfisher_gam_intask/
{sgd,gam}_initlr0.04_lr0.02_e50_task0_shared/imageneta/0/
s1c_{sgd,gam}_task0_shared_e50_lam2000_s0/exp_run/20/
s1c_{sgd,gam}_task0_shared_e50_lam2000_s0_vit_base_patch16_224.log
```

Both original logs directly record dataset/tasks/model (lines 6–17), optimization (18–53), identity class order (87), exact pretrained tag (89–92), actual GAM rho on the GAM branch (94), and zero first-task EWC plus completed 50 epochs (447–448). Current-source training transforms and implementation descriptions are corroborating source evidence, not a claim that every current source file is the byte-identical June execution source. The checkpoint contains full model state but does not itself store a Hugging Face model revision; the historical log identifies the pretrained tag. Diagnostic strict full-state restoration avoids relying on that mutable alias.
