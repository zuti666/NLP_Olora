# ImageNet-A single200 LoRA / full-FT results

Queue: completed. Updated UTC: 2026-10-11T18:56:08.016619+00:00.

Fixed epoch50 endpoint on all 1519 test images. Accuracies are sample percentages. Only accepted formal runs appear; smoke measurements are excluded.

| Scope | Method | Status | Test accuracy | Test CE | Updates | Native CE gradient reads/step |
|---|---|---|---:|---:|---:|---:|
| lora | sgd | completed_reused | 63.002 | 1.796 | 2350/2350 | 1 |
| lora | sam | completed_reused | 64.187 | 1.645 | 2350/2350 | 2 |
| lora | mirror | completed | 64.055 | 1.678 | 2350/2350 | 2 |
| lora | g2 | completed_reused | 68.598 | 1.378 | 2350/2350 | 3 |
| lora | gam | completed | 67.610 | 1.388 | 2350/2350 | 4 |
| full_ft | sgd | completed | 53.325 | 2.116 | 2350/2350 | 1 |
| full_ft | sam | completed | 3.687 | 4.958 | 2350/2350 | 2 |
| full_ft | mirror | completed | 1.843 | 5.031 | 2350/2350 | 2 |
| full_ft | g2 | completed | 1.382 | 5.055 | 2350/2350 | 3 |
| full_ft | gam | completed | 24.161 | 3.833 | 2350/2350 | 4 |

Within each scope all methods share the initialized ViT/classifier, data stream, 50 epochs, batch128 and cosine SGD. LoRA LR=.04→.004; full-FT LR=.001→.0001. rho=.05 and outer r=1. GAM alone retains its intrinsic gamma=.1 projection. No EWC/Fisher, weight merge, replay or final norm matching.

g2 comparisons (percentage points; full-FT remains pending until accepted):
- lora: g2 minus sgd = +5.596 percentage points.
- lora: g2 minus sam = +4.411 percentage points.
- lora: g2 minus mirror = +4.542 percentage points.
- lora: g2 minus gam = +0.987 percentage points.
- full_ft: g2 minus sgd = -51.942 percentage points.
- full_ft: g2 minus sam = -2.304 percentage points.
- full_ft: g2 minus mirror = -0.461 percentage points.
- full_ft: g2 minus gam = -22.778 percentage points.

Single seed: these comparisons assess this configuration and do not establish multi-seed statistical significance. Different native gradient-read costs are recorded, not equalized.
