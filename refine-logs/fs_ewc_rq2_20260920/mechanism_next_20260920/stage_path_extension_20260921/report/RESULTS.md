# ImageNet-A seed0 stage chord results

Status: **complete**; validated 18/18 paths (16/16 new, 2/2 legacy).

Each target uses its own straight effective-weight chord from the method's Task0 anchor. Fisher uses the same fixed 256 Task0 training examples and fixed 20-class head. S_path = I/r_T²; T1 has zero displacement and undefined S_path, so it is omitted. The first three panels retain the original 20 measured checkpoint rows. No missing stage is interpolated or imputed.

Numerical validation recomputes Simpson quadrature, S_path, the same-subset loss bound, and endpoint recovery against the original checkpoint trajectory. Full train/test damage is reported separately and is not substituted into the subset bound. Legacy T10 gates are derived from their original quadrature/bound receipts; endpoint recovery is recomputed from their measured nodes. Checkpoint identity is checked against the existing hashes; this CPU collector does not rehash the large weight files.

| Method | T | r_T | I | S_path | Nodes |
|---|---:|---:|---:|---:|---:|
| EWCLoRA | 2 | 0.3678701 | 0.020528728 | 0.15169562 | 9 |
| EWCLoRA | 3 | 0.63015948 | 0.070191938 | 0.17676093 | 9 |
| EWCLoRA | 4 | 0.74967399 | 0.08102609 | 0.14417169 | 9 |
| EWCLoRA | 5 | 0.80485388 | 0.06887041 | 0.10631599 | 9 |
| EWCLoRA | 6 | 0.90567011 | 0.1498487 | 0.18268921 | 9 |
| EWCLoRA | 7 | 0.97924762 | 0.26880773 | 0.28032169 | 9 |
| EWCLoRA | 8 | 1.0242543 | 0.20057191 | 0.19118532 | 9 |
| EWCLoRA | 9 | 1.1248863 | 0.2651168 | 0.20951734 | 9 |
| EWCLoRA | 10 | 1.1975841 | 0.52787104 | 0.3680576 | 9 |
| FS-EWCLoRA | 2 | 0.50377837 | 0.0053052418 | 0.020903844 | 9 |
| FS-EWCLoRA | 3 | 0.81742693 | 0.0059149962 | 0.0088523095 | 9 |
| FS-EWCLoRA | 4 | 0.96251886 | 0.0088440613 | 0.0095462597 | 9 |
| FS-EWCLoRA | 5 | 0.9936438 | 0.022988236 | 0.023283281 | 9 |
| FS-EWCLoRA | 6 | 1.2159859 | 0.066820713 | 0.045191214 | 9 |
| FS-EWCLoRA | 7 | 1.2242749 | 0.067200713 | 0.04483488 | 9 |
| FS-EWCLoRA | 8 | 1.2579763 | 0.063792331 | 0.040310998 | 9 |
| FS-EWCLoRA | 9 | 1.3517437 | 0.046388584 | 0.025387644 | 9 |
| FS-EWCLoRA | 10 | 1.4004726 | 0.02901404 | 0.014793093 | 9 |

One seed per method; these curves do not estimate between-seed uncertainty.
