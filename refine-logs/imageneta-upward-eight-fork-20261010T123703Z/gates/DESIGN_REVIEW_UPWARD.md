# Independent design review: upward eight-way shared-parent study

Status: **design accepted, conditional on implementation, CPU, native GPU smoke and pairing gates**. This approves the experiment design, not future numerical results.

The study answers three separated questions: which upward-response readout is worth submitting; whether its two-task advantage reflects acquisition, retention or both; and how it trades performance for native gradient-read cost. The existing ten-task, native, restored and strict-native tables remain separate evidence cohorts.

## Registered main comparisons

Let `s=g1-g0` and `d=g3-g2`, where the first probe is upward, the response-selected outer point starts at the original parameter state, and the fourth read starts at that outer point. All virtual reads use the same realized current-task batch and entry RNG state; one natural SGD commit updates the original parameters.

| Direction | Submitted update | Native reads per update | Main interpretation |
|---|---|---:|---|
| g0 | g0 | 1 | SGD acquisition/retention baseline |
| g1 | g1 | 2 | SAM baseline |
| g2_minus_s | g2-(g1-g0) | 3 | Remove the explicit first SAM response |
| g2 | g2 | 3 | Submit the selected outer-point readout |
| g2_plus_s | g2+(g1-g0) | 3 | Add the first response after using it for point selection |
| g0_plus_d | g0+(g3-g2) | 4 | Add the terminal local response to SGD |
| g1_plus_d | g1+(g3-g2) | 4 | Add that response to SAM |
| g3 | g3 | 4 | Direct test of the fourth gradient read at the chosen outer point |

All three seeds use `rho=.05,r=1`, fixed residual coefficients 1, natural SGD LR, rank10 and the complete trainable Q/V LoRA plus full current classifier parameter scope. None of the training branches uses final projection, HVP, Fisher, EWC, replay, fusion, history, clipping or final norm matching. The pair `g2_minus_s/g2/g2_plus_s` isolates the submitted algebraic role of the first SAM response under the same three-read rule; `g2/g3` isolates the terminal probe. Full-training trajectories diverge, so “same residual” means the same rule; the offline common-state study supplies literally shared numerical vectors.

## Parent and SGD reuse

The accepted seed2 SGD Task0 parent is reusable without retraining. Its full-model, complete-state, native checkpoint, accepted summary, RNG and fixed dev-split provenance were independently revalidated in `REUSE_AUDIT.json`; all old 57 execution pins and 190 source members were unchanged.

The old seed2 Task1 SGD B0 endpoint is valid, but it has no 50-epoch test retention curve. Its complete snapshots only cover epochs0/1/10/25/49. Accordingly the new study retrains only Task1 SGD from the same accepted seed2 parent, obtains the full curve, and uses the old B0 as a read-only equivalence oracle: all400 input/target/index/LR identities and the terminal parameter hash must agree. It is not counted as an additional independent baseline or copied into a fabricated RetentionAUC curve.

Seeds0/1 each train their own SGD Task0 parent, then fork eight Task1 methods. Within every seed, the FC20->40 **post-expansion** complete parameter identity, inherited old rows, new classifier rows, first g0 and realized training sequence must agree. The original Task0 hashes alone are insufficient.

The fixed train-only dev split remains unchanged: Task0 train1141/dev118; Task1 train906/dev91. Training and dev CE use the current20 logit slice; main test evaluation uses all-seen40. Fixed endpoint Task1 devCE is the sole selection convention. No old/test retention outcome chooses epochs, radii or methods.

## Counts and staged execution

- 24 new `r=1` Task1 forks, eight methods by seeds2/0/1.
- One extra upward `g2,r=.25` seed2 bridge.
- Two new Task0 parents for seeds0/1; seed2 parent is reused.
- 18 full200 single-task trainings, six methods by three seeds.
- Total45 new formal training jobs, plus three offline common-state jobs and bounded smoke/pairing checks.

Start with native CPU checks and seed2 parent-bound GPU smoke/pairing checks. Only accepted successful dependencies release formal work. Use idle physical GPUs2–6 under index+UUID locks, never GPUs0/1, no process preemption. Failed jobs must stop new admission rather than silently skipping an arm or continuing with an invalid shared parent.

The complete snapshots must retain the full model, gradients, buffers/caches, modes, frozen provenance and RNG, rather than silently shrinking to adapter-only files. Nine complete snapshots for each of25 Task1 forks require approximately148GB at the measured old snapshot size. The local `/data/140-1` filesystem is insufficient for this budget. A dedicated output root under the explicitly writable `/data/115-2/users/liying/cache/tmp/imageneta-upward-eight-fork-20261010T123703Z/outputs` is appropriate, with `NEW/outputs` as an access symlink. All actual outputs must resolve beneath that exact exclusive run directory, not merely under the broad cache/tmp prefix. The queue must reject nonempty output reuse and keep logs/configs/source/pins in NEW; old parent/checkpoints remain read-only and are never migrated or overwritten. The storage change affects location only, not the complete-state scope or training protocol.

The `.25` bridge compares upward `g2^A(.05,.25)` with existing accepted descending `g2^D(.05,.25)` (R25) and natural-inner descending `g2^D(h,.25)` (S_F). At `r=1`, compare upward to descending R100. Upward r1 versus descending r.25 confounds the side and the radius and is not the bridge contrast.

## Curves, local diagnosis and cost

The new forks record current-devCE, old-allseen40, new-allseen40, old20-only accuracy and complete shared-state snapshots. Report pre-expansion old accuracy, immediately post-expansion old accuracy, final old accuracy, BWT relative to the first, and training-only change relative to the second. Two-task stage AAA is `(parent_accuracy+FAA)/2`; it is distinct from the50-epoch RetentionAUC.

Matching current-task learning must use predeclared current train/dev quantities in the common covered range, without looking at old-test outcomes to select checkpoints. Compare the old curve at those learning levels. State whether each algorithm learns more, retains more at matched learning, or trades one for the other; a final FAA score cannot answer all three.

Common-state offline diagnosis samples SGD/SAM/g2 trajectories at registered early/middle/late epochs. It constructs the same eight vectors from the same four reads, evaluates raw and `eta*||g0||` nominal-parameter-length matched virtual updates, and restores the full model/RNG after every candidate. Report achieved FP32 displacement and matching discrepancy; nominal matching is not guaranteed exact actual displacement matching. Old-task images are used for offline losses only, with zero old-data training gradients, no model selection and zero committed training updates.

Native methods genuinely skip unused probes. Their endpoint learning curves use equal samples and true update counts, not equal gradient reads. Native timing must separate compute+SGD from passive per-batch CE, per-epoch dev/test evaluation, snapshot serialization and controller overhead. A three-read method has25% fewer nominal training gradient reads than a four-read method; do not replace observed speed with that arithmetic. Costs of mandatory state restoration/hash checks must be described consistently with what the timer actually includes. No formal noninferiority or algorithm equivalence claim is pre-authorized by a similar point estimate.

## Full200 single-task scope

The single-task study uses all5981 original training images and1519 original test images together in a200-class head, 50 epochs, rank10 and LR.04->.004; radii are locked from the prior development work. It is not the20-class parent task and not500epochs. Its six methods are g0/g1/g2_minus_s/g2/g1_plus_d/g3; each seed has the same initial full200 model and identical data/LR sequence across these methods.

Report fixed epoch50 sample accuracy, equal-weight accuracy over the original ten20-class groups, CE and native time. BWT is null because there is no task switch. Unlike the two-task holdout study, its training membership includes all5981 images; this difference is explicit. It constrains general optimization/generalization explanations, but cannot itself be a replay-free continual-learning baseline or prove that a two-task difference is caused solely by forgetting.

## Permissible claims and limitations

The three-seed paired main table can support replicated two-task algorithm effects. Shared-state length-matched virtual steps constrain amplitude explanations locally, while independent training measures eventual performance. The full200 table distinguishes general optimization benefits from possible sequence dependence. None alone proves a Hessian/H-squared mechanism, globally smooth update path, universal radius optimum, strict equivalence or noninferiority to a four-read method, or general rank robustness. The primary competition includes `g2_minus_s`, which had a slightly higher historic FAA mean, rather than only weak SGD/SAM controls.
