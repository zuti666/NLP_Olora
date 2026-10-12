# IncLoRA/SeqLoRA SAM+EWC Code Review

Date: 2026-08-29

Scope:

- `inclora_sam_ewc`: CE-only SAM probes plus clean-point EWC on a fresh LoRA branch.
- `seqlora_sam_ewc`: CE-only SAM probes plus clean-point online EWC on a fixed-rank shared LoRA branch.
- Llama-3.2-1B launchers for Orders 1, 2, and 3.

## Fresh review result

Final verdict: **APPROVE**.

The first review requested changes for:

1. Multi-rank adapter/tokenizer write races.
2. New launchers being hidden by `.gitignore` rules.
3. Partial or mis-shaped Fisher state being silently accepted.
4. Missing negative-path and chain-persistence tests.

Implemented corrections:

- Only global rank zero writes adapter, tokenizer, and sidecar; all ranks synchronize afterward.
- Added explicit `.gitignore` exceptions for the shared launcher, six order wrappers, run-all entry, and verifier.
- Public SAM+EWC strategies require exact Fisher module coverage and `[out, in]` shapes; SeqLoRA additionally requires matching compact references.
- A non-empty Fisher path must exist and contain non-empty `omega`; only an empty path denotes task one.
- The post-stage verifier enforces IncLoRA concat invariants and SeqLoRA fixed-rank/reference invariants.

The final fresh-agent re-review found no remaining blocking issues.

## Verification

- Python compilation: pass.
- Shell syntax and six launcher dry-runs: pass.
- Unit tests: 28/28 pass.
- Tiny PEFT fixed-rank save/load round trip: exact match.
- Two-task T5-small CPU smoke, IncLoRA-SAM-EWC: pass; rank sum 2 to 4 and historical prefix exact.
- Two-task T5-small CPU smoke, SeqLoRA-SAM-EWC: pass; rank sum remains 2 and fixed shapes verified.
- In both two-task smokes, task-two step one starts at zero EWC loss and step two reports non-zero EWC loss and regularization-gradient norm.

The available GPUs were all occupied during this implementation review, so no new Llama-1B GPU training was started.
