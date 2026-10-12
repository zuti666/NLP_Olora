# Acceptance correction v2

The actual registered offline smoke is the SGD trajectory at epoch1, as specified by upward_local_audit.py and its CLI dry-run. The initial validator expected epoch0 and rejected a fully passed diagnostic. No training or diagnostic formula, data, budget, parameter update or source changes.

validate_job_v2.py changes only that smoke coordinate (g0,0)->(g0,1). run_queue_v2.py imports this validator and reads RUN_PINS_V2.json. All original frozen files and failed rejection receipt are retained. The completed offline output is independently revalidated and its completion receipt then recorded before queue resume. Existing17 accepted gates remain accepted; all45 formal trainings still depend on the now18 accepted gates.

Resume in the actual host using pinned Python: `python run_queue_v2.py --suite suite.json --dry-run`, then a durable `run_queue_v2.py --suite suite.json` controller. Review receipts are in runtime/ACCEPTANCE_CORRECTION_REVIEW.json; progress remains runtime/queue_state.json.
