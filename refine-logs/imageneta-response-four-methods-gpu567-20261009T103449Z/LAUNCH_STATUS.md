# Formal validation launch status

The user requested physical GPU5/6/7 only. This host exposes GPU0–6, so GPU7 is absent. Two accepted full-budget jobs are running on physicalGPU5 (SGD) and GPU6 (Current-G2). Each already has a real seeded CUDA witness and at least ten finite actual SGD updates. The classifier spans all200 ImageNet-A classes. No new complete performance result is available yet.

Five new jobs are pending. The two remaining single200 methods (History Fusion, History-G2) run as those two cards become free; all four single200 experiments must finish and pass full-budget acceptance before three new CL10 experiments start. CL10 History Fusion reuses the independently audited existing result. Seven new formal trainings total, seed2/rank10,50epochs/task,batch128,SGD. Source/trainer are the previously accepted implementation; old GAM fixed radii are inactive, response radius is dynamic LR×norm(g0).

No EWC/Fisher, actual weight fusion, projection, HVP, QR, clipping, replay or final update normalization is performed. Shared physical-index/UUID leases restrict GPU admission, and every completed job is validated before automatic continuation. GPU0–4 are not used by this campaign.

Live state: [queue_state.json](runtime/queue_state.json). Accepted results: [RESULTS.md](RESULTS.md), [RESULTS.csv](RESULTS.csv). Settings: [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md), [RUN.md](RUN.md). Launch proof: [LAUNCH_VERIFIED.json](LAUNCH_VERIFIED.json).
