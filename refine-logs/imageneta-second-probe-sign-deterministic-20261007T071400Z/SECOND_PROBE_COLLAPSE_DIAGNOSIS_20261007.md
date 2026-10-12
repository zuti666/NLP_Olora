The r=1 collapse begins inside Task0 after initial learning; later-task forgetting cannot explain its onset.
Epoch rows are online local training accuracy, not epoch-end test evaluations. Task0 end test is 85.02–85.67% in B0/B1/N5/N6 versus 6.51–7.17% in N1–N4.

| Cell | Best train before <50% | First train <50% | First train <10% | Max Task0 CE | Max norm g0 | Max norm U | Negative cos(U,g0), Task0/all | Max actual Task0 effective step |
|---|---|---|---|---:|---:|---:|---:|---:|
| B0 | e48: 89.36% / CE 0.359 | — | — | 3.564 | 4.286 | 5.510 | 0/0 | 0.172 |
| B1 | e48: 89.36% / CE 0.357 | — | — | 3.564 | 4.286 | 4.439 | 0/0 | 0.176 |
| N1 | e7: 78.08% / CE 0.698 | e12: 34.71% / CE 6.123 | e13: 4.05% / CE 54.151 | 149.732 | 193.576 | 189.777 | 1/1 | 18.201 |
| N2 | e5: 77.05% / CE 0.781 | e9: 49.17% / CE 2.807 | e10: 6.12% / CE 25.408 | 99.033 | 1043.918 | 334.792 | 2/18 | 22.364 |
| N3 | e5: 76.17% / CE 0.795 | e11: 29.07% / CE 13.188 | e12: 4.77% / CE 29.526 | 81.869 | 188.066 | 200.338 | 2/22 | 18.928 |
| N4 | e7: 78.87% / CE 0.680 | e10: 5.24% / CE 45.138 | e10: 5.24% / CE 45.138 | 219.598 | 355.735 | 257.210 | 10/36 | 23.730 |
| N5 | e48: 89.12% / CE 0.361 | — | — | 3.564 | 4.286 | 4.344 | 0/0 | 0.172 |
| N6 | e48: 89.36% / CE 0.359 | — | — | 3.564 | 4.286 | 4.906 | 0/0 | 0.176 |

All eight share exactly the same 2550 batch identities and learning rates, all 500 sampler orders, and identical Task0 entry g0/g1/g2 fingerprints. Each run uses 10200 CE training gradients and 2550 SGD updates; no Fisher, merge, projection, HVP, or create_graph calls.
The first ||U||>10 happens at Task0 e8/b9 in N1, e7/b6 in N2, e6/b9 in N3, and e8/b6 in N4 (batch indexes zero-based). Four successful cells never exceed ||U||=5.511 during Task0.
The loss and update remain finite. Logs contain only expected NaN placeholders for unseen future-task score entries, with no OOM/CUDA/Traceback/Nonfinite error lines; trace and epoch artifacts contain no nonfinite numbers.
Code and every-step scalar audits support phi3=phi2+s*rho*normalize(g2), U=anchor+c*(g3-g2), actual SGD=phi0-lr*U; no global CE-gradient reversal bug is evident. Negative cos(U,g0) counts are sparse, often after collapse: 1/500,2/500,2/500,10/500 for N1–N4 versus 0/500 in all four successful cells.
Natural update amplitude is a major confound: max Task0 effective joint step is 18.201–23.730 in failed cells versus 0.172–0.176 in successful cells. This supports observed optimization instability, not a demonstrated Hessian/anti-flatness causal explanation.

Detailed artifact: /tmp/second_probe_collapse_diagnosis_20261007.json
