# Reused result provenance

The existing seed2, 10-task history-response fusion result was verified on CPU. The audit checked 190 source files, 26 original execution pins, the original completion/exit receipts, 2,550 update records, the accuracy matrix, 10 serialized task checkpoints and 10 native task exports. The original validator checked the cached gradient/displacement tensors.

FAA = 41.297%, AAA = 60.483%, BWT = -10.757 pp.

Reuse is supported by the prior CPU bitwise algorithm and native data/loss/evaluation equivalence checks. It does not claim a new full CUDA trajectory. Every collection rechecks all frozen source/output/checkpoint hashes.
