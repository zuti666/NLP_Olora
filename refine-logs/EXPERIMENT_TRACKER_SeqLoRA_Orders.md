# Experiment Tracker: SeqLoRA Multi-Order

| Run ID | Milestone | Purpose | Method | Order | Task Sequence | GPU | Priority | Status | Notes |
|--------|-----------|---------|--------|-------|---------------|-----|----------|--------|-------|
| R000 | M0 | Fix GPU conflict in run_all_seqlora.sh | — | All | — | — | MUST | TODO | 修改 Base 脚本传参 0,1,2,3 |
| R001 | M1 | SeqLoRA-SAM training | SAM | Order1 | dbpedia→amazon→yahoo→agnews | 4,5,6,7 | MUST | TODO | |
| R002 | M1 | SeqLoRA-GAM training | GAM | Order1 | dbpedia→amazon→yahoo→agnews | 0,1,2,3 | MUST | TODO | 并行于 R001 |
| R003 | M1 | SeqLoRA-RWP training | RWP | Order1 | dbpedia→amazon→yahoo→agnews | 4,5,6,7 | MUST | TODO | 修复后并行于 R004 |
| R004 | M1 | SeqLoRA-Base training | Base | Order1 | dbpedia→amazon→yahoo→agnews | 0,1,2,3 | MUST | TODO | 修复后并行于 R003 |
| R005 | M2 | SAM+GAM eval | SAM,GAM | Order1 | — | 0,1,2,3 | MUST | TODO | 并行于 R006 |
| R006 | M2 | Base+RWP eval | Base,RWP | Order1 | — | 4,5,6,7 | MUST | TODO | 并行于 R005 |
| R007 | M3 | SeqLoRA-SAM training | SAM | Order2 | dbpedia→amazon→agnews→yahoo | 4,5,6,7 | MUST | TODO | |
| R008 | M3 | SeqLoRA-GAM training | GAM | Order2 | dbpedia→amazon→agnews→yahoo | 0,1,2,3 | MUST | TODO | 并行于 R007 |
| R009 | M3 | SeqLoRA-RWP training | RWP | Order2 | dbpedia→amazon→agnews→yahoo | 4,5,6,7 | MUST | TODO | 修复后并行于 R010 |
| R010 | M3 | SeqLoRA-Base training | Base | Order2 | dbpedia→amazon→agnews→yahoo | 0,1,2,3 | MUST | TODO | 修复后并行于 R009 |
| R011 | M4 | SAM+GAM eval | SAM,GAM | Order2 | — | 0,1,2,3 | MUST | TODO | 并行于 R012 |
| R012 | M4 | Base+RWP eval | Base,RWP | Order2 | — | 4,5,6,7 | MUST | TODO | 并行于 R011 |
| R013 | M4 | SeqLoRA-SAM training | SAM | Order3 | yahoo→amazon→agnews→dbpedia | 4,5,6,7 | MUST | TODO | |
| R014 | M4 | SeqLoRA-GAM training | GAM | Order3 | yahoo→amazon→agnews→dbpedia | 0,1,2,3 | MUST | TODO | 并行于 R013 |
| R015 | M4 | SeqLoRA-RWP training | RWP | Order3 | yahoo→amazon→agnews→dbpedia | 4,5,6,7 | MUST | TODO | 修复后并行于 R016 |
| R016 | M4 | SeqLoRA-Base training | Base | Order3 | yahoo→amazon→agnews→dbpedia | 0,1,2,3 | MUST | TODO | 修复后并行于 R015 |
| R017 | M5 | SAM+GAM eval | SAM,GAM | Order3 | — | 0,1,2,3 | MUST | TODO | 并行于 R018 |
| R018 | M5 | Base+RWP eval | Base,RWP | Order3 | — | 4,5,6,7 | MUST | TODO | 并行于 R017 |
