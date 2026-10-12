# 复现命令（仅保留计划，不排队）

先完成并核验全部六组短测，然后运行六组正式训练。将CUDA_VISIBLE_DEVICES设为当时空闲且允许使用的单张GPU；GPU0/1排除。以下命令仅为复现参考，不属于待执行队列；Local-CD已按用户要求设为plan_only。

**smoke_L01_CE_natural_s2**

```bash
CUDA_VISIBLE_DEVICES=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python3.11 /data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/configs/smoke_L01_CE_natural_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/outputs/smoke_L01_CE_natural_s2 --max-tasks 2 --max-epochs 1 --max-batches 1
```

**smoke_L02_Forward_natural_s2**

```bash
CUDA_VISIBLE_DEVICES=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python3.11 /data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/configs/smoke_L02_Forward_natural_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/outputs/smoke_L02_Forward_natural_s2 --max-tasks 2 --max-epochs 1 --max-batches 1
```

**smoke_L03_Backward_natural_s2**

```bash
CUDA_VISIBLE_DEVICES=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python3.11 /data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/configs/smoke_L03_Backward_natural_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/outputs/smoke_L03_Backward_natural_s2 --max-tasks 2 --max-epochs 1 --max-batches 1
```

**smoke_L04_CD_natural_s2**

```bash
CUDA_VISIBLE_DEVICES=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python3.11 /data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/configs/smoke_L04_CD_natural_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/outputs/smoke_L04_CD_natural_s2 --max-tasks 2 --max-epochs 1 --max-batches 1
```

**smoke_L05_Backward_matched_s2**

```bash
CUDA_VISIBLE_DEVICES=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python3.11 /data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/configs/smoke_L05_Backward_matched_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/outputs/smoke_L05_Backward_matched_s2 --max-tasks 2 --max-epochs 1 --max-batches 1
```

**smoke_L06_CD_matched_s2**

```bash
CUDA_VISIBLE_DEVICES=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python3.11 /data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/configs/smoke_L06_CD_matched_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/outputs/smoke_L06_CD_matched_s2 --max-tasks 2 --max-epochs 1 --max-batches 1
```

**L01_CE_natural_s2**

```bash
CUDA_VISIBLE_DEVICES=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python3.11 /data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/configs/L01_CE_natural_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/outputs/L01_CE_natural_s2
```

**L02_Forward_natural_s2**

```bash
CUDA_VISIBLE_DEVICES=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python3.11 /data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/configs/L02_Forward_natural_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/outputs/L02_Forward_natural_s2
```

**L03_Backward_natural_s2**

```bash
CUDA_VISIBLE_DEVICES=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python3.11 /data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/configs/L03_Backward_natural_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/outputs/L03_Backward_natural_s2
```

**L04_CD_natural_s2**

```bash
CUDA_VISIBLE_DEVICES=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python3.11 /data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/configs/L04_CD_natural_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/outputs/L04_CD_natural_s2
```

**L05_Backward_matched_s2**

```bash
CUDA_VISIBLE_DEVICES=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python3.11 /data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/configs/L05_Backward_matched_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/outputs/L05_Backward_matched_s2
```

**L06_CD_matched_s2**

```bash
CUDA_VISIBLE_DEVICES=2 /home-local/liying/.conda/envs/Pilot_new_local/bin/python3.11 /data/140-1/users/liying/O-LoRA/scripts/imageneta_local_cd/train.py --source /data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source --config /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/configs/L06_CD_matched_s2.json --output-dir /data/140-1/users/liying/O-LoRA/refine-logs/local-cd-method-20261003T082009Z/outputs/L06_CD_matched_s2
```
