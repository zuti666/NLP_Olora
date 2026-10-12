#!/usr/bin/env python3
"""Run a 14-epoch budget along the original 50-epoch cosine trajectory."""
import task0_trainer as trainer

original_resolve = trainer.resolve_config


def resolve_validation_config(raw, cli):
    if cli.max_epochs != 14 or cli.max_batches:
        raise ValueError("This entrypoint requires --max-epochs 14 and all training batches")
    cfg = original_resolve(raw, cli)
    if cfg["stage1_direction"] != "true_sam":
        raise ValueError("Recovery preflight is only for True SAM")
    cfg["init_epoch"] = 14
    cfg["epochs"] = 50  # native scheduler uses epochs as T_max
    cfg["recovery_validation_budget_epochs"] = 14
    cfg["recovery_preserved_scheduler_horizon"] = 50
    return cfg


if __name__ == "__main__":
    trainer.resolve_config = resolve_validation_config
    trainer.main()
