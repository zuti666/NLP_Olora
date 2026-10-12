"""TreeLoRA adapter for the existing UIE data, training and generation pipeline.

This is the upstream parameter-trajectory regularizer, applied before the
common gradient-accumulation division. Its fixed-rank adapter is continued
across tasks; the historical UIE optimizer/LR schedule restarts each task.
"""
import hashlib
import json
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import TrainerCallback

from treelora import TreeLoRARegularizer
from uie_trainer_lora import UIETrainer

UPSTREAM_COMMIT = "1c7260c42b34e1961283797c742f08b9c3842501"


def _hash_parameters(model):
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if "loranew_" in name:
            digest.update(name.encode())
            digest.update(parameter.detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class _TreeEpochCallback(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer

    def on_epoch_begin(self, args, state, control, train_dataloader=None, **kwargs):
        self.trainer.tree.begin_epoch(len(train_dataloader))


class TreeLoRATrainer(UIETrainer):
    def __init__(self, *args, treelora_adapter_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        if self.args.do_train and (self.args.train_method != "lora" or self.args.optimizer_type.lower() != "adamw_hf"):
            raise ValueError("TreeLoRA currently requires train_method=lora and optimizer_type=adamw_hf")
        if self.args.treelora_reg <= 0:
            raise ValueError("TreeLoRA requires a positive treelora_reg to build task history")
        if self.args.resume_from_checkpoint:
            raise ValueError("TreeLoRA supports completed-task adapter continuation, not mid-task resume")
        config = self.model.peft_config["default"]
        if not config.save_loranew or config.r_sum != 0 or getattr(config, "save_lora_only", False):
            raise ValueError("TreeLoRA requires save_loranew=True, r_sum=0 (fixed-rank adapter)")
        self.tree = TreeLoRARegularizer(reg=self.args.treelora_reg, num_tasks=4)
        self.a_parameters = [(n, p) for n, p in self.model.named_parameters() if "loranew_A" in n]
        if not self.a_parameters or len({tuple(p.shape) for _, p in self.a_parameters}) != 1:
            raise ValueError("TreeLoRA needs equally shaped LoRA A matrices, as in Llama q_proj/v_proj")
        self.parameter_names = [name for name, _ in self.a_parameters]
        if any(p.requires_grad and "loranew_" not in n for n, p in self.model.named_parameters()):
            raise ValueError("TreeLoRA must train only the fixed-rank loranew A/B branch")
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.task_id = 0
        if treelora_adapter_path:
            state_path = Path(treelora_adapter_path) / "treelora_state.pt"
            if not state_path.is_file():
                raise ValueError(f"Missing TreeLoRA history: {state_path}")
            state = torch.load(state_path, map_location="cpu")
            if (state["method"] != "TreeLoRA" or state["schema_version"] != 1
                    or state["upstream_commit"] != UPSTREAM_COMMIT):
                raise ValueError("Incompatible TreeLoRA sidecar")
            if (self.args.do_train and state["world_size"] != self.world_size) or state["parameter_names"] != self.parameter_names:
                raise ValueError("TreeLoRA continuation requires matching ranks and A parameter layout")
            self.task_id = state["completed_tasks"]
            maximum_tasks = 3 if self.args.do_train else 4
            if not 1 <= self.task_id <= maximum_tasks:
                raise ValueError("TreeLoRA task count is incompatible with training/prediction mode")
            # Pure generation does not use the tree; permit another inference
            # world size and the final four-task checkpoint.
            self.tree.load_state_dict(state["rank_states"][self.rank if self.args.do_train else 0])
            if self.tree.task_id != self.task_id:
                raise ValueError("TreeLoRA sidecar task count and trajectory history differ")
        self.initial_hash = _hash_parameters(self.model)
        if treelora_adapter_path and self.initial_hash != state["adapter_parameter_sha256"]:
            raise ValueError("TreeLoRA adapter parameters do not match the saved task history")
        self.train_microbatches = 0
        self.max_reg_gradient_norm = 0.0
        self.max_reg_loss_abs = 0.0
        self.observed_forward_dtype = None
        self._finished = False
        self._saved_state = None
        # Record actual AMP dispatch, not only the requested TrainingArguments.
        module = next(m for n, m in self.model.named_modules() if n.endswith("loranew_A.default"))
        def observe_dtype(module, inputs, output):
            if module.training and torch.is_grad_enabled():
                self.observed_forward_dtype = str(output.dtype)
        self._dtype_hook = module.register_forward_hook(observe_dtype)
        self.add_callback(_TreeEpochCallback(self))

    def train(self, resume_from_checkpoint=None, *args, **kwargs):
        if resume_from_checkpoint is not None and resume_from_checkpoint is not False:
            raise ValueError("TreeLoRA does not support automatic or explicit mid-task checkpoint resume")
        return super().train(resume_from_checkpoint=resume_from_checkpoint, *args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        result = super().compute_loss(model, inputs, return_outputs=return_outputs)
        task_loss = result[0] if return_outputs else result
        if model.training and torch.is_grad_enabled():
            current = torch.stack([p.reshape(-1) for _, p in self.a_parameters])
            loss = self.tree.loss(current, task_loss)
            if not torch.isfinite(loss.detach()).all():
                raise FloatingPointError("TreeLoRA produced a nonfinite training loss")
            self.train_microbatches += 1
            self.max_reg_loss_abs = max(self.max_reg_loss_abs, float((loss-task_loss).detach().abs()))
            if self.task_id and self.max_reg_gradient_norm == 0 and self.train_microbatches > 1:
                gradient = torch.autograd.grad(loss-task_loss, current, retain_graph=True)[0]
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("Nonfinite TreeLoRA regularizer gradient")
                self.max_reg_gradient_norm = float(gradient.float().norm())
            return (loss, result[1]) if return_outputs else loss
        return result

    def finish_treelora_task(self):
        if self._finished:
            raise RuntimeError("TreeLoRA task was already finalized")
        if self.train_microbatches == 0:
            raise RuntimeError("No TreeLoRA training microbatches observed")
        self.tree.end_task()
        if self.tree.task_id != self.task_id + 1:
            raise RuntimeError("TreeLoRA failed to record the completed task trajectory")
        self._dtype_hook.remove()
        finite = all(torch.isfinite(p).all().item() for n, p in self.model.named_parameters() if "loranew_" in n)
        final_hash = _hash_parameters(self.model)
        hashes = [None] * self.world_size
        local_state = self.tree.state_dict()
        states = [None] * self.world_size
        if dist.is_initialized():
            dist.all_gather_object(hashes, final_hash)
            dist.all_gather_object(states, local_state)
        else:
            hashes, states = [final_hash], [local_state]
        audit = dict(method="TreeLoRA", upstream_commit=UPSTREAM_COMMIT,
                     task_id=self.task_id, completed_tasks=self.task_id+1,
                     rank=self.rank, world_size=self.world_size,
                     train_microbatches=self.train_microbatches, optimizer_steps=self.state.global_step,
                     initial_parameter_sha256=self.initial_hash, final_parameter_sha256=final_hash,
                     parameter_hashes=hashes, ranks_identical=len(set(hashes)) == 1,
                     parameters_finite=finite, adapter_changed=final_hash != self.initial_hash,
                     max_reg_loss_abs=self.max_reg_loss_abs, max_reg_gradient_norm=self.max_reg_gradient_norm,
                     observed_forward_dtype=self.observed_forward_dtype,
                     fixed_rank=self.model.peft_config["default"].r,
                     accumulated_rank=self.model.peft_config["default"].r_sum,
                     reg=self.args.treelora_reg, tree=self.tree.diagnostics(),
                     global_batch=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps * self.world_size,
                     optimizer_class=type(self.optimizer).__name__,
                     optimizer_eps=self.optimizer.param_groups[0].get("eps"),
                     optimizer_weight_decay=self.optimizer.param_groups[0].get("weight_decay"))
        path = Path(self.args.output_dir) / f"treelora_audit_rank{self.rank}.json"
        path.write_text(json.dumps(audit, indent=2, allow_nan=False)+"\n")
        if not finite or not audit["ranks_identical"] or not audit["adapter_changed"]:
            raise RuntimeError(f"TreeLoRA parameter audit failed: {path}")
        if self.args.bf16 and self.observed_forward_dtype != "torch.bfloat16":
            raise RuntimeError(f"TreeLoRA BF16 dispatch audit failed: {path}")
        if self.task_id and self.args.treelora_reg > 0 and self.max_reg_gradient_norm <= 0:
            raise RuntimeError(f"TreeLoRA continuation has no regularizer gradient: {path}")
        self._saved_state = dict(schema_version=1, method="TreeLoRA", upstream_commit=UPSTREAM_COMMIT,
                                 completed_tasks=self.task_id+1, world_size=self.world_size,
                                 rank_states=states, parameter_names=self.parameter_names,
                                 adapter_parameter_sha256=final_hash)
        self._finished = True

    def save_treelora_state(self, adapter_path):
        if not self._finished:
            raise RuntimeError("Finalize the TreeLoRA task before saving its history")
        torch.save(self._saved_state, Path(adapter_path) / "treelora_state.pt")
