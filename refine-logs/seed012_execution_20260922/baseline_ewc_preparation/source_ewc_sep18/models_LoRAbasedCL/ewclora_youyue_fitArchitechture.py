from __future__ import annotations

import hashlib
import json
import logging
import random
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_LoRAbasedCL.seqlora import Learner as SeqLoRALearner
from optimer_PerturabtionType.util import (
    disable_running_stats,
    enable_running_stats,
    generate_pertubation,
)

logger = logging.getLogger(__name__)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class Learner(SeqLoRALearner):
    """SeqLoRA-based EWCLoRA.

    This version stays fully on top of the current SeqLoRA lifecycle:

    - reuse SeqLoRA's persistent LoRA backbone and task lifecycle;
    - snapshot the task-start effective LoRA update in delta-W space;
    - regularize only the current task-induced delta change with past-task
      Fisher accumulated in the same delta-W space.

    The resulting penalty is:

        0.5 * lambda * sum_k (F_past[k] + eta) * (delta_k - delta_ref_k)^2

    where ``delta_ref_k`` is the task-start checkpoint for task ``t``.
    """

    _UNSUPPORTED_OPTIMIZERS = {
        "flatlora",
        "flatlora_full",
        "faltlora",
        "faltlora_full",
        "mergegam",
        "mergegam_lora",
    }

    def __init__(self, args):
        super().__init__(args)

        if self._optimizer_type in self._UNSUPPORTED_OPTIMIZERS:
            raise ValueError(
                "ewclora_youyue_fitArchitechture currently supports "
                "sgd/adam/adamw/sam/cflat/gam/rwp/arwp only, got "
                f"optimizer_type={self._optimizer_type}."
            )

        self._ewc_lambda = float(args.get("ewc_lambda", args.get("lambda", 20.0)))
        self._ewc_gamma = float(args.get("ewc_gamma", args.get("gamma", 1.0)))
        self._ewc_eta = float(args.get("ewc_eta", args.get("eta", 0.0)))

        self._ewc_max_batches = self._parse_optional_max_batches(
            args.get("ewc_max_batches", 100)
        )
        self._compute_full_fisher = _as_bool(args.get("compute_full_fisher", False))
        # Skip the after_task delta-W Fisher pass entirely (no estimation, no
        # accumulation, no fisher mechanism metrics). Meant for lambda=0
        # controls that need a pure wall-time/compute baseline.
        self._ewc_skip_task_fisher = _as_bool(args.get("ewc_skip_task_fisher", False))
        # A task-end Fisher pass may iterate a shuffled loader with random
        # train-time augmentation.  Preserve the process-global RNG streams so
        # that enabling Fisher estimation does not silently change the next
        # task's minibatch/augmentation trajectory.  This is opt-in to keep
        # existing experiment semantics unchanged.
        self._ewc_preserve_fisher_rng = _as_bool(
            args.get("ewc_preserve_fisher_rng", False)
        )
        self._full_fisher_max_batches = self._parse_optional_max_batches(
            args.get("full_fisher_max_batches", None)
        )
        self._full_fisher_loss_scope = str(
            args.get("full_fisher_loss_scope", "joint_seen")
        ).lower()

        self._fisher_past_delta: Dict[str, torch.Tensor] = {}
        self._delta_reference: Dict[str, torch.Tensor] = {}
        self._last_full_fisher_diff: Dict[str, float] = {}

    @staticmethod
    def _parse_optional_max_batches(value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in {"", "none", "all", "full", "null", "false", "off"}:
                return None
            value = stripped
        parsed = int(value)
        return None if parsed <= 0 else parsed

    @staticmethod
    def _capture_global_rng_state():
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": None,
        }
        if torch.cuda.is_available():
            state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _restore_global_rng_state(state) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if state.get("torch_cuda") is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(state["torch_cuda"])

    def _get_current_fisher_loader(self, fallback=None):
        source = str(self.args.get("ewc_fisher_source", "heldout")).strip().lower()
        if source in {"fit", "train", "optimization"}:
            loader = getattr(self, "_current_fit_loader", None)
            if loader is not None:
                return loader
            return self.train_loader if fallback is None else fallback
        for attr in ("_current_fisher_loader", "fisher_loader", "val_loader"):
            loader = getattr(self, attr, None)
            if loader is not None:
                return loader
        return self.train_loader if fallback is None else fallback

    @staticmethod
    def _hash_tensor_items(items) -> str:
        digest = hashlib.sha256()
        for name, tensor in sorted(items, key=lambda item: str(item[0])):
            value = tensor.detach().cpu().contiguous()
            digest.update(str(name).encode("utf-8"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(str(tuple(value.shape)).encode("ascii"))
            digest.update(value.numpy().tobytes())
        return digest.hexdigest()

    def _producer_origin_receipt(self, optimizer) -> Dict[str, object]:
        model = self._unwrap_network()
        optimizer_groups = []
        for group in optimizer.param_groups:
            optimizer_groups.append(
                {
                    str(key): value
                    for key, value in group.items()
                    if key != "params" and isinstance(value, (bool, int, float, str))
                }
            )
        rng_state = self._capture_global_rng_state()
        rng_items = [("torch_cpu", rng_state["torch_cpu"])]
        for index, value in enumerate(rng_state.get("torch_cuda") or []):
            rng_items.append((f"torch_cuda_{index}", value))
        numpy_state = rng_state["numpy"]
        digest = hashlib.sha256()
        digest.update(repr(rng_state["python"]).encode("utf-8"))
        digest.update(str(numpy_state[0]).encode("ascii"))
        digest.update(np.asarray(numpy_state[1]).tobytes())
        digest.update(repr(tuple(numpy_state[2:])).encode("utf-8"))
        digest.update(self._hash_tensor_items(rng_items).encode("ascii"))

        fc = getattr(model, "fc", None)
        new_head_items = []
        if fc is not None and int(self._known_classes) < int(self._total_classes):
            new_head_items = [
                ("fc.weight.new", fc.weight[int(self._known_classes) :]),
                ("fc.bias.new", fc.bias[int(self._known_classes) :]),
            ]
        return {
            "task": int(self._cur_task),
            "origin_network_sha256": self._hash_tensor_items(model.state_dict().items()),
            "origin_rng_sha256": digest.hexdigest(),
            "expanded_new_head_sha256": (
                self._hash_tensor_items(new_head_items) if new_head_items else ""
            ),
            "optimizer_class": type(optimizer).__name__,
            "optimizer_groups_sha256": hashlib.sha256(
                json.dumps(
                    optimizer_groups, sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            ).hexdigest(),
            "producer_source_bundle_sha256": str(
                self.args.get("producer_source_bundle_sha256", "")
            ),
            "producer_env_spec_hash": str(
                self.args.get("producer_env_spec_hash", "")
            ),
            "producer_env_receipt_sha256": str(
                self.args.get("producer_env_receipt_sha256", "")
            ),
            "frozen_protocol_sha256": str(
                self.args.get("frozen_protocol_sha256", "")
            ),
        }

    @staticmethod
    def _is_delta_module(module: nn.Module) -> bool:
        return all(
            hasattr(module, attr)
            for attr in ("linear_a_q", "linear_b_q", "linear_a_v", "linear_b_v")
        )

    def _iter_delta_modules(self, model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
        for name, module in model.named_modules():
            if self._is_delta_module(module):
                yield name, module

    def _iter_delta_terms(self, model: nn.Module):
        for name, module in self._iter_delta_modules(model):
            delta_q = module.linear_b_q.weight @ module.linear_a_q.weight
            delta_v = module.linear_b_v.weight @ module.linear_a_v.weight
            yield f"{name}.q", delta_q
            yield f"{name}.v", delta_v

    def _snapshot_task_reference(self) -> None:
        base_model = self._unwrap_network()
        self._delta_reference = {
            key: delta.detach().cpu().clone()
            for key, delta in self._iter_delta_terms(base_model)
        }

    def _set_delta_hook_state(self, model: nn.Module, enabled: bool) -> None:
        for _, module in self._iter_delta_modules(model):
            if hasattr(module, "_register_delta_hook"):
                module._register_delta_hook = enabled

    def _clear_delta_hook_grads(self, model: nn.Module) -> None:
        for _, module in self._iter_delta_modules(model):
            if hasattr(module, "delta_w_q_new_grad"):
                module.delta_w_q_new_grad = None
            if hasattr(module, "delta_w_v_new_grad"):
                module.delta_w_v_new_grad = None

    def _iter_hook_grads(self, model: nn.Module):
        for name, module in self._iter_delta_modules(model):
            grad_q = getattr(module, "delta_w_q_new_grad", None)
            grad_v = getattr(module, "delta_w_v_new_grad", None)
            if grad_q is not None:
                yield f"{name}.q", grad_q
            if grad_v is not None:
                yield f"{name}.v", grad_v

    def _task_loss(self, logits: torch.Tensor, targets: torch.Tensor, class_offset: int):
        if class_offset > 0:
            fake_targets = targets - class_offset
            sliced_logits = logits[:, class_offset:]
            return F.cross_entropy(sliced_logits, fake_targets), sliced_logits, fake_targets
        return F.cross_entropy(logits, targets), logits, targets

    def _past_fisher_penalty(self, model: nn.Module) -> torch.Tensor:
        if not self._fisher_past_delta or not self._delta_reference:
            return torch.tensor(0.0, device=self._device)

        penalty = None
        for key, delta_now in self._iter_delta_terms(model):
            fisher = self._fisher_past_delta.get(key)
            delta_ref = self._delta_reference.get(key)
            if fisher is None or delta_ref is None:
                continue

            delta_ref_t = delta_ref.to(device=delta_now.device, dtype=delta_now.dtype)
            fisher_t = fisher.to(device=delta_now.device, dtype=delta_now.dtype)
            if self._ewc_eta != 0.0:
                fisher_t = fisher_t + self._ewc_eta

            delta_update = delta_now - delta_ref_t
            term = (fisher_t * delta_update.pow(2)).sum()
            penalty = term if penalty is None else penalty + term

        if penalty is None:
            return torch.tensor(0.0, device=self._device)
        return 0.5 * self._ewc_lambda * penalty

    def _regularized_task_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        class_offset: int,
        model: nn.Module | None = None,
    ):
        task_loss, eval_logits, eval_targets = self._task_loss(logits, targets, class_offset)
        if not self._fisher_past_delta or float(self._ewc_lambda) == 0.0:
            return task_loss, eval_logits, eval_targets

        if model is None:
            model = self._unwrap_network()
        return task_loss + self._past_fisher_penalty(model), eval_logits, eval_targets

    def _accumulate_fisher(
        self, old: Dict[str, torch.Tensor], new: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not old:
            return new

        out: Dict[str, torch.Tensor] = {}
        for key in set(old.keys()) | set(new.keys()):
            if key in old and key in new:
                out[key] = old[key] * self._ewc_gamma + new[key]
            elif key in old:
                out[key] = old[key] * self._ewc_gamma
            else:
                out[key] = new[key]
        return out

    def after_task(self):
        if self._ewc_skip_task_fisher:
            self._last_full_fisher_diff = {}
            self._delta_reference = {}
            if self._is_main_process:
                self._log(
                    "[EWCLoRA-Seq] ewc_skip_task_fisher=true: skipped task-end "
                    f"delta-W Fisher after task {self._cur_task}."
                )
            super().after_task()
            return

        base_model = self._unwrap_network()
        fisher_rng_state = (
            self._capture_global_rng_state()
            if self._ewc_preserve_fisher_rng
            else None
        )
        try:
            new_fisher = self._compute_delta_fisher(
                loader=self._get_current_fisher_loader(self.train_loader),
                model=base_model,
                class_offset=int(self._known_classes),
            )
        finally:
            if fisher_rng_state is not None:
                self._restore_global_rng_state(fisher_rng_state)
        self._fisher_past_delta = self._accumulate_fisher(self._fisher_past_delta, new_fisher)
        self._last_full_fisher_diff = {}
        if self._compute_full_fisher:
            self._last_full_fisher_diff = self._compute_full_fisher_diff_metrics(base_model)
        self._delta_reference = {}

        if self._is_main_process:
            self._log(
                "[EWCLoRA-Seq] Updated past Fisher for "
                f"{len(new_fisher)} delta tensors after task {self._cur_task}; "
                f"preserve_rng={self._ewc_preserve_fisher_rng}."
            )
        super().after_task()

    def _run_regularized_train(
        self,
        train_loader,
        test_loader,
        optimizer,
        scheduler,
        *,
        epochs: int,
        class_offset: int,
    ) -> None:
        self._snapshot_task_reference()
        self._epoch_dynamics_metrics = []
        receipt_required = _as_bool(self.args.get("producer_receipt_required", False))
        self._task_producer_receipt = (
            self._producer_origin_receipt(optimizer) if receipt_required else {}
        )
        sampler_order_digest = hashlib.sha256()
        sampler_order_count = 0
        trajectory_enabled = bool(
            int(self._cur_task) > 0
            and _as_bool(
                self.args.get(
                    "endpoint_region_trajectory_eval",
                    self.args.get("epoch_dynamics_eval", False),
                )
            )
        )
        if trajectory_enabled and self._is_main_process:
            initial_all_acc = self._compute_accuracy(self._network, test_loader)
            initial_new_acc = self._compute_task_local_accuracy(
                self._network, test_loader, class_offset, self._total_classes
            )
            initial_old = self._compute_old_task_full_head_metrics(test_loader)
            initial_drift = (
                self._compute_delta_drift_metrics()
                if hasattr(self, "_compute_delta_drift_metrics")
                else {}
            )
            self._epoch_dynamics_metrics.append(
                {
                    "task": int(self._cur_task),
                    "epoch": 0,
                    "epochs": int(epochs),
                    "learning_rate_used": float(optimizer.param_groups[0]["lr"]),
                    "train_accuracy": None,
                    "train_loss": None,
                    "all_seen_test_accuracy": float(initial_all_acc),
                    "current_task_local_test_accuracy": float(initial_new_acc),
                    **initial_old,
                    "delta_w_l2_squared": float(
                        initial_drift.get("delta_norm", 0.0)
                    ),
                    "fisher_weighted_drift": float(
                        initial_drift.get("fisher_weighted_drift", 0.0)
                    ),
                    "ewc_penalty_value": float(
                        initial_drift.get("ewc_penalty_value", 0.0)
                    ),
                    "expanded_new_head_sha256": self._task_producer_receipt.get(
                        "expanded_new_head_sha256", ""
                    ),
                    "origin_network_sha256": self._task_producer_receipt.get(
                        "origin_network_sha256", ""
                    ),
                    "origin_rng_sha256": self._task_producer_receipt.get(
                        "origin_rng_sha256", ""
                    ),
                    "producer_source_bundle_sha256": self._task_producer_receipt.get(
                        "producer_source_bundle_sha256", ""
                    ),
                    "producer_env_spec_hash": self._task_producer_receipt.get(
                        "producer_env_spec_hash", ""
                    ),
                    "producer_env_receipt_sha256": self._task_producer_receipt.get(
                        "producer_env_receipt_sha256", ""
                    ),
                    "frozen_protocol_sha256": self._task_producer_receipt.get(
                        "frozen_protocol_sha256", ""
                    ),
                }
            )
        prog_bar = tqdm(range(epochs), disable=not self._is_main_process)
        info = ""

        for epoch in prog_bar:
            self._network.train()
            learning_rate_used = float(optimizer.param_groups[0]["lr"])
            losses = 0.0
            correct, total = 0, 0

            for batch in train_loader:
                if len(batch) == 3:
                    sample_indices, inputs, targets = batch
                    if receipt_required:
                        index_tensor = torch.as_tensor(sample_indices).detach().cpu().contiguous()
                        sampler_order_digest.update(int(epoch).to_bytes(8, "little"))
                        sampler_order_digest.update(index_tensor.numpy().tobytes())
                        sampler_order_count += int(index_tensor.numel())
                else:
                    inputs, targets = batch
                    if receipt_required:
                        raise RuntimeError(
                            "producer receipt requires batches carrying sample indices"
                        )
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                logits, loss_value = self._step_batch(
                    optimizer=optimizer,
                    scheduler=scheduler,
                    inputs=inputs,
                    targets=targets,
                    class_offset=class_offset,
                )
                losses += float(loss_value)

                with torch.no_grad():
                    if class_offset > 0:
                        eval_logits = logits[:, class_offset:]
                        eval_targets = targets - class_offset
                    else:
                        eval_logits = logits
                        eval_targets = targets
                    preds = torch.argmax(eval_logits, dim=1)
                    correct += int(preds.eq(eval_targets).sum().item())
                    total += int(eval_targets.numel())

            if scheduler is not None:
                scheduler.step()

            train_acc = 0.0 if total == 0 else round(100.0 * correct / total, 2)
            if ((epoch % 5 == 4) or epoch == epochs - 1) and self._is_main_process:
                test_acc = self._compute_accuracy(self._network, test_loader)
                cur_task_test_acc = self._compute_task_local_accuracy(
                    self._network,
                    test_loader,
                    class_offset,
                    self._total_classes,
                )
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / max(1, len(train_loader)):.3f}, "
                    f"Train_accy {train_acc:.2f}, Test_accy {test_acc:.2f}, "
                    f"CurTask_Test {cur_task_test_acc:.2f}"
                )
                if trajectory_enabled:
                    old_metrics = self._compute_old_task_full_head_metrics(test_loader)
                    drift_metrics = (
                        self._compute_delta_drift_metrics()
                        if hasattr(self, "_compute_delta_drift_metrics")
                        else {}
                    )
                    self._epoch_dynamics_metrics.append(
                        {
                            "task": int(self._cur_task),
                            "epoch": int(epoch + 1),
                            "epochs": int(epochs),
                            "learning_rate_used": learning_rate_used,
                            "train_accuracy": float(train_acc),
                            "train_loss": float(losses / max(1, len(train_loader))),
                            "all_seen_test_accuracy": float(test_acc),
                            "current_task_local_test_accuracy": float(
                                cur_task_test_acc
                            ),
                            **old_metrics,
                            "delta_w_l2_squared": float(
                                drift_metrics.get("delta_norm", 0.0)
                            ),
                            "fisher_weighted_drift": float(
                                drift_metrics.get("fisher_weighted_drift", 0.0)
                            ),
                            "ewc_penalty_value": float(
                                drift_metrics.get("ewc_penalty_value", 0.0)
                            ),
                            "expanded_new_head_sha256": self._task_producer_receipt.get(
                                "expanded_new_head_sha256", ""
                            ),
                            "origin_network_sha256": self._task_producer_receipt.get(
                                "origin_network_sha256", ""
                            ),
                            "origin_rng_sha256": self._task_producer_receipt.get(
                                "origin_rng_sha256", ""
                            ),
                            "producer_source_bundle_sha256": self._task_producer_receipt.get(
                                "producer_source_bundle_sha256", ""
                            ),
                            "producer_env_spec_hash": self._task_producer_receipt.get(
                                "producer_env_spec_hash", ""
                            ),
                            "producer_env_receipt_sha256": self._task_producer_receipt.get(
                                "producer_env_receipt_sha256", ""
                            ),
                            "frozen_protocol_sha256": self._task_producer_receipt.get(
                                "frozen_protocol_sha256", ""
                            ),
                        }
                    )
            elif self._is_main_process:
                info = (
                    f"Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => "
                    f"Loss {losses / max(1, len(train_loader)):.3f}, "
                    f"Train_accy {train_acc:.2f}"
                )

            if self._is_main_process:
                prog_bar.set_description(info)

        if self._is_main_process and info:
            self._log(info)
        if receipt_required:
            self._task_producer_receipt.update(
                {
                    "sampler_order_sha256": sampler_order_digest.hexdigest(),
                    "sampler_order_samples": int(sampler_order_count),
                    "sampler_epochs": int(epochs),
                }
            )
            for row in self._epoch_dynamics_metrics:
                row["sampler_order_sha256"] = sampler_order_digest.hexdigest()

    def _compute_old_task_full_head_metrics(self, loader) -> Dict[str, float]:
        """Measure old-task damage under the full current classifier head."""
        model = self._network
        was_training = bool(model.training)
        model.eval()
        loss_sum = 0.0
        correct = 0
        total = 0
        try:
            with torch.no_grad():
                for batch in loader:
                    if len(batch) == 3:
                        _, inputs, targets = batch
                    else:
                        inputs, targets = batch
                    mask = (targets >= 0) & (targets < int(self._known_classes))
                    if not bool(mask.any()):
                        continue
                    inputs = inputs[mask].to(self._device)
                    targets = targets[mask].to(self._device)
                    output = model(inputs)
                    logits = output["logits"] if isinstance(output, dict) else output
                    loss_sum += float(
                        F.cross_entropy(logits, targets, reduction="sum").item()
                    )
                    correct += int(logits.argmax(dim=1).eq(targets).sum().item())
                    total += int(targets.numel())
        finally:
            model.train(was_training)
        if total <= 0:
            raise RuntimeError(
                "endpoint-region trajectory loader contains no old-task examples"
            )
        return {
            "old_task_samples": int(total),
            "old_task_full_head_loss": float(loss_sum / total),
            "old_task_full_head_accuracy": float(100.0 * correct / total),
        }

    def get_epoch_dynamics_metrics(self) -> List[Dict[str, float]]:
        return [dict(row) for row in getattr(self, "_epoch_dynamics_metrics", [])]

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        self._run_regularized_train(
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            epochs=int(self.args["init_epoch"]),
            class_offset=0,
        )

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        self._run_regularized_train(
            train_loader,
            test_loader,
            optimizer,
            scheduler,
            epochs=int(self.args["epochs"]),
            class_offset=int(self._known_classes),
        )

    def _step_batch(self, optimizer, scheduler, inputs, targets, class_offset: int):
        if self._optimizer_type == "cflat":

            def closure():
                optimizer.zero_grad()
                outputs = self._network(inputs)
                loss, _, _ = self._regularized_task_loss(
                    outputs["logits"],
                    targets,
                    class_offset,
                    model=self._unwrap_network(),
                )
                loss.backward()
                return outputs, [loss]

            _, loss_list = optimizer.step(closure=closure)
            with torch.no_grad():
                logits = self._network(inputs)["logits"]
            loss_value = torch.stack([loss_term.detach() for loss_term in loss_list]).sum()
            return logits.detach(), float(loss_value.item())

        if self._optimizer_type == "gam":

            def closure():
                optimizer.zero_grad()
                outputs = self._network(inputs)
                loss, _, _ = self._regularized_task_loss(
                    outputs["logits"],
                    targets,
                    class_offset,
                    model=self._unwrap_network(),
                )
                loss_value = loss.detach()
                loss.backward()
                return outputs, loss_value

            outputs, loss_value = optimizer.step(closure=closure)
            scalar = loss_value.item() if torch.is_tensor(loss_value) else loss_value
            return outputs["logits"].detach(), float(scalar)

        if self._optimizer_type == "rwp":
            return self._rwp_step(optimizer, scheduler, inputs, targets, class_offset)

        if self._optimizer_type == "arwp":
            if getattr(self, "_rwp_std_follow_lr", False):
                try:
                    cur_lr = (
                        float(scheduler.get_last_lr()[0])
                        if hasattr(scheduler, "get_last_lr")
                        else float(optimizer.param_groups[0]["lr"])
                    )
                except Exception:
                    cur_lr = float(optimizer.param_groups[0]["lr"])
                base_lr = float(getattr(self, "_rwp_lr0", cur_lr))
                scale = (cur_lr / base_lr) if base_lr > 0 else 1.0
                optimizer.std = float(self._rwp_std) * scale

            def closure():
                optimizer.zero_grad()
                outputs = self._network(inputs)
                loss, _, _ = self._regularized_task_loss(
                    outputs["logits"],
                    targets,
                    class_offset,
                    model=self._unwrap_network(),
                )
                loss_value = loss.detach()
                loss.backward()
                return outputs, loss_value

            outputs, loss_value = optimizer.step(closure=closure)
            scalar = loss_value.item() if torch.is_tensor(loss_value) else loss_value
            return outputs["logits"].detach(), float(scalar)

        optimizer.zero_grad()
        logits = self._network(inputs)["logits"]
        loss, _, _ = self._regularized_task_loss(
            logits,
            targets,
            class_offset,
            model=self._unwrap_network(),
        )

        if self._optimizer_type == "sam":
            loss.backward()
            optimizer.first_step(zero_grad=True)
            logits = self._network(inputs)["logits"]
            second_loss, _, _ = self._regularized_task_loss(
                logits,
                targets,
                class_offset,
                model=self._unwrap_network(),
            )
            second_loss.backward()
            optimizer.second_step(zero_grad=True)
            return logits.detach(), float(second_loss.detach().item())

        loss.backward()
        optimizer.step()
        return logits.detach(), float(loss.detach().item())

    def _rwp_step(self, optimizer, scheduler, inputs, targets, class_offset: int):
        if getattr(self, "_rwp_std_follow_lr", False):
            try:
                cur_lr = (
                    float(scheduler.get_last_lr()[0])
                    if hasattr(scheduler, "get_last_lr")
                    else float(optimizer.param_groups[0]["lr"])
                )
            except Exception:
                cur_lr = float(optimizer.param_groups[0]["lr"])
            base_lr = float(getattr(self, "_rwp_lr0", cur_lr))
            scale = (cur_lr / base_lr) if base_lr > 0 else 1.0
            rwp_std = float(self._rwp_std) * scale
        else:
            rwp_std = float(self._rwp_std)

        enable_running_stats(self._network)
        optimizer.zero_grad()
        outputs = self._network(inputs)
        logits_clean = outputs["logits"]
        base_model = self._unwrap_network()
        loss_clean, _, _ = self._regularized_task_loss(
            logits_clean,
            targets,
            class_offset,
            model=base_model,
        )
        loss_clean.backward()

        g0 = {}
        for name, param in base_model.named_parameters():
            if param.requires_grad and param.grad is not None:
                g0[name] = param.grad.detach().clone()

        disable_running_stats(self._network)
        noise_dict = {}
        with torch.no_grad():
            std_for_noise = float(self.args.get("noise_std", rwp_std))
            for name, param in base_model.named_parameters():
                if self._rwp_range == "lora" and (not param.requires_grad or param.numel() == 0):
                    continue
                fisher_param = None
                if hasattr(self, "_rwp_fisher") and name in self._rwp_fisher:
                    fisher_param = self._rwp_fisher[name]
                perturb = generate_pertubation(
                    param,
                    pertubation_mode=self.rwp_noise_type,
                    std=std_for_noise,
                    fisher_param=fisher_param,
                    fisher_scaler=float(self._rwp_eta),
                )
                param.data.add_(perturb)
                noise_dict[name] = perturb

        optimizer.zero_grad()
        outputs_noisy = self._network(inputs)
        logits_noisy = outputs_noisy["logits"]
        loss_noisy, _, _ = self._regularized_task_loss(
            logits_noisy,
            targets,
            class_offset,
            model=base_model,
        )
        loss_noisy.backward()

        if hasattr(self, "_rwp_fisher"):
            with torch.no_grad():
                for name, param in base_model.named_parameters():
                    if not param.requires_grad or param.grad is None:
                        continue
                    g2 = param.grad.detach() ** 2
                    if name not in self._rwp_fisher:
                        self._rwp_fisher[name] = g2
                    else:
                        self._rwp_fisher[name] = float(self._rwp_beta) * self._rwp_fisher[name] + g2

        with torch.no_grad():
            for name, param in base_model.named_parameters():
                if name in noise_dict:
                    param.data.sub_(noise_dict[name])

        lam = float(self._rwp_lambda)
        for name, param in base_model.named_parameters():
            if not param.requires_grad or param.grad is None:
                continue
            g1 = param.grad.detach()
            g0_n = g0.get(name, torch.zeros_like(g1))
            param.grad.data.copy_(lam * g1 + (1.0 - lam) * g0_n)

        optimizer.step()
        enable_running_stats(self._network)

        mixed_loss = lam * float(loss_noisy.detach().item()) + (1.0 - lam) * float(
            loss_clean.detach().item()
        )
        return logits_clean.detach(), mixed_loss

    def _compute_delta_fisher(self, loader, model: nn.Module, class_offset: int) -> Dict[str, torch.Tensor]:
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        total = 0

        self._set_delta_hook_state(model, True)
        try:
            for batch_idx, batch in enumerate(loader):
                if self._ewc_max_batches is not None and batch_idx >= self._ewc_max_batches:
                    break

                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                model.zero_grad(set_to_none=True)
                self._clear_delta_hook_grads(model)

                with torch.enable_grad():
                    logits = model(inputs)["logits"]
                    loss, _, _ = self._task_loss(logits, targets, class_offset)
                    loss.backward()

                batch_size = int(inputs.size(0))
                total += batch_size
                for key, grad in self._iter_hook_grads(model):
                    grad_sq = (grad.detach().cpu() ** 2) * batch_size
                    fisher[key] = grad_sq if key not in fisher else fisher[key] + grad_sq
        finally:
            self._set_delta_hook_state(model, False)
            self._clear_delta_hook_grads(model)

        if total > 0:
            for key in fisher:
                fisher[key] = fisher[key] / float(total)
        return fisher

    def _build_all_seen_train_loader(self):
        data_manager = getattr(self, "_data_manager", None) or getattr(self, "data_manager", None)
        if data_manager is None:
            return None

        dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="train", mode="train"
        )
        batch_size = int(self.args.get("batch_size", 128))
        num_workers = int(
            self.args.get(
                "full_fisher_num_workers",
                self.args.get("train_num_workers", 8),
            )
        )
        if hasattr(data_manager, "build_dataloader"):
            return data_manager.build_dataloader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
        return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    def _maybe_normalize_full_seen_fisher(
        self, fisher: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        if not fisher or not bool(getattr(self, "_normalize_fisher", False)):
            return fisher

        total_trace = sum(float(value.sum().item()) for value in fisher.values())
        denom = total_trace + float(getattr(self, "_fisher_norm_eps", 1e-12))
        return {key: value / denom for key, value in fisher.items()}

    def _compute_fisher_all_seen(
        self, model: nn.Module
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Batch empirical Fisher over all seen train data in delta-W coordinates."""
        loader = self._build_all_seen_train_loader()
        if loader is None:
            return {}, 0
        if self._full_fisher_loss_scope not in {"joint_seen", "seen", "class_il"}:
            raise ValueError(
                "LoRA full Fisher currently supports only joint_seen/class-IL loss, "
                f"got full_fisher_loss_scope={self._full_fisher_loss_scope}."
            )

        was_training = model.training
        model.eval()
        fisher: Dict[str, torch.Tensor] = {}
        total = 0
        max_batches = self._full_fisher_max_batches

        self._set_delta_hook_state(model, True)
        try:
            for batch_idx, batch in enumerate(
                tqdm(
                    loader,
                    desc="Computing LoRA full Fisher [all seen]",
                    disable=not self._is_main_process,
                )
            ):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                if len(batch) == 3:
                    _, inputs, targets = batch
                else:
                    inputs, targets = batch
                inputs = inputs.to(self._device, non_blocking=True)
                targets = targets.to(self._device, non_blocking=True)

                model.zero_grad(set_to_none=True)
                self._clear_delta_hook_grads(model)

                with torch.enable_grad():
                    logits = model(inputs)["logits"][:, : int(self._total_classes)]
                    loss = F.cross_entropy(logits, targets)
                    loss.backward()

                batch_size = int(inputs.size(0))
                total += batch_size
                for key, grad in self._iter_hook_grads(model):
                    grad_sq = (grad.detach().cpu() ** 2) * batch_size
                    fisher[key] = grad_sq if key not in fisher else fisher[key] + grad_sq
        finally:
            self._set_delta_hook_state(model, False)
            self._clear_delta_hook_grads(model)
            model.zero_grad(set_to_none=True)
            model.train(was_training)

        if total > 0:
            for key in fisher:
                fisher[key] = fisher[key] / float(total)
        fisher = self._maybe_normalize_full_seen_fisher(fisher)
        if self._is_main_process:
            self._log(
                "[EWCLoRA-Seq] Full Fisher (all seen joint-head delta-W) "
                f"estimated over {total} samples."
            )
        return fisher, total

    @staticmethod
    def _match_fisher_shape(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        value_cpu = value.detach().cpu().to(dtype=reference.dtype)
        ref_cpu = reference.detach().cpu()
        if tuple(value_cpu.shape) == tuple(ref_cpu.shape):
            return value_cpu

        if value_cpu.ndim != ref_cpu.ndim:
            flat_value = value_cpu.reshape(-1)
            matched_flat = torch.zeros(ref_cpu.numel(), dtype=ref_cpu.dtype)
            n = min(int(flat_value.numel()), int(matched_flat.numel()))
            if n > 0:
                matched_flat[:n] = flat_value[:n]
            return matched_flat.view_as(ref_cpu)

        matched = torch.zeros_like(ref_cpu)
        slices = tuple(
            slice(0, min(int(value_cpu.shape[dim]), int(ref_cpu.shape[dim])))
            for dim in range(value_cpu.ndim)
        )
        matched[slices] = value_cpu[slices]
        return matched

    def _compute_full_fisher_diff_metrics(self, model: nn.Module) -> Dict[str, float]:
        """Compare online accumulated delta-W Fisher with all-seen joint-head Fisher."""
        if not self._fisher_past_delta:
            return {}

        full_fisher, num_samples = self._compute_fisher_all_seen(model)
        if not full_fisher:
            return {}

        accum_vals: List[torch.Tensor] = []
        full_vals: List[torch.Tensor] = []
        for key in sorted(set(self._fisher_past_delta.keys()) | set(full_fisher.keys())):
            accum = self._fisher_past_delta.get(key)
            full = full_fisher.get(key)
            if accum is None and full is None:
                continue
            if accum is None:
                accum = torch.zeros_like(full.detach().cpu())
            if full is None:
                full = torch.zeros_like(accum.detach().cpu())
            full_matched = self._match_fisher_shape(full, accum)
            accum_vals.append(accum.detach().cpu().float().view(-1))
            full_vals.append(full_matched.detach().cpu().float().view(-1))
        if not accum_vals:
            return {}

        accum_flat = torch.cat(accum_vals)
        full_flat = torch.cat(full_vals)
        diff = full_flat - accum_flat
        numel = int(accum_flat.numel())

        accum_trace = float(accum_flat.sum().item())
        full_trace = float(full_flat.sum().item())
        trace_diff = float(diff.sum().item())

        accum_fro = float(accum_flat.pow(2).sum().sqrt().item())
        full_fro = float(full_flat.pow(2).sum().sqrt().item())
        diff_fro = float(diff.pow(2).sum().sqrt().item())

        accum_l1 = float(accum_flat.abs().sum().item())
        full_l1 = float(full_flat.abs().sum().item())
        abs_diff_l1 = float(diff.abs().sum().item())

        eps = 1e-12
        cosine_sim = float(
            (accum_flat * full_flat).sum().item() / max(accum_fro * full_fro, eps)
        )
        sign_agree = float(((accum_flat > 0) == (full_flat > 0)).float().mean().item())
        max_abs_diff = float(diff.abs().max().item())

        k1pct = max(1, int(numel * 0.01))
        full_top_idx = full_flat.topk(k1pct, largest=True).indices
        top1pct_full_sum = float(full_flat[full_top_idx].sum().item())
        top1pct_accum_sum = float(accum_flat[full_top_idx].sum().item())

        metrics = {
            "compute_full_fisher": True,
            "full_fisher_delta_space": "delta_w",
            "full_fisher_loss_scope": "joint_seen",
            "full_fisher_head_scope": "seen_joint_head",
            "full_fisher_normalized": bool(getattr(self, "_normalize_fisher", False)),
            "full_fisher_num_samples": int(num_samples),
            "full_fisher_num_params": int(numel),
            "full_fisher_trace": full_trace,
            "full_fisher_fro": full_fro,
            "full_fisher_l1": full_l1,
            "full_fisher_top1pct_trace": top1pct_full_sum,
            "accum_fisher_trace": accum_trace,
            "accum_fisher_fro": accum_fro,
            "accum_fisher_l1": accum_l1,
            "accum_fisher_top1pct_by_full": top1pct_accum_sum,
            "full_vs_accum_trace_diff": trace_diff,
            "full_vs_accum_abs_trace_diff": abs(trace_diff),
            "full_vs_accum_trace_ratio": full_trace / max(abs(accum_trace), eps),
            "full_vs_accum_fro_diff": diff_fro,
            "full_vs_accum_rel_fro": diff_fro / max(accum_fro, eps),
            "full_vs_accum_rel_fro_wrt_full": diff_fro / max(full_fro, eps),
            "full_vs_accum_rel_l1": abs_diff_l1 / max(accum_l1, eps),
            "full_vs_accum_rel_l1_wrt_full": abs_diff_l1 / max(full_l1, eps),
            "full_vs_accum_max_abs_diff": max_abs_diff,
            "full_vs_accum_cosine_sim": cosine_sim,
            "full_vs_accum_sign_agree": sign_agree,
            "full_vs_accum_top1pct_accum_coverage": (
                top1pct_accum_sum / max(top1pct_full_sum, eps)
            ),
        }
        if self._is_main_process:
            self._log(f"[EWCLoRA-Seq][FullFisherDiff] task={self._cur_task} {metrics}")
        return metrics


class EWCLoRA(Learner):
    """Backward-compatible alias."""
