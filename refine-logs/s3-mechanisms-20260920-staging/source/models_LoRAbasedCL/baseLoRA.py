import hashlib
import json
import logging
import os
from typing import Optional
import numpy as np
from torch import optim
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

# from models.base import BaseLearner
from models_CL.baseLearner import BaseLearner
from optimer_PerturabtionType.util import generate_pertubation
from backbone.lora import _LoRA_qkv_timm_train
from utils.endpoint_region import (
    task_resume_artifact_sha256,
    validate_task_resume_sidecars,
)


class LoraBaseLearner(BaseLearner):
    """Base helper for LoRA learners with DataParallel/DDP awareness."""

    def __init__(self, args):
        super().__init__(args)
        self._ddp_enabled = False
        self._is_main_process = True
        self._dp_device_ids: list[int] = []
        self._data_parallel_enabled = False
        self._rank = self.args.get("lora_rank", 10)


    # ------------------------------------------------------------------
    # Distribution helpers
    # ------------------------------------------------------------------
    def _refresh_distributed_context(self) -> None:
        """Refresh cached distributed flags based on current torch.distributed state."""
        self._ddp_enabled = dist.is_available() and dist.is_initialized()
        self._is_main_process = (not self._ddp_enabled) or dist.get_rank() == 0

        device_ids: list[int] = []
        for device in self._multiple_gpus:
            if isinstance(device, torch.device):
                if device.type == "cuda" and device.index is not None:
                    device_ids.append(device.index)
            elif isinstance(device, int):
                device_ids.append(device)
        self._dp_device_ids = device_ids
        self._data_parallel_enabled = (len(self._dp_device_ids) > 1) and (not self._ddp_enabled)

    # ------------------------------------------------------------------
    def _unwrap_network(self) -> nn.Module:
        """Return the underlying nn.Module, removing any parallel wrappers."""
        if isinstance(self._network, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            return self._network.module
        return self._network

    @staticmethod
    def _file_sha256(path: str) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _canonical_sha256(value) -> str:
        payload = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _apply_parallel_wrapper(self) -> None:
        """Apply DataParallel/DDP wrapping according to current context."""
        if self._ddp_enabled:
            if isinstance(self._network, nn.parallel.DistributedDataParallel):
                return
            device = self._device
            device_index: Optional[int] = device.index if device.type == "cuda" else None
            self._network = nn.parallel.DistributedDataParallel(
                self._network,
                device_ids=[device_index] if device_index is not None else None,
                output_device=device_index,
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
        elif self._data_parallel_enabled:
            if isinstance(self._network, nn.DataParallel):
                return
            if self._dp_device_ids:
                self._network = nn.DataParallel(self._network, device_ids=self._dp_device_ids)
        else:
            if isinstance(self._network, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
                self._network = self._network.module

    def _prepare_network(self) -> None:
        """Move the underlying network to device and wrap if needed."""
        base_network = self._unwrap_network()
        base_network.to(self._device)
        self._network = base_network
        self._apply_parallel_wrapper()

    

    # ------------------------------------------------------------------
    def _log(self, message: str) -> None:
        if self._is_main_process:
            logging.info(message)

    def _maybe_barrier(self) -> None:
        if self._ddp_enabled:
            dist.barrier()

    # ------------------------------------------------------------------
    # FlatLoRA helpers
    # ------------------------------------------------------------------
    def _init_flatlora_state(self, args) -> None:
        self._flatlora_rho = float(args.get("flatlora_rho", args.get("rho", 0.05)))
        self._flatlora_noise_type = str(args.get("flatlora_noise_type", "flatLoRA_Gauss"))
        self._flatlora_use_cosine_schedule = bool(args.get("flatlora_use_cosine_schedule", True))
        self._flatlora_total_steps = 1
        self._flatlora_step_idx = 0
        self._flatlora_warned_no_modules = False

    def _is_flatlora_optimizer(self) -> bool:
        return getattr(self, "_optimizer_type", "").lower() in {
            "flatlora",
            "flatlora_full",
            "faltlora",
            "faltlora_full",
        }

    def _reset_flatlora_schedule(self, total_steps: int) -> None:
        self._flatlora_total_steps = max(int(total_steps), 1)
        self._flatlora_step_idx = 0

    def _get_scale_value(self, scale_module) -> float:
        if hasattr(scale_module, "param"):
            param = getattr(scale_module, "param")
            if torch.is_tensor(param):
                return float(param.detach().view(-1)[0].item())
        return 1.0

    def _collect_flatlora_modules(self):
        base_model = self._unwrap_network()
        return [module for module in base_model.modules() if isinstance(module, _LoRA_qkv_timm_train)]

    def _build_effective_qkv_weight(self, module: _LoRA_qkv_timm_train) -> torch.Tensor:
        merged_weight = module.qkv.weight.detach().clone()
        device = merged_weight.device
        dtype = merged_weight.dtype

        for task_idx in range(int(module.task_id)):
            key_a = f"saved_A_{task_idx}"
            key_b = f"saved_B_{task_idx}"
            if key_a not in module.saved_A or key_b not in module.saved_B:
                continue

            saved_A = module.saved_A[key_a]
            saved_B = module.saved_B[key_b]
            layer_pairs = list(zip(saved_A, saved_B))[module.t_layer_i * 2 : module.t_layer_i * 2 + 2]
            if len(layer_pairs) < 2:
                continue

            q_entry, v_entry = layer_pairs
            A_q, B_q = q_entry
            A_v, B_v = v_entry
            hist_scale = (
                self._get_scale_value(module.scaling_factor_prev[task_idx])
                if task_idx < len(module.scaling_factor_prev)
                else 1.0
            )

            delta_q = (
                B_q.weight.detach().to(device=device, dtype=dtype)
                @ A_q.weight.detach().to(device=device, dtype=dtype)
            )
            delta_v = (
                B_v.weight.detach().to(device=device, dtype=dtype)
                @ A_v.weight.detach().to(device=device, dtype=dtype)
            )
            merged_weight[: module.dim, :].add_(hist_scale * delta_q)
            merged_weight[-module.dim :, :].add_(hist_scale * delta_v)

        cur_scale = self._get_scale_value(module.scaling_factor[0])
        delta_q_cur = (
            module.linear_b_q.weight.detach().to(device=device, dtype=dtype)
            @ module.linear_a_q.weight.detach().to(device=device, dtype=dtype)
        )
        delta_v_cur = (
            module.linear_b_v.weight.detach().to(device=device, dtype=dtype)
            @ module.linear_a_v.weight.detach().to(device=device, dtype=dtype)
        )
        merged_weight[: module.dim, :].add_(cur_scale * delta_q_cur)
        merged_weight[-module.dim :, :].add_(cur_scale * delta_v_cur)
        return merged_weight

    def _current_flatlora_std(self) -> float:
        base_rho = float(self._flatlora_rho)
        if (not self._flatlora_use_cosine_schedule) or self._flatlora_total_steps <= 1:
            return base_rho
        progress = min(max(float(self._flatlora_step_idx) / float(self._flatlora_total_steps), 0.0), 1.0)
        factor = 0.5 * (1.0 - np.cos(progress * np.pi))
        return base_rho * float(factor)

    def _apply_flatlora_noise(self, std: float):
        flatlora_modules = self._collect_flatlora_modules()
        if not flatlora_modules:
            if not self._flatlora_warned_no_modules:
                self._log("[FlatLoRA] No mergeable LoRA-qkv modules found; falling back to the clean objective.")
                self._flatlora_warned_no_modules = True
            return []

        injected = []
        with torch.no_grad():
            for module in flatlora_modules:
                effective_weight = self._build_effective_qkv_weight(module)
                noise = generate_pertubation(
                    effective_weight,
                    pertubation_mode=self._flatlora_noise_type,
                    std=float(std),
                )
                module.qkv.weight.data.add_(noise)
                injected.append((module.qkv.weight, noise))
        return injected

    def _revert_flatlora_noise(self, injected) -> None:
        if not injected:
            return
        with torch.no_grad():
            for weight, noise in injected:
                weight.data.sub_(noise)

    def _flatlora_step(self, optimizer, loss_closure):
        optimizer.zero_grad()
        std = self._current_flatlora_std()
        injected = self._apply_flatlora_noise(std) if std > 0 else []
        try:
            logits, loss = loss_closure()
            loss.backward()
        finally:
            self._revert_flatlora_noise(injected)
        optimizer.step()
        self._flatlora_step_idx += 1
        return logits.detach(), float(loss.detach().item())

    # ------------------------------------------------------------------
    # Evaluation helpers
    # ------------------------------------------------------------------
    def load_task_checkpoint(self, checkpoint_dir: str, task_idx: int, data_manager) -> None:
        """Restore a SeqLoRA-style task checkpoint and continue from it.

        The saved SeqLoRA files store the trainable current A/B factors after
        each task. For forked trajectory experiments we copy those factors into
        the current trainable LoRA modules instead of rebuilding an eval-only
        backbone that would treat them as frozen historical adapters.
        """
        checkpoint_dir = os.path.abspath(os.path.expanduser(str(checkpoint_dir)))
        task_idx = int(task_idx)
        if task_idx < 0:
            raise ValueError(f"resume task_idx must be non-negative, got {task_idx}")

        self._validate_resume_experiment_identity(checkpoint_dir, task_idx)

        _, task_end = data_manager.get_task_class_range(task_idx)
        self._cur_task = task_idx
        self._known_classes = int(task_end)
        self._total_classes = int(task_end)

        self._refresh_distributed_context()

        network = self._unwrap_network()
        network.update_fc(self._total_classes)
        if hasattr(network, "load_fc"):
            network.load_fc(checkpoint_dir, task_idx)

        backbone = getattr(network, "backbone", None)
        self._load_current_lora_factors(backbone, checkpoint_dir, task_idx)

        self._network = network
        self._prepare_network()
        self._load_auxiliary_task_state(checkpoint_dir, task_idx)
        self._log(
            f"[Resume] Restored task {task_idx} checkpoint from {checkpoint_dir}; "
            f"next task will be {task_idx + 1}"
        )

    def save_task_checkpoint(self, ckpt_dir: str, task_idx: int) -> None:
        """Save a staged LoRA checkpoint plus optional EWC/Fisher state.

        The resume path uses the side-car LoRA/FC/Fisher files.  The compact
        ``task_XX.pt`` manifest is kept for external analysis scripts that load
        a full ``network_state`` directly.
        """
        ckpt_dir = os.path.abspath(os.path.expanduser(str(ckpt_dir)))
        os.makedirs(ckpt_dir, exist_ok=True)
        task_idx = int(task_idx)

        base_net = self._unwrap_network()
        backbone = getattr(base_net, "backbone", None)
        w_as = getattr(backbone, "w_As", None)
        w_bs = getattr(backbone, "w_Bs", None)
        if w_as is not None:
            torch.save(
                [
                    layer.weight.detach().cpu().clone()
                    if hasattr(layer, "weight") else layer.detach().cpu().clone()
                    for layer in w_as
                ],
                os.path.join(ckpt_dir, f"lora_w_a_{task_idx}.pt"),
            )
        if w_bs is not None:
            torch.save(
                [
                    layer.weight.detach().cpu().clone()
                    if hasattr(layer, "weight") else layer.detach().cpu().clone()
                    for layer in w_bs
                ],
                os.path.join(ckpt_dir, f"lora_w_b_{task_idx}.pt"),
            )
        if hasattr(base_net, "save_fc"):
            base_net.save_fc(ckpt_dir, task_idx)

        fisher_state = {}
        if hasattr(self, "_fisher_past_delta"):
            fisher_state["fisher_past_delta"] = {
                key: value.detach().cpu().clone()
                for key, value in getattr(self, "_fisher_past_delta", {}).items()
            }
        if hasattr(self, "_fisher_past_trace_mass"):
            fisher_state["fisher_past_trace_mass"] = float(getattr(self, "_fisher_past_trace_mass", 0.0))
        if hasattr(self, "_last_task_new_fisher_stats"):
            fisher_state["last_task_new_fisher_stats"] = dict(getattr(self, "_last_task_new_fisher_stats", {}))
        if hasattr(self, "_fisher_new_task_raw_trace"):
            fisher_state["fisher_new_task_raw_trace"] = float(self._fisher_new_task_raw_trace)
        require_fisher = str(
            self.args.get("checkpoint_require_nonempty_fisher", False)
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        if require_fisher:
            self._validate_fisher_checkpoint_state(
                fisher_state,
                source=f"in-memory task {task_idx} Fisher",
                require_shape_match=True,
            )
        fisher_state_path = os.path.join(ckpt_dir, f"fisher_state_{task_idx}.pt")
        if fisher_state:
            torch.save(fisher_state, fisher_state_path)

        producer_receipt = dict(getattr(self, "_task_producer_receipt", {}) or {})
        split_info = dict(getattr(self, "_current_fisher_split_info", {}) or {})
        train_source_indices = [
            int(value) for value in split_info.get("train_source_indices", [])
        ]
        diagnostic_source_indices = [
            int(value) for value in split_info.get("val_source_indices", [])
        ]
        if split_info:
            producer_receipt.update(
                {
                    "split_seed": int(split_info.get("seed", -1)),
                    "fit_samples": int(split_info.get("train_samples", 0)),
                    "diagnostic_samples": int(split_info.get("val_samples", 0)),
                    "fit_source_indices_sha256": self._canonical_sha256(
                        train_source_indices
                    ),
                    "diagnostic_source_indices_sha256": self._canonical_sha256(
                        diagnostic_source_indices
                    ),
                    "fit_diagnostic_disjoint": bool(
                        set(train_source_indices).isdisjoint(diagnostic_source_indices)
                    ),
                }
            )
        producer_receipt.update(
            {
                "fisher_source": str(
                    self.args.get("ewc_fisher_source", "heldout")
                ).lower(),
                "fisher_sidecar_sha256": (
                    self._file_sha256(fisher_state_path)
                    if os.path.isfile(fisher_state_path)
                    else ""
                ),
            }
        )

        manifest = {
            "cur_task": int(task_idx),
            "known_classes": int(getattr(self, "_known_classes", 0)),
            "total_classes": int(getattr(self, "_total_classes", 0)),
            "model_name": self.args.get("model_name", ""),
            "optimizer_type": self.args.get("optimizer_type", ""),
            "experiment_identity": {
                "fs_direction_mode": self.args.get("fs_direction_mode", ""),
                "fs_base_optimizer": self.args.get("fs_base_optimizer", ""),
                "fs_alpha_mode": self.args.get("fs_alpha_mode", ""),
                "fs_alpha": self.args.get("fs_alpha", ""),
                "fs_rho": self.args.get("fs_rho", ""),
                "fs_preserve_probe_rng": self.args.get("fs_preserve_probe_rng", ""),
                "fisher_mode": self.args.get("fisher_mode", ""),
                "fisher_source": self.args.get("ewc_fisher_source", "heldout"),
                "ewc_normalize_fisher": self.args.get("ewc_normalize_fisher", ""),
                "ewc_normalize_accumulated_fisher": self.args.get(
                    "ewc_normalize_accumulated_fisher", ""
                ),
                "ewc_fisher_damping_beta": self.args.get(
                    "ewc_fisher_damping_beta", ""
                ),
                "ewc_gamma": self.args.get("ewc_gamma", ""),
            },
            "network_state": {
                key: value.detach().cpu().clone()
                for key, value in base_net.state_dict().items()
            },
            "has_fisher_state": bool(fisher_state),
            "producer_receipt": producer_receipt,
        }
        torch.save(manifest, os.path.join(ckpt_dir, f"task_{task_idx:02d}.pt"))
        self._save_experiment_identity(ckpt_dir, task_idx, fisher_state)
        self._log(f"[TaskCheckpoint] Saved staged checkpoint to {ckpt_dir}")

    def _save_experiment_identity(self, ckpt_dir: str, task_idx: int, fisher_state) -> None:
        fisher = fisher_state.get("fisher_past_delta", {}) if isinstance(fisher_state, dict) else {}
        fisher_trace = float(
            sum(value.detach().float().sum().item() for value in fisher.values())
        ) if fisher else 0.0
        seed = self.args.get("seed", "")
        if isinstance(seed, (list, tuple)) and len(seed) == 1:
            seed = seed[0]
        identity = {
            "status": "complete",
            "task": int(task_idx),
            "dataset": self.args.get("dataset", ""),
            "backbone_type": self.args.get("backbone_type", ""),
            "lora_rank": int(self.args.get("lora_rank", getattr(self, "_rank", 0))),
            "seed": seed,
            "class_shuffle": self.args.get("class_shuffle", ""),
            "init_cls": int(self.args.get("init_cls", 0)),
            "increment": int(self.args.get("increment", 0)),
            "batch_size": int(self.args.get("batch_size", 0)),
            "epochs": int(self.args.get("init_epoch", self.args.get("epochs", 0))),
            "lr": float(self.args.get("init_lr", self.args.get("lrate", 0.0))),
            "scheduler": self.args.get("scheduler", ""),
            "momentum": float(self.args.get("init_momentum", self.args.get("momentum", 0.0))),
            "weight_decay": float(
                self.args.get("init_weight_decay", self.args.get("weight_decay", 0.0))
            ),
            "fs_direction_mode": getattr(
                self, "_fs_direction_mode", self.args.get("fs_direction_mode", "")
            ),
            "actual_optimizer_class": getattr(self, "_last_optimizer_class", ""),
            "fs_base_optimizer": self.args.get("fs_base_optimizer", ""),
            "fs_alpha_mode": getattr(
                self, "_fs_alpha_mode", self.args.get("fs_alpha_mode", "")
            ),
            "fs_alpha": float(getattr(self, "_fs_alpha", self.args.get("fs_alpha", 0.0))),
            "fs_rho": float(getattr(self, "_fs_rho", self.args.get("fs_rho", 0.0))),
            "fs_sam_equivalent": bool(getattr(self, "_fs_sam_equivalent", False)),
            "fs_preserve_probe_rng": bool(
                getattr(self, "_fs_preserve_probe_rng", False)
            ),
            "stage_lambda": float(getattr(self, "_lambda_ewc", self.args.get("ewc_lambda", 0.0))),
            "fisher_mode": getattr(self, "_fisher_mode", self.args.get("fisher_mode", "")),
            "fisher_source": str(
                self.args.get("ewc_fisher_source", "heldout")
            ).lower(),
            "fisher_scope": "effective_delta_w",
            "fisher_max_batches": self.args.get("ewc_max_batches", ""),
            "ewc_normalize_fisher": bool(getattr(self, "_normalize_fisher", False)),
            "ewc_normalize_accumulated_fisher": bool(
                getattr(self, "_normalize_accumulated_fisher", False)
            ),
            "ewc_fisher_damping_beta": float(
                getattr(self, "_ewc_fisher_damping_beta", 0.0)
            ),
            "ewc_gamma": float(getattr(self, "_ewc_gamma", self.args.get("ewc_gamma", 1.0))),
            "fisher_tensor_count": int(len(fisher)),
            "fisher_trace": fisher_trace,
            "fisher_shapes": {key: list(value.shape) for key, value in sorted(fisher.items())},
            "producer_receipt": dict(
                getattr(self, "_task_producer_receipt", {}) or {}
            ),
        }
        split_info = dict(getattr(self, "_current_fisher_split_info", {}) or {})
        train_source_indices = [
            int(value) for value in split_info.get("train_source_indices", [])
        ]
        diagnostic_source_indices = [
            int(value) for value in split_info.get("val_source_indices", [])
        ]
        identity["producer_receipt"].update(
            {
                "split_seed": int(split_info.get("seed", -1)),
                "fit_samples": int(split_info.get("train_samples", 0)),
                "diagnostic_samples": int(split_info.get("val_samples", 0)),
                "fit_source_indices_sha256": self._canonical_sha256(
                    train_source_indices
                ),
                "diagnostic_source_indices_sha256": self._canonical_sha256(
                    diagnostic_source_indices
                ),
                "fit_diagnostic_disjoint": bool(
                    train_source_indices
                    and diagnostic_source_indices
                    and set(train_source_indices).isdisjoint(diagnostic_source_indices)
                ),
                "fisher_source": identity["fisher_source"],
            }
        )
        require_resume_sidecars = str(
            self.args.get("checkpoint_require_resume_sidecars", False)
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        if require_resume_sidecars:
            identity["artifact_sha256"] = validate_task_resume_sidecars(
                ckpt_dir, task_idx
            )
        else:
            checkpoint_path = os.path.join(
                ckpt_dir, f"task_{int(task_idx):02d}.pt"
            )
            sidecar_path = os.path.join(
                ckpt_dir, f"fisher_state_{int(task_idx)}.pt"
            )
            identity["artifact_sha256"] = {
                "task_checkpoint": (
                    self._file_sha256(checkpoint_path)
                    if os.path.isfile(checkpoint_path)
                    else ""
                ),
                "fisher_sidecar": (
                    self._file_sha256(sidecar_path)
                    if os.path.isfile(sidecar_path)
                    else ""
                ),
            }
        path = os.path.join(ckpt_dir, f"experiment_identity_{int(task_idx)}.json")
        tmp_path = path + f".tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(identity, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)

    def _validate_resume_experiment_identity(self, checkpoint_dir: str, task_idx: int) -> None:
        identity_required = str(
            self.args.get("resume_require_experiment_identity", False)
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        sidecars_required = str(
            self.args.get("resume_require_model_sidecars", False)
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        if not identity_required and not sidecars_required:
            return
        path = os.path.join(
            checkpoint_dir, f"experiment_identity_{int(task_idx)}.json"
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Required staged identity manifest is missing: {path}")
        with open(path, encoding="utf-8") as handle:
            identity = json.load(handle)
        if identity.get("status") != "complete":
            raise ValueError(f"Incomplete staged identity manifest: {path}")

        expected = {
            "task": int(task_idx),
            "dataset": self.args.get("dataset", ""),
            "backbone_type": self.args.get("backbone_type", ""),
            "lora_rank": int(self.args.get("lora_rank", getattr(self, "_rank", 0))),
            "seed": int(self.args.get("resume_expected_seed", -1)),
            "class_shuffle": False,
            "init_cls": int(self.args.get("init_cls", 0)),
            "increment": int(self.args.get("increment", 0)),
            "batch_size": int(self.args.get("batch_size", 0)),
            "epochs": int(self.args.get("resume_expected_epochs", 0)),
            "lr": float(self.args.get("init_lr", 0.0)),
            "scheduler": self.args.get("scheduler", ""),
            "momentum": float(self.args.get("init_momentum", 0.0)),
            "weight_decay": float(self.args.get("init_weight_decay", 0.0)),
            "fs_direction_mode": self.args.get(
                "resume_expected_fs_direction_mode", ""
            ),
            "actual_optimizer_class": self.args.get(
                "resume_expected_optimizer_class", ""
            ),
            "fs_alpha_mode": self.args.get("fs_alpha_mode", ""),
            "fs_alpha": float(self.args.get("resume_expected_fs_alpha", 0.0)),
            "fs_rho": float(self.args.get("fs_rho", 0.0)),
            "fs_preserve_probe_rng": True,
            "stage_lambda": 0.0,
            "fisher_mode": str(self.args.get("fisher_mode", "")),
            "fisher_scope": "effective_delta_w",
            "ewc_normalize_fisher": True,
            "ewc_normalize_accumulated_fisher": False,
            "ewc_fisher_damping_beta": float(
                self.args.get("ewc_fisher_damping_beta", 0.0)
            ),
            "ewc_gamma": float(self.args.get("ewc_gamma", 1.0)),
        }
        receipt_required = str(
            self.args.get("resume_require_producer_receipt", False)
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        if receipt_required:
            expected["fisher_source"] = str(
                self.args.get("ewc_fisher_source", "fit")
            ).lower()
        mismatches = {}
        for key, expected_value in expected.items():
            actual = identity.get(key)
            if isinstance(expected_value, float):
                try:
                    matches = bool(np.isclose(float(actual), expected_value, rtol=0.0, atol=1e-12))
                except (TypeError, ValueError):
                    matches = False
            else:
                matches = actual == expected_value
            if not matches:
                mismatches[key] = {"expected": expected_value, "actual": actual}
        if int(identity.get("fisher_tensor_count", 0)) <= 0 or float(
            identity.get("fisher_trace", 0.0)
        ) <= 0.0:
            mismatches["fisher_nonempty"] = {
                "expected": True,
                "actual": {
                    "count": identity.get("fisher_tensor_count"),
                    "trace": identity.get("fisher_trace"),
                },
            }
        if receipt_required:
            receipt = identity.get("producer_receipt", {})
            required_receipt_fields = (
                "origin_network_sha256",
                "origin_rng_sha256",
                "optimizer_groups_sha256",
                "producer_source_bundle_sha256",
                "producer_env_spec_hash",
                "producer_env_receipt_sha256",
                "frozen_protocol_sha256",
                "sampler_order_sha256",
                "fit_source_indices_sha256",
                "diagnostic_source_indices_sha256",
            )
            missing_receipt = [
                key for key in required_receipt_fields if not receipt.get(key)
            ]
            if missing_receipt:
                mismatches["producer_receipt_fields"] = {
                    "expected": list(required_receipt_fields),
                    "missing": missing_receipt,
                }
            if not bool(receipt.get("fit_diagnostic_disjoint", False)):
                mismatches["fit_diagnostic_disjoint"] = {
                    "expected": True,
                    "actual": receipt.get("fit_diagnostic_disjoint"),
                }
            expected_source_bundle = str(
                self.args.get("producer_source_bundle_sha256", "")
            )
            if receipt.get("producer_source_bundle_sha256") != expected_source_bundle:
                mismatches["producer_source_bundle_sha256"] = {
                    "expected": expected_source_bundle,
                    "actual": receipt.get("producer_source_bundle_sha256"),
                }
            expected_env_spec_hash = str(
                self.args.get("producer_env_spec_hash", "")
            )
            if receipt.get("producer_env_spec_hash") != expected_env_spec_hash:
                mismatches["producer_env_spec_hash"] = {
                    "expected": expected_env_spec_hash,
                    "actual": receipt.get("producer_env_spec_hash"),
                }
            expected_env_receipt = str(
                self.args.get("producer_env_receipt_sha256", "")
            )
            if receipt.get("producer_env_receipt_sha256") != expected_env_receipt:
                mismatches["producer_env_receipt_sha256"] = {
                    "expected": expected_env_receipt,
                    "actual": receipt.get("producer_env_receipt_sha256"),
                }
            expected_protocol = str(self.args.get("frozen_protocol_sha256", ""))
            if receipt.get("frozen_protocol_sha256") != expected_protocol:
                mismatches["frozen_protocol_sha256"] = {
                    "expected": expected_protocol,
                    "actual": receipt.get("frozen_protocol_sha256"),
                }
            artifact_hashes = identity.get("artifact_sha256", {})
            observed_artifacts = (
                validate_task_resume_sidecars(checkpoint_dir, task_idx)
                if sidecars_required
                else task_resume_artifact_sha256(checkpoint_dir, task_idx)
            )
            if artifact_hashes != observed_artifacts:
                mismatches["artifact_sha256"] = {
                    "expected": artifact_hashes,
                    "actual": observed_artifacts,
                }
            if int(identity.get("fisher_tensor_count", 0)) != 24:
                mismatches["fisher_tensor_count"] = {
                    "expected": 24,
                    "actual": identity.get("fisher_tensor_count"),
                }
            if not np.isclose(
                float(identity.get("fisher_trace", 0.0)),
                1.0,
                rtol=1e-5,
                atol=1e-6,
            ):
                mismatches["fisher_trace_normalized"] = {
                    "expected": 1.0,
                    "actual": identity.get("fisher_trace"),
                }
        if mismatches:
            raise ValueError(
                f"Staged experiment identity mismatch in {path}: {mismatches}"
            )
        self._log(
            f"[Resume][IdentityValidated] source={path} "
            f"direction={identity.get('fs_direction_mode')} "
            f"optimizer={identity.get('actual_optimizer_class')}"
        )

    def _load_auxiliary_task_state(self, checkpoint_dir: str, task_idx: int) -> None:
        require_fisher = str(
            self.args.get("resume_require_fisher_state", False)
        ).strip().lower() in {"1", "true", "yes", "y", "on"}
        fisher_path = os.path.join(checkpoint_dir, f"fisher_state_{int(task_idx)}.pt")
        if not os.path.exists(fisher_path):
            legacy_path = os.path.join(checkpoint_dir, f"fisher_state_{int(task_idx):02d}.pt")
            fisher_path = legacy_path if os.path.exists(legacy_path) else fisher_path
        if not os.path.exists(fisher_path):
            if require_fisher:
                raise FileNotFoundError(
                    "Required Fisher sidecar is missing for staged resume: "
                    f"{fisher_path}"
                )
            return

        state = torch.load(fisher_path, map_location="cpu", weights_only=False)
        if require_fisher:
            self._validate_fisher_checkpoint_state(
                state,
                source=fisher_path,
                require_shape_match=True,
            )
        fisher = state.get("fisher_past_delta", None)
        if fisher is not None and hasattr(self, "_fisher_past_delta"):
            self._fisher_past_delta = {
                key: value.detach().cpu().clone()
                for key, value in fisher.items()
            }
        if "fisher_past_trace_mass" in state and hasattr(self, "_fisher_past_trace_mass"):
            self._fisher_past_trace_mass = float(state["fisher_past_trace_mass"])
        if "last_task_new_fisher_stats" in state and hasattr(self, "_last_task_new_fisher_stats"):
            self._last_task_new_fisher_stats = dict(state["last_task_new_fisher_stats"])
        if "fisher_new_task_raw_trace" in state:
            self._fisher_new_task_raw_trace = float(state["fisher_new_task_raw_trace"])
        self._log(f"[Resume] Loaded Fisher state from {fisher_path}")

    def _validate_fisher_checkpoint_state(
        self,
        state,
        *,
        source: str,
        require_shape_match: bool,
    ) -> None:
        if not isinstance(state, dict):
            raise ValueError(f"Invalid Fisher state in {source}: expected dict")
        fisher = state.get("fisher_past_delta")
        if not isinstance(fisher, dict) or not fisher:
            raise ValueError(f"Invalid Fisher state in {source}: Fisher is empty")

        trace = 0.0
        shapes = {}
        for key, value in fisher.items():
            if not isinstance(key, str) or not torch.is_tensor(value) or value.numel() == 0:
                raise ValueError(
                    f"Invalid Fisher entry in {source}: key={key!r}, type={type(value).__name__}"
                )
            value_float = value.detach().float()
            if not bool(torch.isfinite(value_float).all().item()):
                raise ValueError(f"Non-finite Fisher tensor {key!r} in {source}")
            if bool((value_float < 0).any().item()):
                raise ValueError(f"Negative Fisher tensor {key!r} in {source}")
            trace += float(value_float.sum().item())
            shapes[key] = tuple(value.shape)
        if not np.isfinite(trace) or trace <= 0.0:
            raise ValueError(f"Invalid Fisher trace in {source}: {trace}")

        trace_mass = state.get("fisher_past_trace_mass", None)
        if trace_mass is not None:
            trace_mass = float(trace_mass)
            # This legacy accumulator is bookkeeping for one optional
            # accumulated-normalization path and legitimately remains zero
            # when that path is disabled.  The authoritative validity gate is
            # the positive trace computed directly from fisher_past_delta.
            if not np.isfinite(trace_mass) or trace_mass < 0.0:
                raise ValueError(
                    f"Invalid fisher_past_trace_mass in {source}: {trace_mass}"
                )

        if require_shape_match and hasattr(self, "_iter_delta_terms"):
            current_shapes = {
                key: tuple(delta.shape)
                for key, delta in self._iter_delta_terms(self._unwrap_network())
            }
            if set(shapes) != set(current_shapes):
                missing = sorted(set(current_shapes) - set(shapes))
                extra = sorted(set(shapes) - set(current_shapes))
                raise ValueError(
                    f"Fisher key mismatch in {source}: missing={missing}, extra={extra}"
                )
            mismatched = {
                key: (shapes[key], current_shapes[key])
                for key in shapes
                if shapes[key] != current_shapes[key]
            }
            if mismatched:
                raise ValueError(
                    f"Fisher shape mismatch in {source}: {mismatched}"
                )

        self._log(
            f"[Resume][FisherValidated] source={source} tensors={len(fisher)} "
            f"trace={trace:.6e} shape_match={bool(require_shape_match)}"
        )

    def _load_current_lora_factors(self, backbone, checkpoint_dir: str, task_idx: int) -> None:
        if backbone is None:
            raise ValueError("Cannot restore LoRA checkpoint: network has no backbone")

        path_a = os.path.join(checkpoint_dir, f"lora_w_a_{task_idx}.pt")
        path_b = os.path.join(checkpoint_dir, f"lora_w_b_{task_idx}.pt")
        if not (os.path.exists(path_a) and os.path.exists(path_b)):
            raise FileNotFoundError(
                f"LoRA checkpoint for task {task_idx} not found in {checkpoint_dir}"
            )

        saved_as = torch.load(path_a, map_location="cpu")
        saved_bs = torch.load(path_b, map_location="cpu")
        dst_as = getattr(backbone, "w_As", None)
        dst_bs = getattr(backbone, "w_Bs", None)
        if dst_as is None or dst_bs is None:
            raise ValueError("Current backbone does not expose w_As/w_Bs for resume")
        if len(dst_as) != len(saved_as) or len(dst_bs) != len(saved_bs):
            raise ValueError(
                "LoRA branch count mismatch while restoring task "
                f"{task_idx}: current=({len(dst_as)}, {len(dst_bs)}), "
                f"saved=({len(saved_as)}, {len(saved_bs)})"
            )

        for dst, src in zip(dst_as, saved_as):
            src_weight = src.weight if hasattr(src, "weight") else src
            dst.weight.data.copy_(src_weight.detach().to(dst.weight.device))
        for dst, src in zip(dst_bs, saved_bs):
            src_weight = src.weight if hasattr(src, "weight") else src
            dst.weight.data.copy_(src_weight.detach().to(dst.weight.device))

        # Keep the resumed adapter as the single trainable SeqLoRA adapter.
        # Historical saved_A/saved_B entries are eval-time artifacts for other
        # LoRA variants; using them here would double-count the cumulative A/B.
        for module in backbone.modules():
            if hasattr(module, "saved_A"):
                module.saved_A = {}
            if hasattr(module, "saved_B"):
                module.saved_B = {}
            if hasattr(module, "task_id"):
                module.task_id = 0
        if hasattr(backbone, "saved_A"):
            backbone.saved_A = {}
        if hasattr(backbone, "saved_B"):
            backbone.saved_B = {}
        if hasattr(backbone, "task_id"):
            backbone.task_id = 0

    def restore_task_snapshot(self, data_manager, task_idx: int) -> None:
        """Restore model state for a particular task for offline evaluation."""
        task_start, task_end = data_manager.get_task_class_range(task_idx)

        self._cur_task = task_idx + 1
        self._total_classes = task_end
        self._known_classes = task_end

        self._refresh_distributed_context()

        self._network.update_fc(self._total_classes)
        if hasattr(self._network, "load_fc"):
            self._network.load_fc(self.args["filepath"], task_idx)

        eval_backbone = self._build_eval_backbone(task_idx)

        network = self._unwrap_network()
        network.backbone = eval_backbone
        network.backbone.to(self._device)
        self._network = network
        self._prepare_network()

        batch_size = self.args.get("batch_size", 128)
        eval_num_workers = self.args.get("eval_num_workers", 8)

        train_classes = np.arange(task_start, task_end)
        train_dataset = data_manager.get_dataset(train_classes, source="train", mode="train")
        self.train_loader = data_manager.build_dataloader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=eval_num_workers,
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, task_end), source="test", mode="test"
        )
        self.test_loader = data_manager.build_dataloader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=eval_num_workers,
        )

        # Compute NME class means over all seen classes for offline evaluation
        try:
            self.compute_all_seen_class_means(data_manager)
        except Exception as _nme_exc:  # pylint: disable=broad-except
            logging.exception("[LoRA][restore] Failed to compute class means for NME: %s", _nme_exc)

    def _build_eval_backbone(self, task_idx: int) -> nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError
    
    # ------------------------------------------------------------------
    # Prototype/NME helpers
    # ------------------------------------------------------------------
    def compute_all_seen_class_means(self, data_manager) -> None:
        """Compute per-class prototypes (means) for all seen classes.

        - Uses training split for all seen classes [0, _total_classes)
        - Applies test transforms (mode="test") to avoid heavy augmentation
        - Normalizes features and prototypes (cosine-friendly)
        """
        if getattr(self, "_total_classes", 0) <= 0:
            return

        nb_classes = int(self._total_classes)
        feat_dim = int(self.feature_dim)
        class_means = np.zeros((nb_classes, feat_dim), dtype=np.float64)

        bs = int(self.args.get("eval_batch_size", self.args.get("batch_size", 128)))
        nw = int(self.args.get("nme_num_workers", self.args.get("eval_num_workers", 8)))

        for c in range(nb_classes):
            try:
                dataset_c = data_manager.get_dataset(
                    np.arange(c, c + 1), source="train", mode="test"
                )
                loader_c = data_manager.build_dataloader(
                    dataset_c,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=nw,
                )
                vectors, _ = self._extract_vectors(loader_c)  # np.ndarray [N, D]
                if vectors.size == 0:
                    continue
                # L2-normalize features then mean; normalize prototype as well
                vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + 1e-12)).T
                mu = np.mean(vectors, axis=0)
                norm = np.linalg.norm(mu) + 1e-12
                class_means[c, :] = (mu / norm).astype(np.float64)
            except Exception as _exc:  # pylint: disable=broad-except
                logging.exception("[LoRA][NME] Failed to compute mean for class %d: %s", c, _exc)

        # Attach for BaseLearner.eval_task() to use
        self._class_means = class_means

    
