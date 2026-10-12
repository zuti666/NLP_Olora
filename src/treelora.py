"""TreeLoRA's published implementation, adapted to a task-at-a-time trainer.

Algorithm source: https://github.com/ZinYY/TreeLoRA at commit
1c7260c42b34e1961283797c742f08b9c3842501, ``utils/kd_lora_tree.py`` and
``model/Regular/Tree_LoRA.py`` (vendored under references/treelora_upstream).

Despite the upstream names "grad" and "gradient-similarity", its input is the
stack of trainable LoRA A *parameters*, before backward, not their gradients.
There is no Fisher estimation, adapter branching, rank merging, or inference
routing. Several surprising upstream operations are deliberately preserved;
changing them would define a different experimental method.
"""

import math

import torch


UPSTREAM_COMMIT = "1c7260c42b34e1961283797c742f08b9c3842501"


def _snapshot(value):
    """Return independent, CPU-only tensors suitable for a sidecar checkpoint."""
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return value


class _KDTreeNode:
    def __init__(self, task_indices, depth, grads_tensor, lora_depth):
        self.task_indices = list(task_indices)
        self.depth = depth
        self.lora_depth = lora_depth
        self.left = self.right = None
        self.is_leaf = depth >= lora_depth or len(task_indices) <= 1
        self.mean_vector = None
        self.median_similarity = None
        if self.is_leaf:
            return

        current_grads = grads_tensor[task_indices, depth, :]
        self.mean_vector = current_grads.mean(dim=0)
        similarities = torch.mv(current_grads, self.mean_vector)
        self.median_similarity = torch.median(similarities).item()
        left = [task_indices[i] for i in range(len(task_indices))
                if similarities[i].item() >= self.median_similarity]
        right = [task_indices[i] for i in range(len(task_indices))
                 if similarities[i].item() < self.median_similarity]
        if not left or not right:
            midpoint = len(task_indices) // 2
            left, right = task_indices[:midpoint], task_indices[midpoint:]
        self.left = _KDTreeNode(left, depth + 1, grads_tensor, lora_depth)
        self.right = _KDTreeNode(right, depth + 1, grads_tensor, lora_depth)

    def state_dict(self):
        return {
            "task_indices": self.task_indices[:],
            "depth": self.depth,
            "lora_depth": self.lora_depth,
            "is_leaf": self.is_leaf,
            "mean_vector": _snapshot(self.mean_vector),
            "median_similarity": self.median_similarity,
            "left": self.left.state_dict() if self.left else None,
            "right": self.right.state_dict() if self.right else None,
        }

    @classmethod
    def from_state_dict(cls, state):
        if state is None:
            return None
        node = cls.__new__(cls)
        for key in ("task_indices", "depth", "lora_depth", "is_leaf",
                    "mean_vector", "median_similarity"):
            setattr(node, key, _snapshot(state[key]))
        node.left = cls.from_state_dict(state["left"])
        node.right = cls.from_state_dict(state["right"])
        return node


class TreeLoRARegularizer:
    """Stateful, source-faithful regularizer for a fixed-rank sequential LoRA.

    Call ``begin_epoch`` at every epoch boundary, ``loss`` once per training
    microbatch before backward, and ``end_task`` after the final epoch. The
    result of ``loss`` is the complete loss to backpropagate. Reuse the same
    trainable A/B adapter across tasks; save this object's state alongside it.

    Sampling uses the global PyTorch RNG, as upstream does. RNG checkpointing
    belongs to the caller (e.g. Trainer's RNG checkpoint), not this state dict.
    """

    FORMAT_VERSION = 1

    def __init__(self, reg=0.5, num_tasks=4):
        if not math.isfinite(reg) or reg < 0:
            raise ValueError("TreeLoRA reg must be finite and nonnegative")
        if int(num_tasks) != num_tasks or num_tasks < 1:
            raise ValueError("TreeLoRA num_tasks must be a positive integer")
        self.reg = float(reg)
        self.num_tasks = int(num_tasks)
        self.history = []
        self.kd_tree_root = None
        self.current_grad = None
        self.all_grad = None
        self.sim = None
        self.num_of_selected = None
        self.tmp_rounds = -1
        self.total_rounds = 0
        self.tmp_reg = 0.0
        self.microbatches_seen = 0
        self.epochs_seen = 0
        self.epoch_active = False
        self.last_selection = None
        self.last_reg_loss = 0.0
        self.last_task_loss = None

    @property
    def task_id(self):
        return len(self.history)

    def begin_epoch(self, total_microbatches):
        if int(total_microbatches) != total_microbatches or total_microbatches < 1:
            raise ValueError("TreeLoRA requires a positive epoch microbatch count")
        if self.task_id >= self.num_tasks:
            raise ValueError("TreeLoRA has already completed all configured tasks")
        # The upstream reset discards previous epochs' trajectories and counts.
        self.current_grad = self.all_grad = None
        self.sim = self.num_of_selected = None
        self.tmp_rounds = -1
        self.total_rounds = int(total_microbatches)
        self.tmp_reg = 0.0
        self.last_selection = None
        self.last_reg_loss = 0.0
        self.epoch_active = True
        self.epochs_seen += 1

    def _move_state(self, current):
        for index, past in enumerate(self.history):
            if past.shape != current.shape or past.dtype != current.dtype:
                raise ValueError("TreeLoRA A shape/dtype differs from saved history")
            if past.device != current.device:
                self.history[index] = past.to(current.device)
        for name in ("current_grad", "all_grad", "sim", "num_of_selected"):
            value = getattr(self, name)
            if value is not None and value.device != current.device:
                setattr(self, name, value.to(current.device))

    def _insert(self, current):
        # Preserve upstream's loop bug: the *whole stack* is accumulated once
        # for every LoRA module, multiplying the trajectory by module count.
        # Do not replace this with a single average or change reduction order.
        for _ in range(len(current)):
            if self.current_grad is None:
                self.current_grad = current.detach() * 1.0 / self.total_rounds
            else:
                self.current_grad += current.detach() * (1.0 / self.total_rounds)

    def _search(self, device):
        task_id = self.task_id
        if self.all_grad is None:
            self.all_grad = torch.stack(self.history, dim=0).to(device, non_blocking=True)
            if self.sim is None:
                self.sim = torch.zeros((task_id, self.all_grad.shape[1]), device=device)
                self.num_of_selected = torch.zeros(
                    self.num_tasks, self.all_grad.shape[1], device=device
                )

        sim = self.sim.clone()
        valid = self.num_of_selected[:task_id, :] > 0
        sim[valid] = sim[valid] / self.num_of_selected[:task_id, :][valid]
        sim -= (
            1.0 / torch.sqrt(2 * self.num_of_selected[:task_id, :] + 1e-5)
            * math.sqrt(math.log(
                2 * self.total_rounds * (self.tmp_rounds + 1) * (self.tmp_rounds + 2)
            ))
        )
        sim = -sim
        # Upstream adds (rather than subtracts) the minimum. This affects the
        # subsequent branch weighting, so it is not an interchangeable shift.
        sim += torch.min(sim)
        first_idx = torch.multinomial(
            torch.softmax(torch.sum(sim, dim=1), dim=0),
            num_samples=1, replacement=True,
        ).item()
        if self.kd_tree_root is not None and self.kd_tree_root.left is not None:
            branch = (self.kd_tree_root.left
                      if first_idx in self.kd_tree_root.left.task_indices
                      else self.kd_tree_root.right)
            similarity = (branch.median_similarity
                          if branch.median_similarity is not None else 1.0)
            sim[branch.task_indices] *= min(similarity, 1.5)

        sim = sim / (torch.max(sim) - torch.min(sim) + 1e-5)
        sim[task_id:, :] = -torch.inf
        selected = torch.multinomial(
            torch.softmax(sim, dim=0).T, num_samples=1, replacement=True
        ).reshape(-1)
        self.num_of_selected[selected, torch.arange(sim.shape[1])] += 1
        for depth_idx, prev_id in enumerate(selected):
            # Preserve upstream's negative L1 update, despite its "distance"
            # comments. Reversing the sign changes future task selection.
            self.sim[prev_id, depth_idx] -= torch.sum(torch.abs(
                self.current_grad[depth_idx] - self.all_grad[prev_id, depth_idx]
            )).item()
        return selected

    def loss(self, current_A_stack, task_loss):
        if not self.epoch_active:
            raise RuntimeError("Call TreeLoRA.begin_epoch before loss")
        if current_A_stack.ndim != 2 or current_A_stack.shape[0] == 0:
            raise ValueError("TreeLoRA expects [LoRA modules, flattened A parameters]")
        self.microbatches_seen += 1
        self.last_task_loss = float(task_loss.detach().item())
        if self.reg == 0:
            return task_loss
        self._move_state(current_A_stack)
        self.tmp_rounds += 1
        self.tmp_reg = self.reg * self.tmp_rounds / self.total_rounds
        self._insert(current_A_stack)
        if self.task_id == 0:
            return task_loss

        selected = self._search(current_A_stack.device)
        raw_reg = None
        for depth_idx, prev_id in enumerate(selected):
            term = -(current_A_stack[depth_idx] * self.all_grad[prev_id][depth_idx]).sum()
            raw_reg = term if raw_reg is None else raw_reg + term
        # This signed detached denominator and the final subtraction are both
        # from upstream. abs(), clamp(), or addition would change the gradient.
        reg_loss = (raw_reg / (raw_reg.detach().clone() + 1e-5)
                    * task_loss.detach().clone() * self.tmp_reg)
        self.last_selection = selected.detach().clone()
        self.last_reg_loss = float(reg_loss.detach().item())
        return task_loss - reg_loss

    def end_task(self):
        if not self.epoch_active:
            raise RuntimeError("TreeLoRA task is not active or has already ended")
        if self.reg == 0:
            self.epoch_active = False
            return
        if self.current_grad is None:
            raise RuntimeError("Cannot finish a TreeLoRA task without training microbatches")
        self.history.append(self.current_grad.detach().clone())
        grads_tensor = torch.stack(self.history).clone()
        # Only tree construction uses adjacent task differences. Regularization
        # and bandit distances retain the original cumulative A trajectories.
        for index in range(grads_tensor.shape[0] - 1, 0, -1):
            grads_tensor[index] = grads_tensor[index] - grads_tensor[index - 1]
        self.kd_tree_root = _KDTreeNode(
            list(range(self.task_id)), 0, grads_tensor, self.current_grad.shape[0]
        )
        self.epoch_active = False

    def diagnostics(self):
        return {
            "upstream_commit": UPSTREAM_COMMIT,
            "task_id": self.task_id,
            "history_count": len(self.history),
            "epoch_step": self.tmp_rounds,
            "epoch_microbatches": self.total_rounds,
            "microbatches_seen": self.microbatches_seen,
            "epochs_seen": self.epochs_seen,
            "effective_reg": self.tmp_reg,
            "last_reg_loss": self.last_reg_loss,
            "last_task_loss": self.last_task_loss,
            "last_selected": (self.last_selection.detach().cpu().tolist()
                              if self.last_selection is not None else None),
            "selected_counts": (self.num_of_selected.detach().cpu().tolist()
                                if self.num_of_selected is not None else None),
        }

    def state_dict(self):
        state = {"format_version": self.FORMAT_VERSION, "upstream_commit": UPSTREAM_COMMIT}
        for key in (
            "reg", "num_tasks", "history", "current_grad", "all_grad", "sim",
            "num_of_selected", "tmp_rounds", "total_rounds", "tmp_reg",
            "microbatches_seen", "epochs_seen", "epoch_active", "last_selection",
            "last_reg_loss", "last_task_loss",
        ):
            state[key] = _snapshot(getattr(self, key))
        state["kd_tree_root"] = self.kd_tree_root.state_dict() if self.kd_tree_root else None
        return state

    def load_state_dict(self, state):
        if state.get("format_version") != self.FORMAT_VERSION:
            raise ValueError("Unsupported TreeLoRA sidecar format version")
        if state.get("upstream_commit") != UPSTREAM_COMMIT:
            raise ValueError("TreeLoRA sidecar uses a different upstream algorithm")
        if state["reg"] != self.reg or state["num_tasks"] != self.num_tasks:
            raise ValueError("TreeLoRA sidecar reg/num_tasks do not match this run")
        if len(state["history"]) > self.num_tasks:
            raise ValueError("TreeLoRA sidecar contains too many completed tasks")
        for key in (
            "history", "current_grad", "all_grad", "sim", "num_of_selected",
            "tmp_rounds", "total_rounds", "tmp_reg", "microbatches_seen",
            "epochs_seen", "epoch_active", "last_selection", "last_reg_loss",
            "last_task_loss",
        ):
            setattr(self, key, _snapshot(state[key]))
        self.kd_tree_root = _KDTreeNode.from_state_dict(state["kd_tree_root"])
