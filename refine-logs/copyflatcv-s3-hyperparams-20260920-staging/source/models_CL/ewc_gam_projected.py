import logging
import math
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models_CL.baseLearner import BaseLearner
from optimer_PerturabtionType.gam import GAM
from utils.inc_net import IncrementalNet
from utils.toolkit import tensor2numpy

num_workers = 1

_FISHER_MODE_ALIASES = {"full": "batch_empirical"}
_BATCH_FISHER_MODES = {
    "batch_empirical",
    "max_pred_batch_empirical",
    "lr_batch_empirical",
}
_PROTOTYPE_FISHER_MODES = {"prototype", "prototype_all"}
_SUPPORTED_FISHER_MODES = _BATCH_FISHER_MODES | _PROTOTYPE_FISHER_MODES


def _canonical_fisher_mode(mode):
    mode = str(mode).strip().lower()
    return _FISHER_MODE_ALIASES.get(mode, mode)


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


class EWCGAMProjected(BaseLearner):
    """
    Full-model EWC + GAM with projected gradient update.

    Per-batch step (task 1+):
        1. Compute EWC gradient g_ewc from accumulated Fisher.
        2. Run GAM closure on current-task CE only → g_gam.
        3. Project:
               g_final = g_gam - <g_gam, g_ewc> / (||g_ewc||² + ε) * g_ewc + g_ewc

    Uses the same IncrementalNet backbone/head path as EWCon_Probe.
    Fisher is computed over trainable backbone parameters (fc excluded; no LoRA).
    Online accumulation:  Ω_t = γ · Ω_{t-1} + F̂_t
    """

    def __init__(self, args):
        super().__init__(args)
        self.topk = 1
        self._network = IncrementalNet(args, True)

        self.gamma = float(args.get("gamma", 1.0))
        self.ewc_weight = float(args.get("lambda", 0.0))
        self.fisher_project_eps = float(args.get("fisher_project_eps", 1e-12))

        # "euclidean" (default) | "fisher" (Ω-weighted inner product)
        self.projection_metric = str(args.get("projection_metric", "euclidean"))

        self.omega = {}       # param name -> diagonal Fisher importance
        self.theta_star = {}  # param name -> reference params from previous task
        self.count_updates = 0

        # Fisher computation mode:
        #   "batch_empirical"          – CE(logits, y), batch-gradient-square proxy
        #   "max_pred_batch_empirical" – CE(logits, argmax(logits)), ref max_pred
        #   "lr_batch_empirical"       – CE(-logits, y), logit-reversed variant
        #   "prototype"                – K representative images per class, current task
        #   "prototype_all"            – K representative images per class, all seen tasks
        #   "full"                     – legacy alias of "batch_empirical"
        raw_fisher_mode = str(args.get("fisher_mode", "batch_empirical")).lower()
        self.fisher_mode = _canonical_fisher_mode(raw_fisher_mode)
        assert self.fisher_mode in _SUPPORTED_FISHER_MODES, (
            f"Unknown fisher_mode={raw_fisher_mode!r}. "
            "Expected 'batch_empirical', 'max_pred_batch_empirical', "
            "'lr_batch_empirical', 'prototype', 'prototype_all', "
            "or legacy alias 'full'."
        )
        # Number of representative images to store per class for prototype modes.
        self.prototype_per_class = int(args.get("prototype_per_class", 20))
        # class_id -> Tensor[K, C, H, W] stored on CPU
        self.prototype_images: dict = {}

        self._compute_full_fisher = _as_bool(args.get("compute_full_fisher", False))
        self._full_fisher_max_batches = args.get("full_fisher_max_batches", None)
        if self._full_fisher_max_batches is not None:
            self._full_fisher_max_batches = int(self._full_fisher_max_batches)
        self._data_manager = None
        self._last_full_fisher_diff: dict = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def after_task(self):
        print("=== Computing Fisher Information ===")
        net = self._net()
        task_start, task_end = self._task_range()

        if self.fisher_mode in _BATCH_FISHER_MODES:
            fisher = self._compute_fisher(
                net, task_start, task_end, mode=self.fisher_mode
            )
            if self.count_updates == 0:
                self.omega = {name: f.clone() for name, f in fisher.items()}
            else:
                for name, f in fisher.items():
                    prev = self.omega.get(name, torch.zeros_like(f))
                    self.omega[name] = self.gamma * prev + f

        elif self.fisher_mode == "prototype":
            # Build prototypes for the current task, then compute Fisher on them.
            # Accumulation structure is identical to "full" mode (supports gamma).
            self._build_prototype_set(net, task_start, task_end)
            fisher = self._compute_fisher_from_loader(
                net,
                self._make_prototype_loader(range(task_start, task_end)),
                all_classes=self._total_classes,
            )
            if self.count_updates == 0:
                self.omega = {name: f.clone() for name, f in fisher.items()}
            else:
                for name, f in fisher.items():
                    prev = self.omega.get(name, torch.zeros_like(f))
                    self.omega[name] = self.gamma * prev + f

        else:  # prototype_all
            # Build prototypes for the current task, then recompute Fisher over
            # ALL accumulated prototype images (all seen tasks) in one pass.
            # omega is fully replaced — no gamma accumulation needed.
            self._build_prototype_set(net, task_start, task_end)
            fisher = self._compute_fisher_from_loader(
                net,
                self._make_prototype_loader(range(0, self._total_classes)),
                all_classes=self._total_classes,
            )
            self.omega = {name: f.clone() for name, f in fisher.items()}

        self.count_updates += 1

        self.theta_star = {
            name: param.detach().clone()
            for name, param in net.named_parameters()
            if param.requires_grad and not name.startswith("fc.")
        }

        self._known_classes = self._total_classes

        if self._compute_full_fisher:
            full_diff = self._compute_full_fisher_diff_metrics(net)
            if full_diff:
                full_diff["task"] = int(self._cur_task)
                full_diff["method_type"] = "ewc_gam_projected"
                self._last_full_fisher_diff = full_diff
                logging.info("[EWCGAMProjected][FullFisherDiff] task=%d %s",
                             self._cur_task, full_diff)

    def incremental_train(self, data_manager):
        self._data_manager = data_manager
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def _train(self, train_loader, test_loader):
        # Resolve task-aware hyperparameters
        if self._cur_task == 0:
            self._run_epochs    = int(self.args.get("init_epoch", 20))
            self._run_lr        = float(self.args.get("init_lr", 0.01))
            self._run_wd        = float(self.args.get("init_weight_decay", 0.0))
            self._run_momentum  = float(self.args.get("init_momentum", 0.9))
            self._run_milestones = list(self.args.get("init_milestones", []))
            self._run_lr_decay  = float(self.args.get("init_lr_decay", 0.1))
        else:
            self._run_epochs    = int(self.args.get("epochs", 20))
            self._run_lr        = float(self.args.get("lrate", 0.01))
            self._run_wd        = float(self.args.get("weight_decay", 0.0))
            self._run_momentum  = float(self.args.get("momentum", 0.9))
            self._run_milestones = list(self.args.get("milestones", []))
            self._run_lr_decay  = float(self.args.get("lrate_decay", 0.1))
        self._run_scheduler = str(self.args.get("scheduler", "cosine"))

        self._network.to(self._device)

        optimizer, scheduler = self._build_gam_optimizer()
        self._train_function(train_loader, test_loader, optimizer, scheduler)

    def _build_gam_optimizer(self):
        param_groups = [
            {
                "params": [p for p in self._network.parameters() if p.requires_grad],
                "lr": self._run_lr,
                "weight_decay": self._run_wd,
            }
        ]

        base_name = str(self.args.get("gam_base_optimizer", "adam")).lower()
        if base_name == "sgd":
            base_optimizer = optim.SGD(
                param_groups,
                lr=self._run_lr,
                momentum=self._run_momentum,
                weight_decay=self._run_wd,
            )
        elif base_name == "adamw":
            base_optimizer = optim.AdamW(
                param_groups,
                lr=self._run_lr,
                weight_decay=self._run_wd,
                betas=(float(self.args.get("adam_beta1", 0.9)), float(self.args.get("adam_beta2", 0.999))),
            )
        else:  # default: adam
            base_optimizer = optim.Adam(
                param_groups,
                lr=self._run_lr,
                weight_decay=self._run_wd,
                betas=(float(self.args.get("adam_beta1", 0.9)), float(self.args.get("adam_beta2", 0.999))),
            )

        gam_args = SimpleNamespace(
            grad_beta_1=float(self.args.get("grad_beta_1", 1.0)),
            grad_beta_2=float(self.args.get("grad_beta_2", 1.0)),
            grad_beta_3=float(self.args.get("grad_beta_3", 1.0)),
            grad_gamma=float(self.args.get("grad_gamma", 0.03)),
        )
        optimizer = GAM(
            param_groups,
            base_optimizer,
            model=self._network,
            grad_rho=float(self.args.get("gam_grad_rho", 0.02)),
            grad_norm_rho=float(self.args.get("gam_grad_norm_rho", 0.2)),
            adaptive=_as_bool(self.args.get("gam_adaptive", False)),
            perturb_eps=float(self.args.get("gam_perturb_eps", 1e-12)),
            args=gam_args,
            grad_reduce=str(self.args.get("gam_grad_reduce", "mean")),
        )

        scheduler_type = self._run_scheduler
        if scheduler_type == "constant":
            scheduler = None
        elif scheduler_type == "cosine":
            min_lr_ratio = float(self.args.get("cosine_min_lr_ratio", 0.0))
            if min_lr_ratio > 0:
                min_lr_ratio = min(min_lr_ratio, 1.0)
                def lr_lambda(epoch):
                    cosine = 0.5 * (1.0 + math.cos(math.pi * epoch / self._run_epochs))
                    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine
                scheduler = optim.lr_scheduler.LambdaLR(base_optimizer, lr_lambda=lr_lambda)
            else:
                eta_min = float(self.args.get("cosine_eta_min", 0.0))
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    base_optimizer, T_max=self._run_epochs, eta_min=eta_min
                )
        elif scheduler_type in {"steplr", "multisteplr"}:
            scheduler = optim.lr_scheduler.MultiStepLR(
                base_optimizer, milestones=self._run_milestones, gamma=self._run_lr_decay
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_type!r}")

        return optimizer, scheduler

    def _train_function(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self._run_epochs))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            ce_losses = 0.0
            ewc_losses = 0.0
            ewc_count = 0
            correct, total = 0, 0

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                mask = (targets >= self._known_classes).nonzero().view(-1)
                if mask.numel() == 0:
                    continue
                inputs = torch.index_select(inputs, 0, mask)
                local_targets = torch.index_select(targets, 0, mask) - self._known_classes

                logits, loss_value, log_dict = self._step_projected(optimizer, inputs, local_targets)
                losses += loss_value
                ce_losses += float(log_dict.get("ce", 0.0))
                ewc_losses += float(log_dict.get("ewc", 0.0))
                ewc_count += 1

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(local_targets.expand_as(preds)).cpu().sum()
                total += len(local_targets)

            if scheduler is not None:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2) if total > 0 else 0.0
            avg_loss = losses / max(1, len(train_loader))
            avg_ce = ce_losses / max(1, ewc_count)
            avg_ewc = ewc_losses / max(1, ewc_count)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, CE {:.3f}, EWC {:.3e}".format(
                    self._cur_task, epoch + 1, self._run_epochs,
                    avg_loss, train_acc, test_acc, avg_ce, avg_ewc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, CE {:.3f}, EWC {:.3e}".format(
                    self._cur_task, epoch + 1, self._run_epochs,
                    avg_loss, train_acc, avg_ce, avg_ewc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    # ------------------------------------------------------------------
    # Projected GAM + EWC step
    # ------------------------------------------------------------------

    def _compute_task_loss(self, inputs, local_targets):
        """CE loss on current-task logits. local_targets are already shifted."""
        full_logits = self._network(inputs)["logits"]
        task_logits = full_logits[:, self._known_classes:]
        ce_loss = F.cross_entropy(task_logits, local_targets)
        return task_logits, ce_loss

    def _step_projected(self, optimizer, inputs, local_targets):
        params = _flatten_optimizer_params(optimizer)
        ewc_grad_cache = self._compute_ewc_grad_cache(params)
        log_cache = {"ce": 0.0, "ewc": 0.0}

        optimizer.set_direction_projector(
            self._build_projected_ewc_projector(ewc_grad_cache)
        )

        def closure():
            optimizer.zero_grad()
            logits, ce_loss = self._compute_task_loss(inputs, local_targets)
            ce_loss.backward()
            log_cache["ce"] = float(ce_loss.detach().item())
            return logits, ce_loss.detach()

        try:
            logits, loss_value = optimizer.step(closure=closure)
        finally:
            optimizer.set_direction_projector(None)

        ewc_loss_value = float(self._compute_ewc_loss().detach().item())
        log_cache["ewc"] = ewc_loss_value

        return logits.detach(), float(loss_value) + ewc_loss_value, log_cache

    def _compute_ewc_loss(self):
        if self.count_updates == 0 or not self.omega:
            return torch.zeros((), device=self._device)

        net = self._net()
        penalty = torch.tensor(0., device=self._device)
        for name, param in net.named_parameters():
            if name in self.omega and name in self.theta_star:
                ref = self.theta_star[name].to(device=param.device, dtype=param.dtype)
                omega = self.omega[name].to(device=param.device, dtype=param.dtype)
                if ref.shape != param.shape or omega.shape != param.shape:
                    continue
                diff = param - ref
                penalty = penalty + (omega * diff.pow(2)).sum()
        return 0.5 * self.ewc_weight * penalty

    def _compute_ewc_grad_cache(self, params):
        """Compute gradient of EWC loss w.r.t. each optimizer param."""
        if self.count_updates == 0 or not self.omega:
            return {}

        with torch.enable_grad():
            ewc_loss = self._compute_ewc_loss()

        grads = torch.autograd.grad(
            ewc_loss, params, allow_unused=True, retain_graph=False
        )
        return {
            id(param): grad.detach().clone()
            for param, grad in zip(params, grads)
            if grad is not None
        }

    def _build_projected_ewc_projector(self, ewc_grad_cache):
        eps = self.fisher_project_eps
        use_fisher_proj = (
            self.projection_metric == "fisher"
            and self.count_updates > 0
            and bool(self.omega)
        )

        param_omega_map = {}
        if use_fisher_proj:
            net = self._net()
            for name, param in net.named_parameters():
                if param.requires_grad and name in self.omega:
                    param_omega_map[id(param)] = self.omega[name]

        def projector(param_groups):
            for group in param_groups:
                for param in group["params"]:
                    if not param.requires_grad or param.grad is None:
                        continue

                    g_gam = param.grad.detach().clone()
                    g_ewc = ewc_grad_cache.get(id(param))

                    if g_ewc is None:
                        g_final = g_gam
                    elif use_fisher_proj and id(param) in param_omega_map:
                        omega = param_omega_map[id(param)].to(device=g_gam.device, dtype=g_gam.dtype)
                        omega_ewc = omega * g_ewc
                        num = (omega_ewc * g_gam).sum()
                        den = (omega_ewc * g_ewc).sum() + eps
                        coeff = num / den
                        g_proj = g_gam - coeff * g_ewc
                        g_final = g_proj + g_ewc
                    else:
                        denom = (g_ewc * g_ewc).sum() + eps
                        coeff = (g_gam * g_ewc).sum() / denom
                        g_proj = g_gam - coeff * g_ewc
                        g_final = g_proj + g_ewc

                    param.grad.data.copy_(g_final)

        return projector

    # ------------------------------------------------------------------
    # Fisher computation
    # ------------------------------------------------------------------

    def _fisher_loss_from_logits(self, logits, targets, mode):
        if mode == "batch_empirical":
            return F.cross_entropy(logits, targets)
        if mode == "max_pred_batch_empirical":
            pseudo_targets = logits.detach().argmax(dim=1)
            return F.cross_entropy(logits, pseudo_targets)
        if mode == "lr_batch_empirical":
            return F.cross_entropy(-logits, targets)
        raise ValueError(f"Unsupported Fisher mode for logits loss: {mode!r}")

    def _compute_fisher(self, net, task_start, task_end, mode="batch_empirical"):
        """Batch empirical diagonal Fisher on backbone parameters (fc excluded)."""
        net.eval()
        fisher = {
            name: torch.zeros_like(param)
            for name, param in net.named_parameters()
            if param.requires_grad and not name.startswith("fc.")
        }
        num_samples = 0

        for _, inputs, targets in tqdm(self.train_loader, desc="Computing Fisher"):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            mask = ((targets >= task_start) & (targets < task_end)).nonzero().view(-1)
            if mask.numel() == 0:
                continue
            inputs = torch.index_select(inputs, 0, mask)
            local_targets = torch.index_select(targets, 0, mask) - task_start

            net.zero_grad()
            logits = net(inputs)["logits"]
            task_logits = logits[:, task_start:task_end]
            loss = self._fisher_loss_from_logits(task_logits, local_targets, mode)
            loss.backward()

            batch_size = int(inputs.size(0))
            for name, param in net.named_parameters():
                if param.requires_grad and param.grad is not None and name in fisher:
                    fisher[name] += param.grad.detach().pow(2) * batch_size
            num_samples += batch_size

        if num_samples > 0:
            for name in fisher:
                fisher[name] /= num_samples

        net.train()
        return fisher

    # ------------------------------------------------------------------
    # Prototype-based Fisher helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _build_prototype_set(self, net, task_start, task_end):
        """Store K representative images per class for the current task.

        For each class, computes the mean feature vector (prototype) over all
        training samples, then selects the K images closest to that prototype
        (L2 distance in feature space) and stores them in self.prototype_images.
        """
        K = self.prototype_per_class
        net.eval()

        # Accumulate (image, feature) pairs per class.
        class_imgs: dict = {}    # class_id -> list[Tensor (C,H,W)]
        class_feats: dict = {}   # class_id -> list[Tensor (D,)]

        for _, inputs, targets in self.train_loader:
            mask = ((targets >= task_start) & (targets < task_end)).nonzero().view(-1)
            if mask.numel() == 0:
                continue
            inputs = torch.index_select(inputs, 0, mask)
            targets_m = torch.index_select(targets, 0, mask)

            feats = net.extract_vector(inputs.to(self._device)).cpu()

            for img, feat, cls in zip(inputs.cpu(), feats, targets_m.cpu()):
                cid = int(cls.item())
                class_imgs.setdefault(cid, []).append(img)
                class_feats.setdefault(cid, []).append(feat)

        # For each class select K images nearest to the class mean feature.
        for cid in class_imgs:
            imgs_t = torch.stack(class_imgs[cid])   # (N, C, H, W)
            feats_t = torch.stack(class_feats[cid]) # (N, D)
            mean_feat = feats_t.mean(dim=0)         # (D,)
            dists = (feats_t - mean_feat).pow(2).sum(dim=1)  # (N,)
            n_sel = min(K, imgs_t.size(0))
            indices = dists.topk(n_sel, largest=False).indices
            self.prototype_images[cid] = imgs_t[indices]    # (K, C, H, W)

        net.train()
        logging.info(
            "[Prototype] Built prototype set: %d classes, %d imgs/class",
            len(class_imgs), K,
        )

    def _make_prototype_loader(self, class_ids):
        """Build a DataLoader from stored prototype images for the given class IDs."""
        from torch.utils.data import TensorDataset, DataLoader as _DL

        all_imgs, all_labels = [], []
        for cid in class_ids:
            if cid not in self.prototype_images:
                continue
            imgs = self.prototype_images[cid]   # (K, C, H, W)
            all_imgs.append(imgs)
            all_labels.append(torch.full((imgs.size(0),), cid, dtype=torch.long))

        if not all_imgs:
            return None

        imgs_t = torch.cat(all_imgs, dim=0)
        labels_t = torch.cat(all_labels, dim=0)
        dataset = TensorDataset(imgs_t, labels_t)
        batch_size = int(self.args.get("batch_size", 64))
        return _DL(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    def _compute_fisher_from_loader(
        self, net, loader, all_classes: int, mode="batch_empirical"
    ):
        """Batch empirical diagonal Fisher (backbone only) from an arbitrary loader.

        ``loader`` yields (inputs, targets) batches where targets are absolute
        class indices.  ``all_classes`` is the number of output logits to use.
        """
        if loader is None:
            return {}

        net.eval()
        fisher = {
            name: torch.zeros_like(param)
            for name, param in net.named_parameters()
            if param.requires_grad and not name.startswith("fc.")
        }
        num_samples = 0

        for inputs, targets in tqdm(loader, desc="Computing Prototype Fisher"):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            net.zero_grad()
            logits = net(inputs)["logits"][:, :all_classes]
            loss = self._fisher_loss_from_logits(logits, targets, mode)
            loss.backward()

            bs = int(inputs.size(0))
            for name, param in net.named_parameters():
                if param.requires_grad and param.grad is not None and name in fisher:
                    fisher[name] += param.grad.detach().pow(2) * bs
            num_samples += bs

        if num_samples > 0:
            for name in fisher:
                fisher[name] /= num_samples

        net.train()
        logging.info(
            "[Prototype Fisher] Computed over %d prototype samples", num_samples
        )
        return fisher

    # ------------------------------------------------------------------
    # Full Fisher vs accumulated Fisher comparison
    # ------------------------------------------------------------------

    def _compute_fisher_all_seen(self, net):
        """Batch-empirical diagonal Fisher over all seen tasks' training data.

        Uses the standard squared-batch-gradient proxy, same estimator used for
        the per-task Fisher that feeds into self.omega.  No batch cap — uses the
        full dataset so the estimate is as accurate as possible.
        """
        if self._data_manager is None:
            return {}, 0

        from torch.utils.data import DataLoader as _DL
        dataset = self._data_manager.get_dataset(
            np.arange(0, self._total_classes), source="train", mode="train"
        )
        batch_size = int(self.args.get("batch_size", 128))
        loader = _DL(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        net.eval()
        fisher = {
            name: torch.zeros_like(param)
            for name, param in net.named_parameters()
            if param.requires_grad and not name.startswith("fc.")
        }
        num_samples = 0
        for batch_idx, batch in enumerate(tqdm(loader, desc="Full Fisher [all seen]")):
            if self._full_fisher_max_batches is not None and batch_idx >= self._full_fisher_max_batches:
                break
            if len(batch) == 3:
                _, inputs, targets = batch
            else:
                inputs, targets = batch
            inputs  = inputs.to(self._device, non_blocking=True)
            targets = targets.to(self._device, non_blocking=True)

            net.zero_grad()
            logits = net(inputs)["logits"][:, : self._total_classes]
            F.cross_entropy(logits, targets).backward()

            bs = int(inputs.size(0))
            for name, param in net.named_parameters():
                if param.requires_grad and param.grad is not None and name in fisher:
                    fisher[name] += param.grad.detach().pow(2) * bs
            num_samples += bs

        if num_samples > 0:
            for name in fisher:
                fisher[name] /= float(num_samples)

        net.train()
        logging.info(
            "[EWCGAMProjected] Full Fisher (all seen tasks) estimated over %d samples",
            num_samples,
        )
        return fisher, num_samples

    def _compute_full_fisher_diff_metrics(self, net):
        """Compare accumulated omega (Ω) vs full Fisher over all seen training data.

        Called at the end of every task when compute_full_fisher=True.
        Metrics are consistent with EWCon_Probe._compute_full_fisher_diff_metrics.
        """
        if not self.omega:
            return {}
        full_fisher, num_samples = self._compute_fisher_all_seen(net)
        if not full_fisher:
            return {}

        accum_vals, full_vals = [], []
        for name in self.omega:
            if name in full_fisher:
                accum_vals.append(self.omega[name].detach().cpu().float().view(-1))
                full_vals.append(full_fisher[name].detach().cpu().float().view(-1))
        if not accum_vals:
            return {}

        accum_flat = torch.cat(accum_vals)
        full_flat  = torch.cat(full_vals)
        diff       = full_flat - accum_flat
        numel      = int(accum_flat.numel())

        accum_trace = float(accum_flat.sum())
        full_trace  = float(full_flat.sum())
        trace_diff  = float(diff.sum())

        accum_fro = float(accum_flat.pow(2).sum().sqrt())
        full_fro  = float(full_flat.pow(2).sum().sqrt())
        diff_fro  = float(diff.pow(2).sum().sqrt())

        accum_l1    = float(accum_flat.abs().sum())
        full_l1     = float(full_flat.abs().sum())
        abs_diff_l1 = float(diff.abs().sum())

        cosine_sim  = float(
            (accum_flat * full_flat).sum() / max(accum_fro * full_fro, 1e-12)
        )
        sign_agree  = float(((accum_flat > 0) == (full_flat > 0)).float().mean())
        max_abs_diff = float(diff.abs().max())

        k1pct = max(1, int(numel * 0.01))
        full_top_idx       = full_flat.topk(k1pct, largest=True).indices
        top1pct_full_sum   = float(full_flat[full_top_idx].sum())
        top1pct_accum_sum  = float(accum_flat[full_top_idx].sum())

        return {
            "full_fisher_num_samples":        int(num_samples),
            "full_fisher_num_params":         numel,
            "full_fisher_trace":              full_trace,
            "full_fisher_fro":                full_fro,
            "full_fisher_l1":                 full_l1,
            "full_fisher_top1pct_trace":      top1pct_full_sum,
            "accum_fisher_trace":             accum_trace,
            "accum_fisher_fro":               accum_fro,
            "accum_fisher_l1":                accum_l1,
            "accum_fisher_top1pct_by_full":   top1pct_accum_sum,
            "full_vs_accum_trace_diff":       trace_diff,
            "full_vs_accum_abs_trace_diff":   abs(trace_diff),
            "full_vs_accum_trace_ratio":      full_trace / max(abs(accum_trace), 1e-12),
            "full_vs_accum_fro_diff":         diff_fro,
            "full_vs_accum_rel_fro":          diff_fro / max(accum_fro, 1e-12),
            "full_vs_accum_rel_fro_wrt_full": diff_fro / max(full_fro, 1e-12),
            "full_vs_accum_rel_l1":           abs_diff_l1 / max(accum_l1, 1e-12),
            "full_vs_accum_rel_l1_wrt_full":  abs_diff_l1 / max(full_l1, 1e-12),
            "full_vs_accum_max_abs_diff":     max_abs_diff,
            "full_vs_accum_cosine_sim":       cosine_sim,
            "full_vs_accum_sign_agree":       sign_agree,
            "full_vs_accum_top1pct_accum_coverage": (
                top1pct_accum_sum / max(top1pct_full_sum, 1e-12)
            ),
        }

    def get_mechanism_metrics(self) -> dict:
        return dict(self._last_full_fisher_diff)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _net(self):
        return self._network.module if isinstance(self._network, nn.DataParallel) else self._network

    def _task_range(self):
        """Return (start_class, end_class) for the task just trained."""
        return self._known_classes, self._total_classes


def _flatten_optimizer_params(optimizer):
    params = []
    seen = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            if not param.requires_grad:
                continue
            pid = id(param)
            if pid in seen:
                continue
            seen.add(pid)
            params.append(param)
    return params


# Factory alias
Learner = EWCGAMProjected
