import logging
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
from models_CL.baseLearner import BaseLearner
from utils.toolkit import tensor2numpy

num_workers = 8


class Learner(BaseLearner):
    """
    Online EWC on the full model (no LoRA).

    L = L_CE + (λ/2) Σ_i Ω_i (θ_i − θ*_i)²
    Ω_t = γ · Ω_{t-1} + F̂_t   (online accumulation)
    """

    def __init__(self, args):
        super().__init__(args)
        self.topk = 1
        self._network = IncrementalNet(args, True)

        self.gamma = float(args.get("gamma", 1.0))
        self.ewc_weight = float(args["lambda"])

        self.omega = {}
        self.theta_star = {}
        self.count_updates = 0

    def after_task(self):
        print("=== Computing Fisher Information ===")
        net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
        fisher = self._compute_fisher(net)

        if self.count_updates == 0:
            self.omega = {name: f.clone() for name, f in fisher.items()}
        else:
            for name, f in fisher.items():
                prev = self.omega.get(name, torch.zeros_like(f))
                self.omega[name] = self.gamma * prev + f

        self.count_updates += 1

        self.theta_star = {
            name: param.detach().clone()
            for name, param in net.named_parameters()
            if param.requires_grad
        }

        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
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

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        if self._cur_task == 0:
            optimizer = optim.SGD(
                self._network.parameters(),
                momentum=0.9,
                lr=self.args["init_lr"],
                weight_decay=self.args["init_weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["init_milestones"],
                gamma=self.args["init_lr_decay"],
            )
            self._init_train(train_loader, test_loader, optimizer, scheduler)
        else:
            optimizer = optim.SGD(
                self._network.parameters(),
                lr=self.args["lrate"],
                momentum=0.9,
                weight_decay=self.args["weight_decay"],
            )
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer=optimizer,
                milestones=self.args["milestones"],
                gamma=self.args["lrate_decay"],
            )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["init_epoch"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                loss = F.cross_entropy(logits, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["init_epoch"],
                    losses / len(train_loader), train_acc, test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task, epoch + 1, self.args["init_epoch"],
                    losses / len(train_loader), train_acc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args["epochs"]))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            ce_losses = 0.0
            ewc_losses = 0.0
            correct, total = 0, 0

            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                logits = self._network(inputs)["logits"]

                fake_targets = targets - self._known_classes
                loss_clf = F.cross_entropy(logits[:, self._known_classes:], fake_targets)

                net = self._network.module if isinstance(self._network, nn.DataParallel) else self._network
                ewc_loss = torch.tensor(0., device=self._device)
                for name, param in net.named_parameters():
                    if name in self.omega:
                        diff = param - self.theta_star[name].to(self._device)
                        ewc_loss = ewc_loss + (self.omega[name].to(self._device) * diff.pow(2)).sum()
                weighted_ewc = self.ewc_weight / 2. * ewc_loss

                loss = loss_clf + weighted_ewc
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()
                ce_losses += float(loss_clf.detach())
                ewc_losses += float(weighted_ewc.detach())

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            avg_ce = ce_losses / len(train_loader)
            avg_ewc = ewc_losses / len(train_loader)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}, CE {:.3f}, EWC {:.3e}".format(
                    self._cur_task, epoch + 1, self.args["epochs"],
                    losses / len(train_loader), train_acc, test_acc, avg_ce, avg_ewc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, CE {:.3f}, EWC {:.3e}".format(
                    self._cur_task, epoch + 1, self.args["epochs"],
                    losses / len(train_loader), train_acc, avg_ce, avg_ewc,
                )
            prog_bar.set_description(info)

        logging.info(info)

    def _compute_fisher(self, net):
        net.eval()

        fisher = {
            name: torch.zeros_like(param)
            for name, param in net.named_parameters()
            if param.requires_grad
        }
        num_samples = 0

        for _, inputs, targets in tqdm(self.train_loader, desc="Computing Fisher"):
            inputs = inputs.to(self._device)
            targets = targets.to(self._device)

            mask = (targets >= self._known_classes).nonzero().view(-1)
            if mask.numel() == 0:
                continue
            inputs = torch.index_select(inputs, 0, mask)
            local_targets = torch.index_select(targets, 0, mask) - self._known_classes

            net.zero_grad()
            logits = net(inputs)["logits"]
            loss = F.cross_entropy(logits[:, self._known_classes:], local_targets)
            loss.backward()

            batch_size = int(inputs.size(0))
            for name, param in net.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.detach().pow(2) * batch_size
            num_samples += batch_size

        if num_samples > 0:
            for name in fisher:
                fisher[name] /= num_samples

        net.train()
        return fisher
