import logging
import numpy as np
from PIL import Image
from torch.utils.data import Dataset,Subset
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor
from utils.data import iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000, iCIFAR224, iImageNetR,iImageNetA,CUB, Cars196, objectnet, omnibenchmark, vtab, iImageNetC, iTinyImageNetC, iTinyImageNetP,iDomainNet,iAircraft,iFlowers102,iOxfordPet
import torch, math, json, os
from typing import Optional, List, Dict, Tuple
from torch.utils.data import DataLoader

try:
    from torchvision.io import ImageReadMode, read_image
except Exception:  # pragma: no cover - optional torchvision image backend
    ImageReadMode = None
    read_image = None

class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = dict(args) if isinstance(args, dict) else args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        if self._task_splits is not None:
            if shuffle:
                raise ValueError(
                    "Datasets with explicit task boundaries (for example het5) "
                    "require class_shuffle=false."
                )
            task_splits = [int(size) for size in self._task_splits]
            if not task_splits or any(size <= 0 for size in task_splits):
                raise ValueError(f"Invalid task_splits: {task_splits}")
            if sum(task_splits) != len(self._class_order):
                raise ValueError(
                    "task_splits must cover the complete class order: "
                    f"sum={sum(task_splits)}, classes={len(self._class_order)}"
                )
            # iData provided explicit per-task splits (e.g. iHet5Datasets); use them directly.
            self._increments = task_splits
        else:
            init_cls = int(init_cls)
            increment = int(increment)
            if init_cls <= 0 or increment < 0:
                raise ValueError(
                    f"init_cls must be positive and increment non-negative, "
                    f"got {init_cls} and {increment}."
                )
            if increment == 0 and init_cls != len(self._class_order):
                raise ValueError(
                    "increment=0 is only valid for a one-task dataset covering all classes: "
                    f"init_cls={init_cls}, classes={len(self._class_order)}."
                )
            self._increments = [init_cls]
            while sum(self._increments) + increment < len(self._class_order):
                self._increments.append(increment)
            offset = len(self._class_order) - sum(self._increments)
            if offset > 0:
                self._increments.append(offset)
            
    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_task_class_range(self, task):
        assert 0 <= task < len(self._increments), "Task index out of range"
        start = sum(self._increments[:task])
        end = start + self._increments[task]
        return start, end

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(
                    x, y, low_range=idx, high_range=idx + 1
                )
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(
                data, targets, trsf, self.use_path, loader_mode=self.loader_mode
            )
        else:
            return DummyDataset(
                data, targets, trsf, self.use_path, loader_mode=self.loader_mode
            )

    @staticmethod
    def _optional_positive_int(value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in {"", "none", "null", "false", "off"}:
                return None
            value = stripped
        parsed = int(value)
        return parsed if parsed > 0 else None

    @staticmethod
    def _optional_positive_float(value):
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.lower() in {"", "none", "null", "false", "off"}:
                return None
            value = stripped
        parsed = float(value)
        return parsed if parsed > 0.0 else None

    @staticmethod
    def _seed_to_int(seed, default=0):
        if isinstance(seed, (list, tuple)):
            seed = seed[0] if seed else default
        if seed is None:
            seed = default
        return int(seed)

    @staticmethod
    def _parse_class_order_override(value, expected_len):
        if value is None or value == "":
            return None
        if isinstance(value, str):
            raw = value.strip()
            if raw.lower() in {"none", "null", "false", "off"}:
                return None
            if raw.startswith("["):
                value = json.loads(raw)
            else:
                order = []
                for part in raw.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if "-" in part:
                        lo, hi = part.split("-", 1)
                        order.extend(range(int(lo), int(hi) + 1))
                    else:
                        order.append(int(part))
                value = order
        order = [int(item) for item in value]
        expected = list(range(int(expected_len)))
        if sorted(order) != expected:
            raise ValueError(
                "class_order_override must be a permutation of 0..{}; got len={} "
                "unique={}".format(expected_len - 1, len(order), len(set(order)))
            )
        return order

    def _build_transform(self, mode):
        if mode == "train":
            return transforms.Compose([*self._train_trsf, *self._common_trsf])
        if mode == "flip":
            return transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        if mode == "test":
            return transforms.Compose([*self._test_trsf, *self._common_trsf])
        raise ValueError("Unknown mode {}.".format(mode))

    def get_dataset_with_split(
        self,
        indices,
        source,
        mode,
        appendent=None,
        val_samples_per_class=0,
        val_ratio=None,
        val_per_class=None,
        val_min_per_class=0,
        val_max_per_class=None,
        seed=None,
        train_mode=None,
        val_mode="test",
        ret_indices=False,
        min_train_per_class=1,
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        train_mode = mode if train_mode is None else train_mode
        train_trsf = self._build_transform(train_mode)
        val_trsf = self._build_transform(val_mode)
        rng = np.random.default_rng(self._seed_to_int(seed, 0))
        val_ratio = self._optional_positive_float(val_ratio)
        val_per_class = self._optional_positive_int(
            val_per_class if val_per_class is not None else val_samples_per_class
        )
        val_min_per_class = self._optional_positive_int(val_min_per_class) or 0
        val_max_per_class = self._optional_positive_int(val_max_per_class)
        min_train_per_class = max(0, int(min_train_per_class))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        train_source_indices, val_source_indices = [], []
        per_class_counts = {}

        def split_class(class_data, class_targets, source_indices=None):
            n_total = int(len(class_data))
            if n_total <= 0:
                empty = np.asarray([], dtype=int)
                return empty, empty

            if val_per_class is not None:
                n_val = int(val_per_class)
            elif val_ratio is not None:
                n_val = int(round(float(n_total) * float(val_ratio)))
                if n_val <= 0:
                    n_val = 1
                n_val = max(n_val, int(val_min_per_class))
                if val_max_per_class is not None:
                    n_val = min(n_val, int(val_max_per_class))
            elif val_min_per_class > 0:
                n_val = int(val_min_per_class)
            else:
                n_val = 0

            max_val = max(0, n_total - min_train_per_class)
            n_val = max(0, min(int(n_val), int(max_val)))
            perm = rng.permutation(n_total)
            val_local = np.sort(perm[:n_val])
            train_local = np.sort(perm[n_val:])

            train_data.append(class_data[train_local])
            train_targets.append(class_targets[train_local])
            if n_val > 0:
                val_data.append(class_data[val_local])
                val_targets.append(class_targets[val_local])

            if source_indices is not None:
                train_source_indices.extend(np.asarray(source_indices)[train_local].astype(int).tolist())
                val_source_indices.extend(np.asarray(source_indices)[val_local].astype(int).tolist())

            return train_local, val_local

        for idx in indices:
            class_source_indices = np.where(np.logical_and(y >= idx, y < idx + 1))[0]
            class_data, class_targets = x[class_source_indices], y[class_source_indices]
            train_local, val_local = split_class(
                class_data,
                class_targets,
                source_indices=class_source_indices,
            )
            per_class_counts[int(idx)] = {
                "total": int(len(class_data)),
                "train": int(len(train_local)),
                "val": int(len(val_local)),
            }

        if appendent is not None and len(appendent) != 0 and len(appendent[1]) > 0:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                train_local, val_local = split_class(
                    append_data,
                    append_targets,
                    source_indices=None,
                )
                per_class_counts[f"appendent_{int(idx)}"] = {
                    "total": int(len(append_data)),
                    "train": int(len(train_local)),
                    "val": int(len(val_local)),
                }

        data_dtype = getattr(x, "dtype", object)
        target_dtype = getattr(y, "dtype", int)
        train_data = np.concatenate(train_data) if train_data else np.asarray([], dtype=data_dtype)
        train_targets = (
            np.concatenate(train_targets)
            if train_targets
            else np.asarray([], dtype=target_dtype)
        )
        val_data = np.concatenate(val_data) if val_data else np.asarray([], dtype=data_dtype)
        val_targets = (
            np.concatenate(val_targets)
            if val_targets
            else np.asarray([], dtype=target_dtype)
        )

        train_dataset = DummyDataset(
            train_data, train_targets, train_trsf, self.use_path, loader_mode=self.loader_mode
        )
        val_dataset = DummyDataset(
            val_data, val_targets, val_trsf, self.use_path, loader_mode=self.loader_mode
        )

        if not ret_indices:
            return train_dataset, val_dataset

        split_info = {
            "source": str(source),
            "train_mode": str(train_mode),
            "val_mode": str(val_mode),
            "seed": self._seed_to_int(seed, 0),
            "val_ratio": float(val_ratio or 0.0),
            "val_per_class": int(val_per_class or 0),
            "val_min_per_class": int(val_min_per_class or 0),
            "val_max_per_class": int(val_max_per_class or 0),
            "train_samples": int(len(train_dataset)),
            "val_samples": int(len(val_dataset)),
            "train_source_indices": list(train_source_indices),
            "val_source_indices": list(val_source_indices),
            "per_class_counts": per_class_counts,
        }
        return train_dataset, val_dataset, split_info

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        # Data
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path
        self.loader_mode = getattr(idata, "loader_mode", "pil")
        self.loader_options = dict(getattr(idata, "loader_options", {}) or {})

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        # Order
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        override = None
        if isinstance(self.args, dict):
            override = self.args.get("class_order_override", None)
        override_order = self._parse_class_order_override(override, len(order))
        if override_order is not None:
            order = override_order
            logging.info("[DataManager] Using class_order_override")
        self._class_order = order
        logging.info(self._class_order)

        # Expose per-task class counts if iData provides them (e.g. iHet5Datasets)
        self._task_splits = getattr(idata, "task_splits", None)
        self.task_names = list(getattr(idata, "task_names", []) or [])
        self.task_offsets = list(getattr(idata, "task_offsets", []) or [])

        # Map indices
        self._train_targets = _map_new_class_index(
            self._train_targets, self._class_order
        )
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def get_dataloader_kwargs(self, num_workers, **overrides):
        kwargs = dict(overrides)

        pin_memory_default = bool(self.loader_options.get("pin_memory", False) and torch.cuda.is_available())
        if "pin_memory" not in kwargs:
            kwargs["pin_memory"] = pin_memory_default

        if num_workers > 0 and "persistent_workers" not in kwargs:
            kwargs["persistent_workers"] = bool(self.loader_options.get("persistent_workers", False))

        return kwargs

    def build_dataloader(self, dataset, *, batch_size, shuffle, num_workers, **kwargs):
        loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
        }
        loader_kwargs.update(self.get_dataloader_kwargs(num_workers, **kwargs))
        return DataLoader(dataset, **loader_kwargs)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False, loader_mode="pil"):
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.loader_mode = loader_mode

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.use_path:
            if self.loader_mode == "torchvision_tensor":
                image = self.trsf(torchvision_tensor_loader(self.images[idx]))
            else:
                image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]

        return idx, image, label



def fractional_loader(
    loader,
    fraction: Optional[float]=0.1,
    seed: Optional[int] = None,
    balanced: bool = True,
    batch_size: Optional[int] = None,
    shuffle: bool = True,
):
    """Return a DataLoader restricted to a random subset of the original dataset."""
    
    frac = float(fraction)
    

    dataset = getattr(loader, "dataset", None)
    if dataset is None:
        return loader
    
    total = len(dataset)
    
    if total <= 0:
        return loader

    subset_size = max(1, int(math.ceil(total * frac)))
    print(f"Origin dataset size: {total}, subset_size:{subset_size}")

    # ==== NEW: 保证“类别均衡”时至少覆盖所有类别 ====
    labels = extract_labels(dataset) if balanced else None
    if balanced and labels is not None:
        num_classes = len({int(y) for y in labels})
        if subset_size < num_classes:
            # 保障每类至少 1 个样本
            subset_size = num_classes
    # ==== END NEW ====

    if subset_size >= total:
        return loader

    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(int(seed))
    else:
        generator.manual_seed(torch.seed())

    indices: List[int]

    # 若上面没能拿到 labels（例如 balanced=False 或提取失败），此处再尝试一次以兼容旧逻辑
    if labels is None:
        labels = extract_labels(dataset) if balanced else None

    if labels is not None and len(labels) == total:
        per_class: Dict[int, List[int]] = {}
        for local_idx, lbl in enumerate(labels):
            per_class.setdefault(int(lbl), []).append(local_idx)
        class_items = list(per_class.items())
        num_classes = len(class_items)
        chosen: List[int] = []
        if subset_size >= num_classes and num_classes > 0:
            remaining_pool: List[int] = []
            for _, idxs in class_items:
                perm = torch.randperm(len(idxs), generator=generator)
                first_pick = idxs[perm[0].item()]
                chosen.append(first_pick)
                if len(idxs) > 1:
                    remaining_pool.extend([idxs[i] for i in perm[1:].tolist()])
            remaining_needed = subset_size - num_classes
            if remaining_needed > 0 and remaining_pool:
                perm = torch.randperm(len(remaining_pool), generator=generator)[:remaining_needed]
                chosen.extend([remaining_pool[i] for i in perm.tolist()])
            indices = chosen[:subset_size]
        elif num_classes > 0:
            # 注意：这一分支只有在 subset_size < num_classes 时出现；
            # 但我们在“NEW”补丁中已把 subset_size 提升到 num_classes，通常不会走到这里。
            class_perm = torch.randperm(num_classes, generator=generator)[:subset_size]
            for class_idx in class_perm.tolist():
                idxs = class_items[class_idx][1]
                perm = torch.randperm(len(idxs), generator=generator)
                chosen.append(idxs[perm[0].item()])
            indices = chosen
        else:
            indices = torch.randperm(total, generator=generator)[:subset_size].tolist()
    else:
        indices = torch.randperm(total, generator=generator)[:subset_size].tolist()

    subset = Subset(dataset, indices)

    if batch_size is None:
        batch_size = loader.batch_size
        


    dl_kwargs = {
        "batch_size": batch_size,
        "shuffle": bool(shuffle),
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "drop_last": loader.drop_last,
        "collate_fn": loader.collate_fn,
        "worker_init_fn": getattr(loader, "worker_init_fn", None),
        "pin_memory_device": getattr(loader, "pin_memory_device", ""),
        "timeout": loader.timeout,
    }
    if loader.num_workers > 0:
        dl_kwargs["persistent_workers"] = getattr(loader, "persistent_workers", False)
        prefetch_factor = getattr(loader, "prefetch_factor", None)
        if prefetch_factor is not None:
            dl_kwargs["prefetch_factor"] = prefetch_factor
    multiprocessing_context = getattr(loader, "multiprocessing_context", None)
    if multiprocessing_context is not None:
        dl_kwargs["multiprocessing_context"] = multiprocessing_context

    return DataLoader(subset, **dl_kwargs)




def extract_labels(dataset):
    # 返回每个样本的整数标签列表；支持 Subset/张量/ndarray/list
    if isinstance(dataset, Subset):
        base = extract_labels(dataset.dataset)
        idxs = dataset.indices.tolist() if hasattr(dataset.indices, "tolist") else list(dataset.indices)
        return [int(base[i]) for i in idxs] if base is not None else None
    for attr in ("labels","targets","y"):
        lab = getattr(dataset, attr, None)
        if lab is None or callable(lab): 
            continue
        if isinstance(lab, torch.Tensor): return lab.detach().cpu().flatten().tolist()
        if isinstance(lab, np.ndarray):  return lab.astype(int).flatten().tolist()
        if isinstance(lab, list):        return [int(x) for x in lab]
    return None

# def balanced_indices(dataset, fraction: float, seed: Optional[int], ensure_per_class: bool=True) -> List[int]:
#     # 类别均衡随机子采样：至少保证每类抽到 1 个（若 ensure_per_class=True）
#     total = len(dataset)
#     k = max(1, int(math.ceil(total * float(fraction))))
#     g = torch.Generator()
#     g.manual_seed(int(seed) if seed is not None else torch.seed())
#     labels = extract_labels(dataset)
#     if labels is None or len(labels) != total:
#         # 无法拿到标签，退化为纯随机
#         return torch.randperm(total, generator=g)[:k].tolist()
#     per_class: Dict[int, List[int]] = {}
#     for i, y in enumerate(labels):
#         per_class.setdefault(int(y), []).append(i)
#     # 先每类至少 1 个（若需要），剩余配额从“其余池”随机抽
#     chosen: List[int] = []
#     rest: List[int] = []
#     for _, idxs in per_class.items():
#         perm = torch.randperm(len(idxs), generator=g)
#         if ensure_per_class and len(chosen) < k:
#             chosen.append(idxs[perm[0].item()])
#             if len(idxs) > 1:
#                 rest += [idxs[i] for i in perm[1:].tolist()]
#         else:
#             rest += [idxs[i] for i in perm.tolist()]
#     remain = k - len(chosen)
#     if remain > 0 and rest:
#         perm = torch.randperm(len(rest), generator=g)[:remain]
#         chosen += [rest[i] for i in perm.tolist()]
#     return chosen[:k]


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10(args)
    elif name in {"cifar10_224", "cifar10-224", "cifar10vit"}:
        from utils.data import iCIFAR10_224
        return iCIFAR10_224(args)
    elif name == "cifar100":
        return iCIFAR100(args)
    elif name == "imagenet1000":
        return iImageNet1000(args)
    elif name == "imagenet100":
        return iImageNet100(args)
    elif name == "cifar224":
        return iCIFAR224(args)
    elif name == "imagenetr":
        return iImageNetR(args)
    elif name == 'domainnet':
        return iDomainNet(args)
    elif name == "tiny_imagenetc":
        return iTinyImageNetC(args)
    elif name == "tiny_imagenetp":
        return iTinyImageNetP(args)
    elif name == "imageneta":
        return iImageNetA(args)
    elif name in {"cub", "cub200", "cub-200", "cub_200", "cub200-2011", "cub-200-2011", "cub_200_2011"}:
        return CUB(args)
    elif name in {"cars196", "cars-196", "cars"}:
        return Cars196(args)
    elif name == "objectnet":
        return objectnet(args)
    elif name == "omnibenchmark":
        return omnibenchmark(args)
    elif name == "vtab":
        return vtab(args)
    elif name in {"flowers102", "flowers"}:
        return iFlowers102(args)
    elif name in {"pets", "oxfordpet", "oxford-iiit-pet"}:
        return iOxfordPet(args)
    elif name in {"aircraft", "fgvc-aircraft"}:
        return iAircraft(args)
    elif name in {"het5", "5datasets", "het_5datasets"}:
        from utils.data import iHet5Datasets
        return iHet5Datasets(args)
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def torchvision_tensor_loader(path):
    if read_image is not None and ImageReadMode is not None:
        return read_image(path, mode=ImageReadMode.RGB)
    return pil_to_tensor(pil_loader(path))


def accimage_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    accimage is an accelerated Image loader and preprocessor leveraging Intel IPP.
    accimage is available on conda-forge.
    """
    import accimage

    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    from torchvision import get_image_backend

    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)
