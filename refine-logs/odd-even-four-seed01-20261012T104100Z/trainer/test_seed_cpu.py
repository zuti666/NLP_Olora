"""CPU-only seed propagation checks for the four-direction seed extension."""
import ast
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.dont_write_bytecode = True
import numpy as np
import torch
import audit_core as audit
import cuda_witness
import native_lifecycle as native
import odd_even_train as trainer

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent.parent / "odd-even-rho005-gpu123-20261003T185629Z"
SOURCE = Path(json.loads((PARENT / "suite.json").read_text())["source_root"])
DIRECTIONS = ("g1_plus", "backward_reflected", "central", "g0")


def cli(directory):
    return SimpleNamespace(max_tasks=0, max_epochs=0, max_batches=0, output_dir=directory)


def raw(direction, seed):
    return dict(probe_branch="local_symmetric", update_direction=direction,
                step_mode="natural", seed=seed)


def function_from_file(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


class SeedAcceptance(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_registered_scalar_and_singleton_seeds_and_zero(self):
        for direction in DIRECTIONS:
            for seed in (0, 1, 2):
                for value in (seed, [seed]):
                    with self.subTest(direction=direction, seed=value):
                        config = trainer.resolve_config(raw(direction, value), cli("/tmp/seed-config-test"))
                        self.assertEqual(config["seed"], [seed])
                        self.assertEqual(config["gam_grad_rho"], .05)
                        self.assertIsNone(config["outer_radius"])
        for value in ([], [0, 1], [3], 3, -1, True, [False], 0., [1.], "0", None):
            with self.subTest(invalid=value), self.assertRaises(ValueError):
                trainer.resolve_config(raw("g0", value), cli("/tmp/seed-config-test"))

    def test_parent_seed2_exact_resolved_config_and_default(self):
        spec = importlib.util.spec_from_file_location("parent_odd_even_train_seedtest", PARENT / "trainer/odd_even_train.py")
        parent = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(parent)
        for config_path in sorted((PARENT / "configs").glob("OE*.json")):
            config = json.loads(config_path.read_text())
            self.assertEqual(parent.resolve_config(config, cli("/tmp/seed2-parity")),
                             trainer.resolve_config(config, cli("/tmp/seed2-parity")))
        config = raw("g0", 2)
        del config["seed"]
        self.assertEqual(parent.resolve_config(config, cli("/tmp/seed2-parity")),
                         trainer.resolve_config(config, cli("/tmp/seed2-parity")))
        self.assertEqual((PARENT / "trainer/odd_even_step.py").read_bytes(), (HERE / "odd_even_step.py").read_bytes())

    def test_dry_run_reports_actual_seed_for_all_twelve_configs(self):
        with tempfile.TemporaryDirectory(prefix="odd-even-seed-dry-") as directory:
            config_path = Path(directory) / "config.json"
            for direction in DIRECTIONS:
                for seed in (0, 1, 2):
                    config_path.write_text(json.dumps(raw(direction, [seed])))
                    output = io.StringIO()
                    args = ["odd_even_train.py", "--source", str(SOURCE), "--config", str(config_path),
                            "--output-dir", str(Path(directory) / "unused"), "--dry-run"]
                    with patch.object(sys, "argv", args), redirect_stdout(output):
                        trainer.main()
                    receipt = json.loads(output.getvalue())
                    self.assertEqual(receipt["seed"], seed)
                    self.assertEqual(receipt["direction"], direction)
            self.assertFalse((Path(directory) / "unused").exists())

    def test_frozen_source_train_passes_requested_seed_to_controller_and_rng(self):
        # Execute the actual frozen native train() dispatcher and RNG helper on CPU.
        # The expensive _train body is replaced by a receiver at its entry boundary.
        set_random = function_from_file(SOURCE / "utils/random_reproduce.py", "set_random",
                                       dict(os=os, random=random, np=np, torch=torch))
        signatures = {}
        with tempfile.TemporaryDirectory(prefix="odd-even-seed-controller-") as directory:
            for seed in (0, 1, 2):
                received = []
                def receive(args):
                    self.assertIs(type(args["seed"]), int)
                    self.assertEqual(args["seed"], seed)
                    set_random(args["seed"])
                    signatures[seed] = audit.tensor_digest(torch.nn.Linear(4, 20).weight)
                    out = Path(directory) / str(seed)
                    out.mkdir()
                    controller = trainer.Controller(None, args, cli(str(out)))
                    try:
                        self.assertEqual(controller.seed, seed)
                        received.append(controller.seed)
                    finally:
                        controller.handle.close()
                dispatch = function_from_file(SOURCE / "src/trainer.py", "train", dict(copy=copy, _train=receive))
                cfg = trainer.resolve_config(raw("g0", [seed]), cli(directory))
                dispatch(cfg)
                self.assertEqual(received, [seed])
                set_random(seed)
                self.assertEqual(signatures[seed], audit.tensor_digest(torch.nn.Linear(4, 20).weight))
        self.assertEqual(len(set(signatures.values())), 3)

    def test_epoch_loader_actual_data_seed_and_pairing(self):
        class Dataset(torch.utils.data.Dataset):
            def __len__(self): return 6
            def __getitem__(self, index):
                return index, torch.tensor([random.random(), float(np.random.rand()), float(torch.rand(()))]), index % 2
        class Manager:
            def __init__(self): self.seeds = []
            def build_dataloader(self, dataset, **kwargs):
                self.seeds.append(kwargs["generator"].initial_seed())
                return torch.utils.data.DataLoader(dataset, **kwargs)
        signatures = {}
        for seed in (0, 1, 2):
            per_direction = []
            for direction in DIRECTIONS:
                dataset = Dataset()
                manager = Manager()
                controller = SimpleNamespace(seed=seed, task=4, cfg={"train_num_workers": 0}, dm=manager,
                    cli=SimpleNamespace(max_batches=0), begin_epoch=lambda epoch: None)
                original = torch.utils.data.DataLoader(dataset, batch_size=128)
                loader = native.EpochLoader(original, controller)
                before = audit.rng_get()
                list(loader)
                after = audit.rng_get()
                self.assertTrue(audit.rng_equal(before, after))
                self.assertEqual(manager.seeds, [seed * 100000 + 4001])
                self.assertEqual(controller.batch_identity["data_seed"], seed * 100000 + 4001)
                per_direction.append(controller.batch_identity)
            self.assertTrue(all(value == per_direction[0] for value in per_direction))
            signatures[seed] = per_direction[0]["input_sha256"]
        self.assertEqual(len(set(signatures.values())), 3)

    def test_witness_seed_argument_and_receipt_cpu_mock_only(self):
        # Mock only CUDA allocation/name/availability: do not submit any GPU work.
        randn = torch.randn
        with tempfile.TemporaryDirectory(prefix="odd-even-witness-mock-") as directory:
            for seed in (0, 1, 2):
                output = Path(directory) / (str(seed) + ".json")
                args = ["cuda_witness.py", "--output", str(output), "--seed", str(seed)]
                with patch.object(sys, "argv", args), redirect_stdout(io.StringIO()), \
                     patch.object(torch.cuda, "is_available", return_value=True), \
                     patch.object(torch.cuda, "get_device_name", return_value="CPU_UNIT_TEST_MOCK"), \
                     patch.object(torch, "randn", side_effect=lambda *a, **kw: randn(*a, **dict(kw, device="cpu"))):
                    cuda_witness.main()
                receipt = json.loads(output.read_text())
                self.assertEqual(receipt["seed"], seed)
                self.assertEqual(torch.initial_seed(), seed)
                self.assertEqual(receipt["device"], "CPU_UNIT_TEST_MOCK")


def main():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SeedAcceptance)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    files = ("odd_even_train.py", "cuda_witness.py", "odd_even_step.py", "test_seed_cpu.py")
    report = dict(status="passed" if result.wasSuccessful() else "failed", cpu_only=True,
                  gpu_acceptance=False, tests_run=result.testsRun,
                  failures=len(result.failures), errors=len(result.errors),
                  test_names=unittest.defaultTestLoader.getTestCaseNames(SeedAcceptance),
                  registered_seeds=[0, 1, 2], directions=list(DIRECTIONS),
                  parent_seed2_resolved_configs_exact=True if result.wasSuccessful() else None,
                  seed0_not_replaced_by_default=True if result.wasSuccessful() else None,
                  step_byte_identical=(PARENT / "trainer/odd_even_step.py").read_bytes() == (HERE / "odd_even_step.py").read_bytes(),
                  local_sha256={name: audit.digest(HERE / name) for name in files},
                  parent_sha256={name: audit.digest(PARENT / "trainer" / name) for name in files if name != "test_seed_cpu.py"},
                  source_dispatch_sha256=audit.digest(SOURCE / "src/trainer.py"),
                  source_rng_helper_sha256=audit.digest(SOURCE / "utils/random_reproduce.py"),
                  python=sys.executable, torch_version=torch.__version__)
    audit.atomic_json(HERE / "SEED_CPU_GATE.json", report)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
