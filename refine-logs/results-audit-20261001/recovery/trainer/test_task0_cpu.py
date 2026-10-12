"""CPU checks: real GAM/SAM, frozen native epoch loop, hook/checkpoint/data invariants."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import torch
from torch.utils.data import Dataset, DataLoader

sys.dont_write_bytecode = True
import task0_trainer as t
SOURCE = Path('/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source')
sys.path.insert(0, str(SOURCE))
from optimer_PerturabtionType.gam import GAM
from optimer_PerturabtionType.optimer_sam import SAM
from tqdm import tqdm


class Backbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w_As = torch.nn.ModuleList([torch.nn.Linear(4, 2, bias=False) for _ in range(2)])
        self.w_Bs = torch.nn.ModuleList([torch.nn.Linear(2, 4, bias=False) for _ in range(2)])
        self.frozen = torch.nn.Parameter(torch.eye(4), requires_grad=False)
        self.w_meta = [{'layer': 0, 'channel': c, 'task_id': 0} for c in ['q', 'v']]
        self.task_id = 0
    def forward(self, x):
        torch.rand(1)
        return x @ self.frozen + sum(b(a(x)) for a, b in zip(self.w_As, self.w_Bs))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone, self.fc = Backbone(), torch.nn.Linear(4, 20)
    def forward(self, x):
        return {'logits': self.fc(self.backbone(x))}


class Data(Dataset):
    def __init__(self):
        self.x = torch.arange(160, dtype=torch.float64).reshape(40, 4) / 160
    def __len__(self): return 40
    def __getitem__(self, i):
        return i, self.x[i] + torch.rand(4, dtype=torch.float64) * .01, i % 20


class DM:
    _class_order = list(range(200))
    def build_dataloader(self, dataset, *, num_workers, **kwargs):
        if num_workers == 0:
            kwargs.pop('persistent_workers', None)
        return DataLoader(dataset, num_workers=num_workers, **kwargs)


tree = ast.parse((SOURCE / 'models_LoRAbasedCL/as1_normfisher_gam_intask.py').read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Learner')
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_run_regularized_train')
ns = {'tqdm': tqdm, 'torch': torch}
exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), '<frozen-native-epoch>', 'exec'), ns)


class Learner:
    _run_regularized_train = ns['_run_regularized_train']
    def __init__(self, cfg):
        self.args, self._network = cfg, Model().double()
        self._device, self._cur_task, self._total_classes = torch.device('cpu'), 0, 20
        self._task_endpoint_weight_merge = self._mechanism_eval = False
        self._is_main_process = True
        self._mechanism_eps, self._lambda_ewc, self._delta_w_penalty_lambda = 1e-12, 0., 0.
    def _unwrap_network(self): return self._network
    def _snapshot_task_reference(self): pass
    def _reset_train_log_stats(self): pass
    def _log_task_start_state(self): pass
    def _current_fisher_drift_value(self): return 0.
    def _should_capture_in_task_fisher(self, epoch): return False
    def _apply_in_task_weight_merge(self, model): pass
    def _apply_endpoint_weight_merge(self, model): pass
    def _log(self, value): pass
    def _build_optimizer(self, params, stage):
        if self.args['stage1_direction'] == 'true_sam':
            return SAM(params, torch.optim.SGD, rho=.05, lr=.04, momentum=0., weight_decay=0.)
        base = torch.optim.SGD(params, lr=.04)
        return GAM(params, base, self._network, grad_rho=.05, grad_norm_rho=self.args['gam_grad_norm_rho'],
                   adaptive=False, args=SimpleNamespace(grad_beta_1=self.args['gam_grad_beta_1'],
                    grad_beta_2=1., grad_beta_3=self.args['gam_grad_beta_3'], grad_gamma=0.))
    def _step_batch_with_log(self, optimizer, inputs, targets, class_offset):
        clean_loss = torch.nn.functional.cross_entropy(self._network(inputs)['logits'], targets)
        torch.autograd.grad(clean_loss, [p for p in self._network.parameters() if p.requires_grad])
        def closure():
            optimizer.zero_grad()
            out = self._network(inputs)
            loss = torch.nn.functional.cross_entropy(out['logits'], targets)
            loss.backward()
            return out, loss.detach()
        out, loss = optimizer.step(closure)
        return out['logits'].detach(), float(loss), {'ce_clean': float(loss), 'gam': float(loss), 'ewc': 0., 'delta_w_penalty': 0.}


def main():
    torch.set_num_threads(1)
    t.logging.getLogger('optimer_PerturabtionType.optimer_sam').setLevel(t.logging.WARNING)
    root = Path(__file__).resolve().parent / 'cpu_checks'
    outputs = []
    for direction in ['ce', 's', 'g', 'sum', 'mean', 'true_sam']:
        t.audit.seed_all(2)
        cli = SimpleNamespace(output_dir=str(root / direction), max_epochs=2, max_batches=1)
        cfg = t.resolve_config({'stage1_direction': direction, 'seed': 2, 'train_num_workers': 0}, cli)
        learner = Learner(cfg)
        ctl = t.Controller(learner, DM(), cfg, cli)
        ctl.install()
        params = [p for p in learner._network.parameters() if p.requires_grad]
        opt = learner._build_optimizer(params, 'init')
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=2, eta_min=.004)
        loader = DataLoader(Data(), batch_size=20, shuffle=False)
        learner._run_regularized_train(loader, loader, opt, sched, epochs=2, class_offset=0)
        assert len(ctl.rows) == 2 and len(ctl.records) == 2
        assert ctl.rows[0]['train_samples'] == 40  # wrapper's common config batch128 uses all 40 toy examples
        assert ctl.counters['training_ce_gradient_passes'] == (4 if direction == 'true_sam' else 10)
        for epoch in [0, 2]:
            path = root / direction / 'checkpoints' / ('epoch_%03d' % epoch)
            state = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=False)
            assert state['epoch'] == epoch and state['task'] == 0
            aa = torch.load(path / 'lora_w_a_0.pt', map_location='cpu', weights_only=False)
            assert len(aa) == 2
        outputs.append({'direction': direction, 'initial_hash': ctl.identity['initial_model_sha256'],
                        'first_batch_hashes': [r['first_batch']['input_sha256'] for r in ctl.rows],
                        'order_hashes': [r['sampler_order_sha256'] for r in ctl.rows],
                        'max_step_error': max(x['actual_step_vs_selected_relative_error'] for x in ctl.records),
                        'gradient_passes': ctl.counters['training_ce_gradient_passes']})
    assert len({r['initial_hash'] for r in outputs}) == 1
    assert len({tuple(r['first_batch_hashes']) for r in outputs}) == 1
    assert len({tuple(r['order_hashes']) for r in outputs}) == 1
    low_lr_results = []
    for direction in ['ce', 's', 'g', 'sum', 'mean', 'true_sam']:
        for lr in [4e-5, 0.0]:
            t.audit.seed_all(2)
            cli = SimpleNamespace(output_dir=str(root / ('low_%s_%g' % (direction, lr))), max_epochs=1, max_batches=1)
            cfg = t.resolve_config({'stage1_direction': direction, 'seed': 2, 'train_num_workers': 0}, cli)
            learner = Learner(cfg)
            learner._network.float()
            ctl = t.Controller(learner, DM(), cfg, cli)
            ctl.install()
            opt = learner._build_optimizer([p for p in learner._network.parameters() if p.requires_grad], 'init')
            for group in opt.param_groups:
                group['lr'] = lr
            ctl.begin_epoch(1)
            learner._step_batch_with_log(opt, torch.randn(8, 4), torch.arange(8), 0)
            record = ctl.records[0]
            assert record['base_sgd_audit']['passed']
            assert record['base_sgd_audit']['predicted_after_max_abs_error'] == 0
            low_lr_results.append({'direction': direction, 'lr': lr, 'passed': True,
                'roundtrip_drift': record['base_sgd_audit']['probe_roundtrip_parameter_norm'],
                'sgd_prediction_error': record['base_sgd_audit']['predicted_after_max_abs_error']})
    report = {'status': 'passed', 'real_optimizers': ['frozen GAM', 'frozen SAM'],
              'native_epoch_loop': 'AST extracted unchanged from as1_normfisher_gam_intask.py',
              'results': outputs, 'all_initial_states_match': True, 'all_batch_data_match': True,
              'float32_low_and_zero_lr_results': low_lr_results}
    t.audit.atomic_json(root.parent / 'cpu_test_report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
