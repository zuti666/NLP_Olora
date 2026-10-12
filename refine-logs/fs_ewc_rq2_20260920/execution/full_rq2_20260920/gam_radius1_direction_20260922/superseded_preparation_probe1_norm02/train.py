"""Cold-start GAM Task0: probe rho 1.0, gradient-norm rho unchanged at 0.2."""
from pathlib import Path
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
spec = importlib.util.spec_from_file_location('rq2_formal_training_rho1', BASE / 'train.py')
formal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(formal)
original_parser = formal.parser
original_config = formal.build_config
original_runtime = formal.install_runtime


def parser():
    result = original_parser()
    result.set_defaults(output_root=HERE / 'training', workers=4)
    return result


def build_config(ns):
    if ns.mode != 'anchor' or ns.method != 'gam':
        raise ValueError('This intervention only defines cold-start GAM Task0 training.')
    if ns.gpu is not None and ns.gpu not in ('0', '1'):
        raise ValueError('Only physical GPUs 0 and 1 are authorized for this experiment.')
    if ns.workers != 4:
        raise ValueError('Keep the formal four-worker data protocol.')
    cfg, run_dir = original_config(ns)
    original = dict(cfg)
    # as1_rho is the active override used by the instantiated learner.
    cfg.update(gam_grad_rho=1.0, as1_rho=1.0)
    assert cfg['gam_grad_norm_rho'] == cfg['as1_norm_rho'] == .2
    assert {k for k in cfg if cfg[k] != original[k]} == {'gam_grad_rho', 'as1_rho'}
    cfg['rq2_radius_intervention'] = {
        'probe_radius_before': .05, 'probe_radius_after': 1.0,
        'norm_radius_unchanged': .2,
        'wrapper_sha256': formal.sha256(__file__),
        'formal_runner_sha256': formal.sha256(BASE / 'train.py'),
        'user_authorized_gpus': [0, 1],
        'initialization': 'fresh paired Task0 LoRA and head; same frozen backbone',
    }
    return cfg, run_dir


def install_runtime(ns, cfg, run_dir):
    runtime = original_runtime(ns, cfg, run_dir)
    construct = runtime['factory'].get_model

    def checked_factory(name, args):
        model = construct(name, args)
        observed = {
            'gam_grad_rho': float(model._gam_grad_rho),
            'as1_rho': float(model._as1_rho),
            'optimizer_args_grad_rho': float(model._gam_args.grad_rho),
            'gam_grad_norm_rho': float(model._gam_grad_norm_rho),
            'as1_norm_rho': float(model._as1_norm_rho),
            'optimizer_args_grad_norm_rho': float(model._gam_args.grad_norm_rho),
        }
        assert list(observed.values()) == [1., 1., 1., .2, .2, .2], observed
        record = dict(status='passed', event='effective_gam_radii_verified',
                      seed=ns.seed, observed=observed, wrapper_sha256=formal.sha256(__file__))
        formal.write_json(run_dir / 'RADIUS_VERIFIED.json', record)
        formal.append_json(run_dir / 'runtime_receipts.jsonl', record)
        build_optimizer = model._build_optimizer

        def checked_optimizer(*a, **kw):
            optimizer = build_optimizer(*a, **kw)
            probe, norm = float(optimizer.grad_rho), float(optimizer.grad_norm_rho)
            assert abs(probe - 1.) < 1e-7 and abs(norm - .2) < 1e-7
            formal.append_json(run_dir / 'runtime_receipts.jsonl', {
                'event': 'actual_optimizer_radii_verified', 'seed': ns.seed,
                'probe_radius': probe, 'norm_radius': norm,
                'cpu_construction_only': ns.validate_cpu,
                'wrapper_sha256': formal.sha256(__file__),
            })
            return optimizer

        model._build_optimizer = checked_optimizer
        if ns.validate_cpu:
            import torch
            checked_optimizer([torch.nn.Parameter(torch.zeros(1))], stage='init')
        return model

    runtime['factory'].get_model = checked_factory
    return runtime


formal.parser = parser
formal.build_config = build_config
formal.install_runtime = install_runtime

if __name__ == '__main__':
    formal.main()
