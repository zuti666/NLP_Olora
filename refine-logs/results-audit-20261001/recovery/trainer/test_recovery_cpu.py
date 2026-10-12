"""Recovery-v2 regression: separate own-SAM validity from cross-GAM comparison."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace, MethodType
import torch
import task0_trainer as t
import test_task0_cpu as base


def setup(name):
    t.audit.seed_all(13)
    cli = SimpleNamespace(output_dir=str(Path(__file__).resolve().parent / 'recovery_cpu_checks' / name), max_epochs=1, max_batches=1)
    cfg = t.resolve_config({'stage1_direction': 'true_sam', 'seed': 2, 'train_num_workers': 0}, cli)
    learner = base.Learner(cfg)
    learner._network.float()
    ctl = t.Controller(learner, base.DM(), cfg, cli)
    ctl.install()
    opt = learner._build_optimizer([p for p in learner._network.parameters() if p.requires_grad], 'init')
    ctl.begin_epoch(1)
    x, y = torch.randn(12, 4), torch.arange(12)
    return learner, ctl, opt, x, y


def expect_failure(name, install):
    learner, ctl, opt, x, y = setup(name)
    restore = install(learner, opt)
    try:
        learner._step_batch_with_log(opt, x, y, 0)
    except (AssertionError, FloatingPointError) as exc:
        return {'case': name, 'rejected': True, 'reason': str(exc)}
    finally:
        if restore:
            restore()
    raise AssertionError(name + ' was not rejected')


def main():
    torch.set_num_threads(1)
    t.logging.getLogger('optimer_PerturabtionType.optimer_sam').setLevel(t.logging.WARNING)
    learner, ctl, opt, x, y = setup('baseline_native_equivalence')
    reference = copy.deepcopy(learner._network)
    rng = t.audit.rng_get()
    learner._step_batch_with_log(opt, x, y, 0)
    actual_rng = t.audit.rng_get()
    ref_opt = base.SAM([p for p in reference.parameters() if p.requires_grad], torch.optim.SGD,
                       rho=.05, adaptive=False, lr=.04, momentum=0., weight_decay=0.)
    t.audit.rng_set(rng)
    ref_opt.zero_grad()
    torch.nn.functional.cross_entropy(reference(x)['logits'], y).backward()
    ref_opt.first_step(zero_grad=True)
    torch.nn.functional.cross_entropy(reference(x)['logits'], y).backward()
    ref_opt.second_step()
    assert all(torch.equal(a, b) for a, b in zip(learner._network.parameters(), reference.parameters()))
    assert t.audit.rng_equal(actual_rng, t.audit.rng_get())
    baseline = ctl.records[0]
    assert baseline['self_sam_audit']['probe_parameter_check']['all_tensors_exact']
    assert baseline['self_sam_audit']['actual_probe_gradient_check']['passed']
    assert baseline['base_sgd_audit']['predicted_after_max_abs_error'] == 0

    original_isolated = t.isolated_gam
    def different_gam(*args, **kwargs):
        grads, points = original_isolated(*args, **kwargs)
        grads['g1'] = grads['g1'] * 1.01
        return grads, points
    t.isolated_gam = different_gam
    try:
        learner2, ctl2, opt2, x2, y2 = setup('cross_algorithm_difference_is_measurement')
        learner2._step_batch_with_log(opt2, x2, y2, 0)
        cross = ctl2.records[0]['cross_algorithm_comparison']
        assert cross['sam_g1_vs_isolated_gam_g1_relative_error'] > .009
        assert not cross['used_as_correctness_gate']
        assert ctl2.records[0]['selected_gradient_relative_error'] < 1e-6
        assert all(torch.equal(a, b) for a, b in zip(learner2._network.parameters(), learner._network.parameters()))
    finally:
        t.isolated_gam = original_isolated

    def corrupt_probe(learner, opt):
        original = opt.first_step
        def bad(_opt, *args, **kwargs):
            result = original(*args, **kwargs)
            with torch.no_grad():
                _opt.param_groups[0]['params'][0].add_(.01)
            return result
        opt.first_step = MethodType(bad, opt)
        return None
    def corrupt_reference(learner, opt):
        original = t.independent_probe_gradient
        t.independent_probe_gradient = lambda *a, **k: [g + .01 for g in original(*a, **k)]
        return lambda: setattr(t, 'independent_probe_gradient', original)
    def nonfinite_gradient(learner, opt):
        handle = opt.param_groups[0]['params'][0].register_hook(lambda grad: grad * float('nan'))
        return handle.remove
    negative = [expect_failure('incorrect_rho_probe', corrupt_probe),
                expect_failure('incorrect_self_gradient', corrupt_reference),
                expect_failure('nonfinite_gradient', nonfinite_gradient)]

    # Same sampled gradient, different floating-point reduction orders.
    reduction_examples = []
    for seed in range(20):
        generator = torch.Generator().manual_seed(seed)
        tensors = [torch.randn(10, 768, generator=generator) * (10 ** ((i % 7) - 3)) for i in range(48)]
        sam_norm = torch.stack([v.norm() for v in tensors]).norm()
        square_sum = None
        for v in tensors:
            term = (v ** 2).sum()
            square_sum = term if square_sum is None else square_sum + term
        gam_norm = square_sum.sqrt()
        if sam_norm.item() != gam_norm.item():
            reduction_examples.append({'seed': seed, 'sam_norm': float(sam_norm), 'gam_norm': float(gam_norm),
                                       'relative_difference': float(abs(sam_norm - gam_norm) / sam_norm)})
    assert reduction_examples
    report = {'status': 'passed', 'version': 2, 'native_SAM_parameter_and_rng_exact_match': True,
              'self_probe_exact': True, 'self_gradient_check': baseline['self_sam_audit']['actual_probe_gradient_check'],
              'base_SGD_prediction_max_abs': 0., 'injected_cross_algorithm_comparison': cross,
              'cross_difference_does_not_change_training_update': True,
              'negative_checks': negative, 'norm_reduction_examples': reduction_examples,
              'limitations': 'Synthetic CPU checks show semantic separation and plausible finite-precision norm differences; they do not quantitatively reproduce the observed ViT GPU epoch13 difference.'}
    t.audit.atomic_json(Path(__file__).resolve().parent / 'RECOVERY_CPU_TEST_REPORT.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
