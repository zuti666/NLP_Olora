"""Reviewer-owned analytic cross-coupled nonlinear oracle; CPU only."""
import json
import math
from pathlib import Path
import sys
from unittest.mock import patch

import torch
from transformers.optimization import AdamW

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from optimer_three_point_probe import ThreePointProbe


def analytic(point):
    x, y = map(float, point)
    return (x**3 + x*y, .5*x*x + math.exp(y))


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([.7, -.4]))
        self.register_buffer('forward_count', torch.zeros(()))


def main():
    cases = []
    for mode in ('g0', 'g1_plus', 'backward_reflected', 'central'):
        for clip in (None, .4):
            for first_ready in (False, True):
                model = Model()
                base = model.theta.detach().clone()
                g0 = analytic(base)
                magnitude = math.sqrt(sum(v*v for v in g0))
                points = [tuple(map(float, base))]
                points += [tuple(float(b) + sign*.05*g/(magnitude+1e-12)
                                 for b, g in zip(base, g0)) for sign in (1, -1)]
                gp, gm = analytic(points[1]), analytic(points[2])
                expected = {
                    'g0': g0,
                    'g1_plus': gp,
                    'backward_reflected': tuple(2*a-c for a, c in zip(g0, gm)),
                    'central': tuple(a+.5*(b-c) for a, b, c in zip(g0, gp, gm)),
                }[mode]
                raw = torch.tensor(expected)
                norm = math.sqrt(sum(v*v for v in expected))
                chosen = raw * min(1., clip/(norm+1e-6)) if clip else raw
                optimizer = AdamW(model.parameters(), lr=.001, eps=1e-6,
                                  weight_decay=0., no_deprecation_warning=True)
                probe = ThreePointProbe(model.parameters(), optimizer, model,
                                        mode=mode, radius=.05, eps=1e-12)
                observed = []
                commit_points, commit_gradients = [], []
                real_step = optimizer.step

                def commit(*args, **kwargs):
                    commit_points.append(model.theta.detach().clone())
                    commit_gradients.append(model.theta.grad.detach().clone())
                    return real_step(*args, **kwargs)

                def closure():
                    observed.append(model.theta.detach().clone())
                    probe.zero_grad(set_to_none=True)
                    x, y = model.theta.unbind()
                    loss = x**4/4 + .5*x*x*y + y.exp()
                    model.forward_count.add_(1)
                    loss.backward()
                    return None, loss.detach()

                with patch.object(torch.cuda, 'is_initialized', return_value=False):
                    if first_ready:
                        closure()
                        model.forward_count.zero_()
                    with patch.object(optimizer, 'step', side_effect=commit), patch(
                            'torch.nn.utils.clip_grad_norm_',
                            wraps=torch.nn.utils.clip_grad_norm_) as clipping:
                        probe.step(closure, clip_grad_norm=clip, first_grad_ready=first_ready)
                    assert clipping.call_count == int(clip is not None)
                assert len(observed) == 3
                for actual, wanted in zip(observed, points):
                    torch.testing.assert_close(actual, torch.tensor(wanted), rtol=1e-6, atol=1e-7)
                assert len(commit_points) == 1 and torch.equal(commit_points[0], base)
                assert model.forward_count.item() == 0
                torch.testing.assert_close(commit_gradients[0], chosen, rtol=2e-6, atol=2e-7)
                reference = torch.nn.Parameter(base.clone())
                reference_opt = AdamW([reference], lr=.001, eps=1e-6,
                                     weight_decay=0., no_deprecation_warning=True)
                reference.grad = chosen.clone()
                reference_opt.step()
                torch.testing.assert_close(model.theta, reference, rtol=0., atol=1e-7)
                assert optimizer.state[model.theta]['step'] == 1
                assert probe.real_updates == 1
                assert probe.last_stats['probe_ce_gradient_phases'] == 3
                assert probe.last_stats['probe_replay_closure_calls'] == 2
                assert probe.last_stats['probe_restore_base_exact'] is True
                cases.append({'mode': mode, 'clip': clip, 'first_grad_ready': first_ready,
                              'status': 'passed', 'raw_gradient': expected})
    result = {'status': 'passed', 'cases': cases, 'checks': len(cases),
              'loss': 'x^4/4 + x^2*y/2 + exp(y)',
              'gradient': '[x^3+x*y, x^2/2+exp(y)]',
              'all_points_analytic': True, 'cpu_only': True, 'gpu_actions': False,
              'checks_per_case': ['BASE/plus/minus analytic points', 'selected analytic gradient',
                                  'exact BASE restored before commit', 'forward buffer restored',
                                  'clip exactly once or disabled', 'one HF AdamW commit and moment step',
                                  'independent ordinary HF AdamW reference displacement',
                                  'three reads and two replays']}
    (Path(__file__).parent / 'INDEPENDENT_CPU_ORACLE.json').write_text(
        json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'checks': result['checks'], 'gpu_actions': False}))


if __name__ == '__main__':
    main()
