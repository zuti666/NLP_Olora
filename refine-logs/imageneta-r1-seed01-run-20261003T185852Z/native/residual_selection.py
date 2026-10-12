"""Global C21 direction selection; no training, loader, or RNG operations."""
import math

import torch

C21_SELECTORS = ('c21_base', 'c21_perp')
C21_ORTHOGONALITY_RTOL = 1e-5


@torch.no_grad()
def selected_c21_direction(states, rule):
    """Reduce over every optimizer coordinate, retaining the native dtype.

    The difference is computed in the training dtype. Dot products/norms use
    FP64. No epsilon changes a nonzero denominator and no output is normalized.
    """
    if rule not in C21_SELECTORS or not states:
        raise ValueError('Nonempty all-parameter states and a C21 rule required')
    gradients = {key: [state[key].detach() for state in states]
                 for key in ('g_0', 'g_1', 'g_2', 'g_3')}
    first = gradients['g_0'][0]
    for i in range(len(states)):
        group = [gradients[key][i] for key in gradients]
        if any(value.shape != group[0].shape or value.device != first.device
               or value.dtype != first.dtype for value in group):
            raise ValueError('Inconsistent C21 gradient shape/device/dtype')
        if not all(bool(torch.isfinite(value).all()) for value in group):
            raise FloatingPointError('Nonfinite native C21 gradient')
    delta = [g2-g1 for g2, g1 in zip(gradients['g_2'], gradients['g_1'])]

    def dot(xs, ys):
        return float(torch.stack([(x.double()*y.double()).sum()
                                  for x, y in zip(xs, ys)]).sum())

    def norm(xs):
        squared = dot(xs, xs)
        if not math.isfinite(squared) or squared < 0:
            raise FloatingPointError('Nonfinite C21 squared norm')
        return math.sqrt(squared)

    numerator = dot(gradients['g_1'], delta)
    denominator = dot(gradients['g_1'], gradients['g_1'])
    if not math.isfinite(numerator) or not math.isfinite(denominator) or denominator < 0:
        raise FloatingPointError('Nonfinite C21 projection reduction')
    coefficient = 0. if denominator == 0. else numerator/denominator
    if not math.isfinite(coefficient):
        raise FloatingPointError('Nonfinite C21 projection coefficient')
    perpendicular = [d-coefficient*g1 for d, g1 in zip(delta, gradients['g_1'])]
    selected = ([g0+d for g0, d in zip(gradients['g_0'], delta)]
                if rule == 'c21_base'
                else [g1+residual for g1, residual in zip(gradients['g_1'], perpendicular)])
    if not all(bool(torch.isfinite(value).all()) for value in delta+perpendicular+selected):
        raise FloatingPointError('Nonfinite C21 residual/selected gradient')
    norms = {key.replace('_', ''): norm(values) for key, values in gradients.items()}
    norms.update(c21=norm(delta), perp=norm(perpendicular), selected=norm(selected))
    c10 = [g1-g0 for g1, g0 in zip(gradients['g_1'], gradients['g_0'])]
    norm_c10 = norm(c10)
    angle_scale = norm_c10*norms['c21']
    cosine = dot(c10, delta)/angle_scale if angle_scale > 0. else 0.
    orthogonality_dot = dot(gradients['g_1'], perpendicular)
    orthogonality_scale = norms['g1']*norms['c21']
    error = abs(orthogonality_dot)/orthogonality_scale if orthogonality_scale > 0. else 0.
    record = {'schema_version': 1, 'rule': rule,
              'projection_scope': 'all_trainable_LoRA_and_seen_FC',
              'reduction_dtype': 'float64', 'tensor_count': len(states),
              'parameter_count': sum(g.numel() for g in gradients['g_1']),
              'formula': 'g0+(g2-g1)' if rule == 'c21_base' else 'g1+((g2-g1)-a*g1)',
              'coefficient_a': coefficient, 'dot_g1_c21': numerator,
              'norm_g1_sq': denominator, 'g1_zero': denominator == 0.,
              'c21_zero': norms['c21'] == 0., 'norms': norms,
              'norm_c10': norm_c10, 'norm_c21_parallel': abs(coefficient)*norms['g1'],
              'cos_c10_c21': cosine, 'cos_c10_c21_defined': angle_scale > 0.,
              'angle_c10_c21_degrees': math.degrees(math.acos(max(-1., min(1., cosine))))
                  if angle_scale > 0. else None,
              'dot_g1_perp': orthogonality_dot,
              'perp_over_c21': norms['perp']/norms['c21'] if norms['c21'] > 0. else 0.,
              'orthogonality_relative_error': error,
              'orthogonality_tolerance': C21_ORTHOGONALITY_RTOL,
              'orthogonality_passed': error <= C21_ORTHOGONALITY_RTOL,
              'final_norm_matching': False}
    scalars = [coefficient, numerator, denominator, norm_c10, cosine,
               orthogonality_dot, error, record['perp_over_c21'], *norms.values()]
    if any(not math.isfinite(value) for value in scalars) or not record['orthogonality_passed']:
        raise AssertionError('C21 global projection orthogonality/finite check failed')
    return selected, record
