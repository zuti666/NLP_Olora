#!/usr/bin/env python3
"""Bounded seed0 radius extension using the unchanged historical Runner."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
EXECUTION = HERE.parents[1] / 'execution'
LEGACY = EXECUTION / 'results_seed0_recovery1'
PYTHON = Path('/data/115-2/users/liying/conda_storage/envs/Pilot_new/bin/python')
sys.path.insert(0, str(EXECUTION))

import torch
import torchvision
import timm
from diagnostic import Journal, Runner
from rq2_common import atomic_json, canonical_hash, random_direction, sha_file


def read_json(path):
    return json.loads(Path(path).read_text())


def historical_rows(method):
    return {r['state_id']: r for r in
            (json.loads(line) for line in (LEGACY / method / 'neighborhood/states.jsonl').read_text().splitlines())}


def compare_measurements(current, reference):
    """FP32 repeat tolerances; preserve every observed error before raising."""
    exact_fields = ('state_id', 'method', 'amount', 'point', 'coordinate_sha256',
                    'point_effective_sha256', 'data_manifest_sha256', 'checkpoint_sha256')
    exact = {key: current.get(key) == reference.get(key) for key in exact_fields}
    for key in ('coordinate', 'estimator', 'floor', 'gradient_batch_size', 'normalized',
                'regularizer', 'samples'):
        exact['fisher.' + key] = current['fisher'][key] == reference['fisher'][key]
    errors = {}

    def number(name, actual, expected, atol, rtol=0.0):
        absolute = abs(actual - expected)
        allowed = atol + rtol * abs(expected)
        errors[name] = dict(current=actual, historical=expected, absolute_error=absolute,
                            relative_error=absolute / max(abs(expected), 1e-30),
                            atol=atol, rtol=rtol, allowed_error=allowed,
                            passed=math.isfinite(actual) and absolute <= allowed)

    for split in ('train', 'test'):
        exact[f'metrics.{split}.samples'] = current['metrics'][split]['samples'] == reference['metrics'][split]['samples']
        number(f'metrics.{split}.loss', current['metrics'][split]['loss'], reference['metrics'][split]['loss'], 3e-6)
        number(f'metrics.{split}.accuracy', current['metrics'][split]['accuracy'], reference['metrics'][split]['accuracy'], 1e-10)
    number('fisher.subset_loss', current['fisher']['subset_loss'], reference['fisher']['subset_loss'], 3e-6)
    number('fisher.raw_trace', current['fisher']['raw_trace'], reference['fisher']['raw_trace'], 1e-6, 1e-4)
    for method in ('sgd', 'gam'):
        number('fisher.q.' + method, current['fisher']['q'][method], reference['fisher']['q'][method], 1e-8, 1e-4)
    return dict(passed=all(exact.values()) and all(item['passed'] for item in errors.values()),
                exact_matches=exact, numerical_errors=errors,
                note='Timings and protocol hashes are intentionally excluded; protocol changes only for output_dir and radii.')


class RadiusRunner(Runner):
    def __init__(self, args):
        if not __debug__:
            raise RuntimeError('Python optimization disables required historical assertions')
        config = read_json(args.config)
        original = read_json(EXECUTION / 'diagnostic_config_recovery1.json')
        expected = dict(original, output_dir=str(HERE / 'results'), radii=[0, *range(11, 20)])
        assert config == expected, 'Only output_dir and radii may differ from the historical configuration'
        assert Path(sys.executable).resolve() == PYTHON.resolve(), 'Use the pinned Pilot_new interpreter'
        assert torch.__version__ == '2.4.1+cu121'
        assert torchvision.__version__ == '0.19.1+cu121'
        assert timm.__version__ == '1.0.9'
        self.legacy_meta = read_json(LEGACY / args.method / 'neighborhood/manifest.json')
        self.legacy_rows = historical_rows(args.method)
        assert self.legacy_meta['config'] == original
        files = [EXECUTION / 'diagnostic_config_recovery1.json']
        files += [LEGACY / args.method / name for name in
                  ('neighborhood/manifest.json', 'neighborhood/states.jsonl', 'neighborhood/summary.json',
                   'data_manifest.json', 'sanity/summary.json')]
        self.historical_evidence = {str(path): sha_file(path) for path in files}
        self.wrapper_sha256 = sha_file(Path(__file__))
        super().__init__(args)
        keys = sorted(set(self.legacy_meta) - {'config', 'protocol_sha256'})
        matches = {key: self.meta.get(key) == self.legacy_meta[key] for key in keys}
        matches['complete_data_manifest'] = self.data == read_json(LEGACY / args.method / 'data_manifest.json')
        matches['data_manifest_self_hash'] = self.data['manifest_sha256'] == canonical_hash(
            {key: value for key, value in self.data.items() if key != 'manifest_sha256'})
        self.identity_guard = dict(passed=all(matches.values()), exact_matches=matches,
            historical_protocol_sha256=self.legacy_meta['protocol_sha256'],
            protocol_sha256=self.protocol, historical_evidence_sha256=self.historical_evidence,
            extension_runner_sha256=self.wrapper_sha256,
            allowed_config_changes=['output_dir', 'radii'],
            environment=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                             torchvision=torchvision.__version__, timm=timm.__version__,
                             cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')))
        atomic_json(self.directory / 'identity_guard.json', self.identity_guard)
        assert self.identity_guard['passed'], f'Historical identity mismatch: {[key for key, value in matches.items() if not value]}'

    def measure(self, state_id, perturbation=None, amount=0.0, **details):
        if details.get('kind') == 'neighborhood':
            index = details['direction']['index']
            historical_direction = self.legacy_rows[f'r10_u{index:02d}']['point']['direction']
            assert details['direction'] == historical_direction, f'Direction {index} differs from the historical experiment'
        return super().measure(state_id, perturbation, amount, **details)

    def origin(self):
        row = super().origin()
        comparison = compare_measurements(row, self.legacy_rows['r0'])
        atomic_json(self.directory / 'anchor_comparison.json', comparison)
        assert comparison['passed'], 'Fresh extension anchor differs from the historical anchor; inspect anchor_comparison.json'
        return row

    def finish(self, payload):
        if self.args.job == 'sanity' and not getattr(self, 'extension_gate_ready', False):
            # Keep the original sanity report, but do not publish a passing queue
            # receipt until the old-radius forward/Fisher repeat also passes.
            self.builtin_sanity_payload = payload
            atomic_json(self.directory / 'builtin_sanity_summary.json',
                        dict(status='complete', protocol_sha256=self.protocol, **payload))
            return
        payload = dict(payload, radius_extension_guard_passed=True,
                       identity_guard_passed=self.identity_guard['passed'],
                       extension_runner_sha256=self.wrapper_sha256,
                       historical_evidence_sha256=self.historical_evidence)
        if self.args.job == 'neighborhood':
            expected = {'r0'} | {f'r{radius}_u{index:02d}' for radius in range(11, 20) for index in range(16)}
            assert set(self.journal.rows) == expected, 'Unexpected or missing extension states'
            payload.update(new_perturbation_states=144, radii=[0, *range(11, 20)],
                           summary_scope='Extension radii only; merged legacy + extension aggregation is separate.')
        return super().finish(payload)

    def sanity(self):
        super().sanity()
        directory, journal = self.directory, self.journal
        repeat_directory = self.output / 'repeat_witness'
        repeat_directory.mkdir(exist_ok=True)
        atomic_json(repeat_directory / 'manifest.json', self.meta)
        try:
            self.directory = repeat_directory
            self.journal = Journal(repeat_directory, self.protocol)
            direction, info = random_direction(0, self.config['direction_seed'], self.device)
            repeated = self.measure('r10_u00', direction, 10.0, kind='neighborhood', radius=10.0, direction=info)
            comparison = compare_measurements(repeated, self.legacy_rows['r10_u00'])
            repeat_receipt = dict(status='complete' if comparison['passed'] else 'failed',
                method=self.args.method, job='repeat_witness', protocol_sha256=self.protocol,
                state_id='r10_u00', states_completed=len(self.journal.rows),
                comparison=comparison, passed=comparison['passed'],
                historical_evidence_sha256=self.historical_evidence,
                extension_runner_sha256=self.wrapper_sha256)
            atomic_json(repeat_directory / 'summary.json', repeat_receipt)
            atomic_json(repeat_directory / 'progress.json', repeat_receipt)
            assert comparison['passed'], 'Historical r10_u00 repeat failed; inspect repeat_witness/summary.json'
        finally:
            self.directory, self.journal = directory, journal
        self.extension_gate_ready = True
        self.finish(dict(self.builtin_sanity_payload, old_radius_repeat_passed=True,
                         old_radius_repeat_receipt=str(repeat_directory / 'summary.json')))

    def require_sanity(self):
        super().require_sanity()
        receipt = read_json(self.output / 'sanity/summary.json')
        repeat = read_json(self.output / 'repeat_witness/summary.json')
        assert receipt['radius_extension_guard_passed'] and receipt['old_radius_repeat_passed']
        for evidence in (receipt, repeat):
            assert evidence['status'] == 'complete' and evidence['passed'] is True
            assert evidence['protocol_sha256'] == self.protocol
            assert evidence['historical_evidence_sha256'] == self.historical_evidence
            assert evidence['extension_runner_sha256'] == self.wrapper_sha256
        assert repeat['comparison']['passed'] is True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE / 'configs/radius_seed0.json')
    parser.add_argument('--method', choices=('sgd', 'gam'), required=True)
    parser.add_argument('--job', choices=('sanity', 'neighborhood'), required=True)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    try:
        runner = RadiusRunner(args)
        getattr(runner, args.job)()
    except Exception as exc:
        atomic_json(HERE / 'results' / args.method / args.job / 'failure.json',
                    dict(status='failed', method=args.method, job=args.job,
                         error=repr(exc), radius_extension_guard_passed=False))
        raise


if __name__ == '__main__':
    main()
