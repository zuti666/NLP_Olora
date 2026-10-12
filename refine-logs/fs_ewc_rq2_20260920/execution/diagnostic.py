#!/usr/bin/env python3
"""RQ2 seed0 offline diagnostics. See --help; all outputs are resumable."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from rq2_common import (ROOT, atomic_json, canonical_hash, checkpoint_path,
                        convert_effective, coordinates, effective_weights, evaluate,
                        evaluate_local_and_seen, historical_tf32_conv_reference,
                        fisher, load_network, norm_sq, prepare_data, random_direction,
                        read_state, set_point, sha_file, tensor_hash, unit_direction)


class Journal:
    def __init__(self, directory, protocol):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'states.jsonl'
        self.protocol = protocol
        self.rows = {}
        if self.path.exists():
            raw = self.path.read_bytes()
            if raw and not raw.endswith(b'\n'):
                # An interrupted final write is uncommitted, unlike complete rows.
                raw = raw[:raw.rfind(b'\n') + 1]
                self.path.write_bytes(raw)
            for line in raw.splitlines():
                row = json.loads(line)
                assert row['protocol_sha256'] == protocol, 'Cannot resume a different measurement protocol'
                assert row['state_id'] not in self.rows
                self.rows[row['state_id']] = row

    def add(self, row):
        row = dict(row, protocol_sha256=self.protocol)
        assert row['state_id'] not in self.rows
        with self.path.open('a') as f:
            f.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
            f.flush(); os.fsync(f.fileno())
        self.rows[row['state_id']] = row
        atomic_json(self.directory / 'progress.json', {'status': 'running', 'states_completed': len(self.rows),
                    'latest_state': row['state_id'], 'updated_unix': time.time()})
        print(json.dumps({'event': 'state_complete', 'state': row['state_id'], 'completed': len(self.rows),
                          'raw_trace': row.get('fisher',{}).get('raw_trace'), 'seconds': row.get('seconds')}), flush=True)
        return row


class Runner:
    def __init__(self, args):
        self.args = args
        self.config = json.loads(Path(args.config).read_text())
        self.device = torch.device(args.device)
        torch.set_num_threads(int(self.config.get('cpu_threads', 4)))
        torch.manual_seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.source = Path(self.config.get('source_path', ROOT / 'source'))
        self.output = Path(self.config.get('output_dir', ROOT / 'results')) / args.method
        self.output.mkdir(parents=True, exist_ok=True)
        print(json.dumps({'event': 'loading_data', 'method': args.method, 'job': args.job}), flush=True)
        self.x, self.y, self.data = prepare_data(self.config, self.output)
        self.paths = {m: {t: checkpoint_path(m, t, self.config) for t in (0, 9)} for m in ('sgd', 'gam')}
        self.checksums = {m: {str(t): sha_file(p) for t, p in paths.items()} for m, paths in self.paths.items()}
        self.delta_cpu, self.drift = {}, {}
        for m in ('sgd', 'gam'):
            s0, s9 = read_state(self.paths[m][0]), read_state(self.paths[m][9])
            assert torch.equal(s0['fc.weight'], s9['fc.weight'][:20])
            assert torch.equal(s0['fc.bias'], s9['fc.bias'][:20])
            for key, value in s0.items():
                if not key.startswith('fc.') and 'linear_a_' not in key and 'linear_b_' not in key:
                    assert torch.equal(value, s9[key]), f'Changed fixed backbone: {m}/{key}'
            self.delta_cpu[m] = [b-a for a,b in zip(effective_weights(s0), effective_weights(s9))]
            self.drift[m] = norm_sq(self.delta_cpu[m]) ** 0.5
        del s0, s9
        self.deltas = {m: [p.to(self.device) for p in d] for m,d in self.delta_cpu.items()}
        self.model = load_network(self.paths[args.method][0], self.source, self.device)
        with torch.no_grad():
            self.original_logits = self.model(self.x['test'][:8].to(self.device)).cpu()
        self.collector, self.params = convert_effective(self.model)
        self.anchor = [p.detach().clone() for p in self.params]
        self.meta = {'method': args.method, 'seed': 0, 'source_path': str(self.source),
                     'source_files': {str(p.relative_to(self.source)): sha_file(p) for p in
                                      (self.source/'backbone/lora.py', self.source/'utils/data.py')},
                     'diagnostic_code_sha256': sha_file(Path(__file__)),
                     'common_code_sha256': sha_file(ROOT/'rq2_common.py'),
                     'checkpoints': {m: {str(t): str(p) for t,p in paths.items()} for m,paths in self.paths.items()},
                     'checkpoint_sha256': self.checksums, 'data_manifest_sha256': self.data['manifest_sha256'],
                     'coordinate_order': coordinates(), 'coordinate_numel': 14155776,
                     'coordinate_sha256': canonical_hash(coordinates()),
                     'head_sha256': tensor_hash([self.model.fc.weight[:20], self.model.fc.bias[:20]]),
                     'anchor_effective_sha256': tensor_hash(self.anchor),
                     'actual_direction_sha256': {m: tensor_hash(d) for m,d in self.delta_cpu.items()},
                     'actual_drift': self.drift, 'dtype': 'float32, float64 scalar reductions',
                     'production_precision':'full FP32; cuDNN TF32 and matmul TF32 disabled; no operand emulation',
                     'historical_precision_witness':'TF32-rounded patch-convolution input and weight, validation only',
                     'fisher_definition': 'mean_i g_i g_i^T; g_i=d CE_i/d effective_QV; no normalization/floor/regularizer',
                     'config': self.config, 'torch_version': torch.__version__}
        self.protocol = canonical_hash(self.meta)
        self.meta['protocol_sha256'] = self.protocol
        self.directory = self.output / args.job
        self.directory.mkdir(exist_ok=True)
        atomic_json(self.directory/'manifest.json', self.meta)
        self.journal = Journal(self.directory, self.protocol)
        self.fisher_batch = int(self.config.get('fisher_batch_size', 8))
        self.eval_batch = int(self.config.get('eval_batch_size', 64))

    def evaluate(self, split):
        return evaluate(self.model, self.x[split], self.y[split], self.device, self.eval_batch)

    def measure(self, state_id, perturbation=None, amount=0.0, **details):
        if state_id in self.journal.rows:
            return self.journal.rows[state_id]
        started = time.monotonic()
        set_point(self.params, self.anchor, perturbation, amount)
        data = {split: self.evaluate(split) for split in ('train','test')}
        sensitivity = fisher(self.model, self.collector, self.params, self.x['subset'], self.y['subset'],
                             self.deltas, self.device, self.fisher_batch)
        return self.journal.add({'state_id': state_id, 'method': self.args.method, 'point': details,
                                'amount': amount, 'metrics': data, 'fisher': sensitivity,
                                'coordinate_sha256': self.meta['coordinate_sha256'],
                                'point_effective_sha256': tensor_hash(self.params),
                                'data_manifest_sha256': self.data['manifest_sha256'],
                                'checkpoint_sha256': self.checksums[self.args.method],
                                'seconds': time.monotonic()-started})

    def origin(self):
        cached = self.output/'anchor_measurement.json'
        if cached.exists():
            row = json.loads(cached.read_text())
            assert row['protocol_sha256'] == self.protocol, 'Origin protocol mismatch'
            if 'r0' not in self.journal.rows:
                self.journal.add(dict(row, reused_origin=True))
            return self.journal.rows['r0']
        row = self.measure('r0', kind='anchor', radius=0)
        atomic_json(cached, row)
        return row

    def finish(self, payload):
        result = {'status':'complete', 'method':self.args.method, 'job':self.args.job,
                  'protocol_sha256':self.protocol, 'states_completed':len(self.journal.rows), **payload}
        atomic_json(self.directory/'summary.json', result)
        atomic_json(self.directory/'progress.json', dict(result, updated_unix=time.time()))
        print(json.dumps({'event':'job_complete', **result}, allow_nan=False), flush=True)

    def require_sanity(self):
        path = self.output/'sanity/summary.json'
        assert path.exists(), 'Run --job sanity before production measurements'
        receipt = json.loads(path.read_text())
        assert receipt['status'] == 'complete' and receipt['passed'] is True
        assert receipt['protocol_sha256'] == self.protocol, 'Sanity receipt uses a different protocol'

    def sanity(self):
        checks = {}
        set_point(self.params, self.anchor)
        with torch.no_grad():
            converted = self.model(self.x['test'][:8].to(self.device)).cpu()
        err = float((converted-self.original_logits).abs().max())
        assert err < 3e-5, f'Effective conversion logits mismatch: {err}'
        checks['source_to_effective_max_logit_error'] = err
        ref = self.origin()
        repeated = self.evaluate('test')
        assert abs(repeated['loss']-ref['metrics']['test']['loss']) < 1e-9
        checks['repeat_test_loss_difference'] = repeated['loss']-ref['metrics']['test']['loss']
        set_point(self.params, self.anchor, self.deltas[self.args.method], 0.0)
        assert all(torch.equal(a,p) for a,p in zip(self.anchor,self.params))
        checks['r0_exact'] = True
        historical = {}
        with (ROOT.parent/'task_trajectory_recomputed.csv').open() as f:
            for row in csv.DictReader(f):
                if row['method'] == self.args.method and int(row['task']) in (0,9):
                    historical[int(row['task'])] = row
        t0_error = ref['metrics']['test']['loss']-float(historical[0]['test_local_loss'])
        checks['native_FP32_vs_historical_t0'] = {
            'loss_difference':t0_error,
            'accuracy_difference_pp':ref['metrics']['test']['accuracy']-float(historical[0]['test_local_acc']),
            'interpretation':'Historical endpoints are reproduced by TF32 patch-convolution operand rounding; current production remains full FP32.'}
        historical_ref0=historical_tf32_conv_reference(self.model,self.x['test'],self.y['test'],self.device,self.eval_batch)
        ref0_error=historical_ref0['loss']-float(historical[0]['test_local_loss'])
        ref0_acc_error=historical_ref0['accuracy']-float(historical[0]['test_local_acc'])
        assert abs(ref0_error)<3e-5 and abs(ref0_acc_error)<1e-10, (ref0_error,ref0_acc_error)
        checks['historical_TF32_t0_reference']={**historical_ref0,'loss_difference':ref0_error,
            'accuracy_difference_pp':ref0_acc_error,'passed':True}
        endpoint = load_network(self.paths[self.args.method][9], self.source, self.device)
        direct = {split:evaluate(endpoint,self.x[split],self.y[split],self.device,self.eval_batch)
                  for split in ('train','test')}
        with torch.no_grad():
            endpoint_logits = endpoint(self.x['test'][:8].to(self.device)).cpu()
        historical_ref9=historical_tf32_conv_reference(endpoint,self.x['test'],self.y['test'],self.device,self.eval_batch)
        ref9_error=historical_ref9['loss']-float(historical[9]['test_local_loss'])
        ref9_acc_error=historical_ref9['accuracy']-float(historical[9]['test_local_acc'])
        assert abs(ref9_error)<3e-5 and abs(ref9_acc_error)<1e-10, (ref9_error,ref9_acc_error)
        checks['historical_TF32_t9_reference']={**historical_ref9,'loss_difference':ref9_error,
            'accuracy_difference_pp':ref9_acc_error,'passed':True}
        del endpoint
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        set_point(self.params,self.anchor,self.deltas[self.args.method],1.0)
        with torch.no_grad():
            path_logits = self.model(self.x['test'][:8].to(self.device)).cpu()
        err = float((endpoint_logits-path_logits).abs().max())
        assert err < 5e-5, f'Task9 straight path endpoint mismatch: {err}'
        checks['s1_direct_endpoint_max_logit_error'] = err
        interp = {split:self.evaluate(split) for split in ('train','test')}
        checks['s1_loss_errors'] = {split:interp[split]['loss']-direct[split]['loss'] for split in direct}
        assert all(abs(v) < 3e-6 for v in checks['s1_loss_errors'].values())
        t9_error = direct['test']['loss']-float(historical[9]['test_local_loss'])
        checks['native_FP32_vs_historical_t9'] = {'loss_difference':t9_error,
            'accuracy_difference_pp':direct['test']['accuracy']-float(historical[9]['test_local_acc']),
            'interpretation':'Historical endpoints are reproduced by TF32 patch-convolution operand rounding; current production remains full FP32.'}
        checks['direct_endpoint_metrics'] = direct
        set_point(self.params,self.anchor)
        xx, yy = self.x['subset'][:2], self.y['subset'][:2]
        batched = fisher(self.model,self.collector,self.params,xx,yy,self.deltas,self.device,2,True)
        references = []
        for i in range(2):
            loss = F.cross_entropy(self.model(xx[i:i+1].to(self.device)),yy[i:i+1].to(self.device))
            grads = torch.autograd.grad(loss,self.params)
            trace = sum(float(g.square().sum(dtype=torch.float64)) for g in grads)
            dots = {m:sum(float((g*d).sum(dtype=torch.float64)) for g,d in zip(grads,ds))
                    for m,ds in self.deltas.items()}
            references.append({'trace':trace,'dots':dots})
        relative_errors = [abs(batched['sample_trace'][i]-references[i]['trace'])/max(references[i]['trace'],1e-12)
                           for i in range(2)]
        dot_errors = {m:[abs(batched['sample_dots'][m][i]-references[i]['dots'][m]) for i in range(2)]
                      for m in self.deltas}
        assert max(relative_errors) < 1e-4, relative_errors
        assert max(v for errs in dot_errors.values() for v in errs) < 1e-5, dot_errors
        checks['per_example_trace_relative_errors'] = relative_errors
        checks['per_example_direction_dot_absolute_errors'] = dot_errors
        # Central finite difference for each individual loss, not square of mean loss.
        unit, drift = unit_direction(self.deltas[self.args.method])
        fd = []
        expected = torch.tensor([r['dots'][self.args.method]/drift for r in references], dtype=torch.float64)
        for eps in (0.01,0.005,0.001):
            values = []
            for sign in (-1,1):
                set_point(self.params,self.anchor,unit,sign*eps)
                with torch.no_grad():
                    values.append(F.cross_entropy(self.model(xx.to(self.device)),yy.to(self.device),reduction='none').double().cpu())
            approx = (values[1]-values[0])/(2*eps)
            err = float((approx-expected).abs().max())
            fd.append({'epsilon':eps,'approx':approx.tolist(),'expected':expected.tolist(),'max_absolute_error':err})
        assert min(v['max_absolute_error'] for v in fd) < max(2e-4,float(expected.abs().max())*.02), fd
        checks['direction_finite_difference'] = fd
        set_point(self.params,self.anchor)
        direction, info = random_direction(0,int(self.config.get('direction_seed',20260920)),self.device)
        assert abs(info['norm']-1) < 1e-6
        set_point(self.params,self.anchor,direction,2)
        realized = norm_sq([p-a for p,a in zip(self.params,self.anchor)])**.5
        assert abs(realized-2)<1e-5
        checks['random_direction'] = dict(info, realized_radius2=realized)
        set_point(self.params,self.anchor)
        checks['throughput_256'] = ref['fisher']
        checks['cuda_peak_memory_bytes'] = torch.cuda.max_memory_allocated(self.device) if self.device.type == 'cuda' else None
        checks['estimated_seconds_per_state'] = ref['seconds']
        self.finish({'passed':True,'measurement_checks_passed':True,'historical_precision_reference_passed':True,
                     'native_FP32_claimed_identical_to_historical':False,'checks':checks})

    def trajectory(self):
        self.require_sanity()
        origin=self.origin()
        anchor_state=read_state(self.paths[self.args.method][0])
        anchor_effective=effective_weights(anchor_state)
        rows=[]
        for task in range(10):
            key=f'task{task:02d}'
            path=checkpoint_path(self.args.method,task,self.config)
            checkpoint_sha=sha_file(path)
            if key in self.journal.rows:
                row=self.journal.rows[key]
                assert row['checkpoint_sha256']==checkpoint_sha
                rows.append(row)
                continue
            started=time.monotonic()
            state=read_state(path)
            assert state['fc.weight'].shape[0]==20*(task+1)
            assert torch.equal(state['fc.weight'][:20],anchor_state['fc.weight'])
            assert torch.equal(state['fc.bias'][:20],anchor_state['fc.bias'])
            for name,value in anchor_state.items():
                if not name.startswith('fc.') and 'linear_a_' not in name and 'linear_b_' not in name:
                    assert torch.equal(value,state[name]), f'Changed backbone at task{task}: {name}'
            drift=norm_sq([b-a for a,b in zip(anchor_effective,effective_weights(state))])**.5
            if task==0:
                metrics={split:{scope:dict(origin['metrics'][split],classes=20) for scope in ('local','seen')}
                         for split in ('train','test')}
            else:
                model=load_network(path,self.source,self.device)
                metrics={split:evaluate_local_and_seen(model,self.x[split],self.y[split],self.device,self.eval_batch)
                         for split in ('train','test')}
                del model
                if self.device.type=='cuda':
                    torch.cuda.empty_cache()
            row=self.journal.add({'state_id':key,'task':task,'method':self.args.method,
                 'checkpoint':str(path),'checkpoint_sha256':checkpoint_sha,
                 'metrics':metrics,'effective_weight_drift':drift,
                 'D0':{split:metrics[split]['local']['loss']-origin['metrics'][split]['loss'] for split in ('train','test')},
                 'head_sha256':self.meta['head_sha256'],'data_manifest_sha256':self.data['manifest_sha256'],
                 'production_precision':self.meta['production_precision'],
                 'source': 'reused current FP32 sanity anchor' if task==0 else 'direct full-checkpoint forward',
                 'seconds':time.monotonic()-started})
            rows.append(row)
            del state
        self.finish({'tasks':rows,'task_count':10,'precision':self.meta['production_precision'],
                     'input_transform':'deterministic test transforms on both train and test',
                     'seen_scope':'old Task0 examples classified among all 20*(task+1) seen classes',
                     'local_scope':'old Task0 examples with fixed original 20-class head'})

    def neighborhood(self):
        self.require_sanity()
        origin = self.origin()
        radii = self.config.get('radii',[0,.5,1,2,5,10,20])
        directions = int(self.config.get('directions',16))
        for index in range(directions):
            direction, meta = random_direction(index,int(self.config.get('direction_seed',20260920)),self.device)
            for radius in radii:
                if radius == 0:
                    continue
                self.measure(f'r{radius:g}_u{index:02d}',direction,float(radius),kind='neighborhood',
                             radius=float(radius),direction=meta)
        summary = []
        for radius in radii:
            rows = ([origin] if radius==0 else [r for r in self.journal.rows.values()
                    if r['point'].get('kind')=='neighborhood' and r['point']['radius']==radius])
            assert len(rows) == (1 if radius==0 else directions)
            item = {'radius':radius,'directions':len(rows),
                    'raw_trace_mean':sum(r['fisher']['raw_trace'] for r in rows)/len(rows)}
            for split in ('train','test'):
                damage = torch.tensor([r['metrics'][split]['loss']-origin['metrics'][split]['loss'] for r in rows],dtype=torch.float64)
                positive = damage.clamp_min(0)
                item[split] = {'D_mean':float(damage.mean()),'G':float(positive.mean()),
                               'Q95':float(torch.quantile(positive,.95)), 'D_each':damage.tolist()}
            summary.append(item)
        tolerance = []
        for split in ('train','test'):
            for metric in ('G','Q95'):
                for epsilon in self.config.get('damage_thresholds',[.001,.005,.01,.02,.05]):
                    radius = 0
                    for row in sorted(summary,key=lambda r:r['radius']):
                        if row[split][metric] > epsilon:
                            break
                        radius = row['radius']
                    tolerance.append({'split':split,'metric':metric,'epsilon':epsilon,'prefix_radius':radius,
                                      'right_censored':radius==max(radii)})
        self.finish({'radius_summary':summary,'tolerance':tolerance,'paired_directions':True})

    def bridge(self):
        self.require_sanity()
        origin = self.origin()
        radii = self.config.get('bridge_radii',[0,.25,.5,1,1.5,2])
        summary = []
        for method,delta in self.deltas.items():
            unit,drift = unit_direction(delta)
            for radius in radii:
                row = origin if radius == 0 else self.measure(f'{method}_r{radius:g}',unit,float(radius),
                       kind='bridge',direction_method=method,radius=float(radius),actual_drift=drift,
                       direction_sha256=self.meta['actual_direction_sha256'][method])
                summary.append({'direction_method':method,'radius':radius,'raw_trace':row['fisher']['raw_trace'],
                   'damage':{s:row['metrics'][s]['loss']-origin['metrics'][s]['loss'] for s in ('train','test')}})
        self.finish({'bridge_summary':summary})

    def path(self):
        self.require_sanity()
        origin = self.origin()
        method = self.args.method
        def get(s):
            return origin if s==0 else self.measure(f's{s:.6f}',self.deltas[method],s,kind='path',
                       path_fraction=s,direction_sha256=self.meta['actual_direction_sha256'][method])
        def simpson(n):
            rows = [get(i/(n-1)) for i in range(n)]
            qq = [r['fisher']['q'][method] for r in rows]
            return (qq[0]+qq[-1]+4*sum(qq[1:-1:2])+2*sum(qq[2:-1:2]))/(3*(n-1)),rows
        i9,rows = simpson(9)
        i5,_ = simpson(5)
        diff = abs(i9-i5)/max(abs(i9),1e-12)
        integrals = {'5':i5,'9':i9}
        selected,nodes,converged = i9,9,diff<=.05
        if not converged:
            i17,rows = simpson(17)
            integrals['17'] = i17
            diff = abs(i17-i9)/max(abs(i17),1e-12)
            selected,nodes,converged = i17,17,diff<=.05
        damage = rows[-1]['fisher']['subset_loss']-origin['fisher']['subset_loss']
        bound_passed = damage**2 <= selected + max(1e-8, .02*abs(selected))
        result = {'integrals_simpson':integrals,'selected_nodes':nodes,'integral':selected,
                  'S_path':selected/self.drift[method]**2 if self.drift[method] else None,
                  'drift':self.drift[method],'quadrature_relative_difference':diff,'quadrature_converged':converged,
                  'D_subset':damage,'D_subset_squared':damage**2,'bound_passed_with_2pct_numeric_tolerance':bound_passed,
                  'bound_gap_I_minus_D2':selected-damage**2,
                  'path': [{'s':i/(nodes-1),'q':r['fisher']['q'][method],'raw_trace':r['fisher']['raw_trace'],
                            'subset_loss':r['fisher']['subset_loss'],'metrics':r['metrics']} for i,r in enumerate(rows)]}
        self.finish(result)


def self_test():
    """Tiny nonlinear 12-layer CPU witness for the exact batch-hook estimator."""
    from torch import nn
    from types import SimpleNamespace
    from rq2_common import EffectiveQKV, PerExampleCollector
    torch.set_num_threads(1)
    torch.manual_seed(42)
    collector = PerExampleCollector()
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList()
            for i in range(12):
                old = SimpleNamespace(qkv=nn.Linear(5,15),linear_a_q=nn.Linear(5,3,bias=False),
                       linear_b_q=nn.Linear(3,5,bias=False),linear_a_v=nn.Linear(5,3,bias=False),
                       linear_b_v=nn.Linear(3,5,bias=False))
                self.blocks.append(EffectiveQKV(old,collector,i*2))
            self.head = nn.Linear(5,3)
        def forward(self,x):
            for block in self.blocks:
                q,k,v = block(x).chunk(3,-1)
                x = torch.tanh(x + .1*q + .1*k + .1*v)
            return self.head(x.mean(1))
    model = Tiny()
    params = [p for b in model.blocks for p in (b.effective_q,b.effective_v)]
    directions = {'a':[torch.randn_like(p) for p in params]}
    x,y = torch.randn(3,4,5),torch.tensor([0,1,2])
    batch = fisher(model,collector,params,x,y,directions,'cpu',3,True)
    errors = []
    for i in range(3):
        loss = F.cross_entropy(model(x[i:i+1]),y[i:i+1])
        gg = torch.autograd.grad(loss,params)
        tr = sum(float(g.square().sum(dtype=torch.float64)) for g in gg)
        dot = sum(float((g*d).sum(dtype=torch.float64)) for g,d in zip(gg,directions['a']))
        errors.extend([abs(tr-batch['sample_trace'][i]),abs(dot-batch['sample_dots']['a'][i])])
    assert max(errors)<1e-6, errors
    assert abs(batch['q']['a']-sum(d*d for d in batch['sample_dots']['a'])/3)<1e-12
    print(json.dumps({'self_test':'passed','max_absolute_error':max(errors),'per_example_hook_exact':True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--method',choices=('sgd','gam'))
    parser.add_argument('--job',choices=('sanity','neighborhood','bridge','path','trajectory'))
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--self-test',action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if not (args.config and args.method and args.job):
        parser.error('--config, --method and --job are required')
    runner = Runner(args)
    getattr(runner,args.job)()


if __name__ == '__main__':
    main()
