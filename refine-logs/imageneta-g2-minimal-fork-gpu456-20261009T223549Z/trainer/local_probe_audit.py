#!/usr/bin/env python3
"""Same-state, TRAIN-only local response audit. No parameter update is committed.

All points use native FP32 forward/backward and the exact same entry RNG.
Counterfactual natural/matched steps are observational and always restored;
their loss is not a trained algorithm's accuracy or a model-selection score.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import torch
import torch.nn.functional as F
from state_transaction import State, rng_get, rng_set, rng_equal
from effective_step import EffectiveStep

EPS = 1e-12


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for x in iter(lambda: f.read(1 << 20), b''): h.update(x)
    return h.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def norm(v):
    ans = float(v.detach().double().square().sum().sqrt())
    if not math.isfinite(ans): raise FloatingPointError('Nonfinite vector norm')
    return ans


def cosine(a, b):
    den = norm(a) * norm(b)
    return float((a.double() * b.double()).sum()) / den if den > EPS else None


def relative_error(a, b):
    den = norm(b)
    return norm(a - b) / den if den > EPS else None


def normalized(v):
    magnitude = norm(v)
    return v / magnitude if magnitude > EPS else torch.zeros_like(v)


def diagnostic_decomposition(change, base):
    """Read-only scalar decomposition; never changes a submitted direction."""
    denominator = norm(base)**2
    coefficient = float((change.double()*base.double()).sum())/denominator if denominator>EPS else 0.
    perpendicular = change-coefficient*base
    magnitude = norm(change)
    return {'parallel_to_g0_coefficient':coefficient,
            'parallel_to_g0_norm':abs(coefficient)*norm(base),
            'perpendicular_to_g0_norm':norm(perpendicular),
            'perpendicular_over_change_norm':norm(perpendicular)/magnitude if magnitude>EPS else None,
            'change_over_g0_norm':magnitude/norm(base) if norm(base)>EPS else None,
            'diagnostic_only_no_projection_in_training':True}


class LocalProbeEngine:
    """One fixed model and realized minibatch; each read is an entry transaction.

    ``closure`` returns (scalar CE, full seen-class logits). Trainable parameter
    coordinates include all LoRA factors and the complete classifier. No
    Hessian-vector products, optimizer, history, projection or norm rescaling
    occurs here; normmatching is only an explicit offline virtual control.
    """
    def __init__(self, model, closure, geometry=None):
        self.model, self.closure, self.geometry = model, closure, geometry
        self.named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.params = [p for _, p in self.named]
        if not self.params: raise ValueError('Expected trainable parameters')
        self.slices, offset = [], 0
        for p in self.params:
            self.slices.append((offset, offset + p.numel())); offset += p.numel()
        self.dimension = offset
        self.entry = State(model)
        self.base = [p.detach().clone() for p in self.params]
        self.parameter_ids = {id(p): i for i, p in enumerate(self.params)}
        self.effective_base = geometry.capture() if geometry is not None else None
        self.gradient_reads = self.loss_reads = self.virtual_parameter_writes = 0
        self.restore_count = 0
        self.committed_updates = 0

    @torch.no_grad()
    def _place(self, direction=None, scale=0.):
        self.entry.restore()
        if direction is not None:
            if direction.numel() != self.dimension or not bool(torch.isfinite(direction).all()):
                raise ValueError('Invalid displacement direction')
            for p, (a, b) in zip(self.params, self.slices):
                p.add_(direction[a:b].reshape_as(p), alpha=float(scale))
            self.virtual_parameter_writes += 1

    @torch.no_grad()
    def _distance(self):
        distance = math.sqrt(sum(float((p.double() - v.double()).square().sum())
                                 for p, v in zip(self.params, self.base)))
        result = {'parameter_norm': distance, 'fp32_distinguishable': distance > 0.}
        if self.geometry is not None:
            result['effective'] = self.geometry.actual_delta(self.effective_base)
        return result

    def _restore(self):
        self.entry.restore()
        checks = self.entry.check()
        self.restore_count += 1
        return checks

    def gradient(self, batch, direction=None, scale=0.):
        """Return cloned gradient/logits and actual realized probe geometry."""
        self._place(direction, scale)
        try:
            geometry = self._distance()
            self.model.zero_grad(set_to_none=True)
            loss, logits = self.closure(*batch)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Nonfinite native probe loss/logits')
            loss.backward()
            if any(p.grad is None or not bool(torch.isfinite(p.grad).all()) for p in self.params):
                raise FloatingPointError('Missing/nonfinite gradient')
            values = torch.cat([p.grad.detach().reshape(-1) for p in self.params]).clone()
            self.gradient_reads += 1
            return {'gradient': values, 'loss': float(loss.detach()),
                    'logits': logits.detach().clone(), 'distance': geometry,
                    'gradient_sha256': tensor_sha(values)}
        finally:
            self._restore()

    @torch.no_grad()
    def loss(self, batch, direction=None, scale=0.):
        self._place(direction, scale)
        try:
            geometry = self._distance()
            loss, logits = self.closure(*batch)
            if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(logits).all()):
                raise FloatingPointError('Nonfinite virtual loss/logits')
            self.loss_reads += 1
            return float(loss), geometry
        finally:
            self._restore()

    @torch.no_grad()
    def virtual_step_distance(self, gradient, step):
        self._place(gradient, -float(step))
        try: return self._distance()
        finally: self._restore()

    def match_parameter_length(self, gradient, eta, target, rtol=1e-4):
        """Match ACTUAL FP32 parameter distance of the baseline virtual step.

        This diagnostic coefficient is never passed to formal training. A
        reported unmatched zero/nonrepresentable case is not hidden as success.
        """
        n = norm(gradient)
        if n <= EPS or target <= EPS:
            return {'matched': False, 'coefficient': 0., 'reason': 'zero_direction_or_target'}
        coefficient = target / (float(eta) * n)
        measured = self.virtual_step_distance(gradient, float(eta) * coefficient)['parameter_norm']
        if abs(measured - target) / target <= rtol:
            return {'matched': True, 'coefficient': coefficient, 'actual_parameter_norm': measured,
                    'target_parameter_norm': target, 'relative_error': abs(measured-target)/target,
                    'tolerance': rtol, 'iterations': 0}
        lo, hi = 0., coefficient * 2.
        for iteration in range(40):
            mid = (lo + hi) / 2.
            measured = self.virtual_step_distance(gradient, float(eta) * mid)['parameter_norm']
            coefficient = mid
            if abs(measured-target)/target <= rtol: break
            if measured < target: lo = mid
            else: hi = mid
        return {'matched': abs(measured-target)/target <= rtol,
                'coefficient': coefficient, 'actual_parameter_norm': measured,
                'target_parameter_norm': target, 'relative_error': abs(measured-target)/target,
                'tolerance': rtol, 'iterations': iteration+1}

    def gradient_shares(self, gradient):
        fc = sum(float(gradient[a:b].double().square().sum())
                 for (name, _), (a, b) in zip(self.named, self.slices)
                 if name.startswith('fc.'))
        total = norm(gradient) ** 2
        return {'total_norm': math.sqrt(total), 'fc_norm': math.sqrt(fc),
                'lora_norm': math.sqrt(max(0., total-fc)),
                'fc_squared_norm_share': fc/total if total>EPS else None}


def logits_kl(base, probe, class_slice):
    """KL(base||probe), explicit current/seen logit coordinates."""
    def kl(a, b):
        la, lb = a.double().log_softmax(-1), b.double().log_softmax(-1)
        return float((la.exp() * (la-lb)).sum(-1).mean())
    start, end = class_slice
    return {'current_task_logits_kl_base_to_probe': kl(base[:, start:end], probe[:, start:end]),
            'all_seen_logits_kl_base_to_probe': kl(base, probe),
            'current_class_slice': [start, end], 'seen_class_count': base.shape[-1]}


def audit_fixed_batch(engine, batch, heldout_batch, nominal_lr, class_slice,
                      random_seed=20261010, smoke=False):
    """Bounded, prespecified a/r points and five diagnostic direction controls."""
    base = engine.gradient(batch)
    g0, u = base['gradient'], normalized(base['gradient'])
    h = float(nominal_lr) * norm(g0)
    points, inner, outer = [], {}, {}
    radii = [('a025', .025), ('a050', .05)] if smoke else [
        ('a025', .025), ('a050', .05), ('a100', .1), ('h', h)]
    for name, a in radii:
        # Preserve native FP32 arithmetic for the coupled SGD-distance point:
        # theta.add_(g0, alpha=-eta), rather than two rounded normalizations.
        read = (engine.gradient(batch, g0, -float(nominal_lr)) if name=='h'
                else engine.gradient(batch, u, -a))
        response = g0-read['gradient']
        q = response / a if a>EPS else torch.zeros_like(response)
        inner[name] = {'read': read, 'a': a, 'response': response,
                       'q': q, 'v': normalized(response)}
        row = {'category': 'inner_response', 'point': name, 'inner_radius': a,
               'h_reference': h, 'nominal_lr': nominal_lr,
               'inner_placement':'native_sgd_add_g0_minus_eta' if name=='h' else 'fixed_unit_g0_descent',
               'actual_parameter_distance': read['distance']['parameter_norm'],
               'fp32_distinguishable': read['distance']['fp32_distinguishable'],
               'effective_distance': read['distance'].get('effective'),
               'loss': read['loss'], 'base_loss': base['loss'],
               'gradient_norm': norm(read['gradient']), 'response_norm': norm(response),
               'q_norm': norm(q), 'response_direction_valid': norm(response)>EPS,
               'gradient_sha256': read['gradient_sha256']}
        row.update(logits_kl(base['logits'], read['logits'], class_slice))
        points.append(row)
    qref = inner['a025']['q']
    for row in points:
        q = inner[row['point']]['q']
        row.update(q_cosine_to_a025=cosine(q,qref), q_relative_error_to_a025=relative_error(q,qref))
        # Only two explicit half pairs are registered; h/2 and a025/2 are
        # deliberately unavailable, rather than claiming an unmeasured check.
        half_name = {'a050':'a025','a100':'a050'}.get(row['point'])
        row['inner_half_pair_available'] = half_name in inner
        if half_name in inner:
            half_q=inner[half_name]['q']
            row.update(inner_half_point=half_name,
                       q_cosine_to_half_a=cosine(q,half_q),
                       q_relative_error_to_half_a=relative_error(q,half_q))
    representative = ['a050'] if smoke else ['a050', 'h']
    rs = [.125,.25] if smoke else [.025,.05,.125,.25,.5,1.]
    for name in representative:
        v = inner[name]['v']
        for r in rs:
            read = engine.gradient(batch, v, r)
            outer[name,r] = read
            change = read['gradient']-g0
            row = {'category':'outer_response', 'point': f'{name}_r{r:g}',
                   'inner_point':name,'inner_radius':inner[name]['a'], 'outer_radius':r,
                   'h_reference':h, 'nominal_lr':nominal_lr, 'loss':read['loss'],
                   'base_loss':base['loss'], 'gradient_norm':norm(read['gradient']),
                   'g2_minus_g0_norm':norm(change), 'cosine_g2_g0':cosine(read['gradient'],g0),
                   'actual_parameter_distance':read['distance']['parameter_norm'],
                   'effective_distance':read['distance'].get('effective'),
                   'fp32_distinguishable':read['distance']['fp32_distinguishable'],
                   'response_direction_valid':norm(inner[name]['response'])>EPS,
                   'frozen_response_direction_sha256':tensor_sha(v),
                   'gradient_sha256':read['gradient_sha256']}
            row.update(logits_kl(base['logits'],read['logits'],class_slice))
            row.update(diagnostic_decomposition(change,g0))
            points.append(row)
    for row in points:
        if row['category']!='outer_response': continue
        name,r = row['inner_point'],row['outer_radius']
        if (name,r/2) in outer:
            full = outer[name,r]['gradient']-g0
            doubled_half = 2*(outer[name,r/2]['gradient']-g0)
            row.update(taylor_doubling_cosine=cosine(full,doubled_half),
                       taylor_doubling_relative_error=relative_error(full,doubled_half))
    directions = {'sgd':base, 'response_g2':outer['a050',.25]}
    local_generator = torch.Generator(device='cpu').manual_seed(random_seed)
    random_v = normalized(torch.randn(g0.shape, generator=local_generator,
                                     dtype=g0.dtype).to(g0.device))
    for name,direction in [('same_radius_sam',u), ('opposite_response',-inner['a050']['v']),
                           ('same_radius_random',random_v)]:
        read = engine.gradient(batch,direction,.25)
        directions[name] = read
        row = {'category':'direction_control', 'point':name, 'inner_radius':.05,
               'outer_radius':.25, 'loss':read['loss'], 'base_loss':base['loss'],
               'gradient_norm':norm(read['gradient']),
               'g_candidate_minus_g0_norm':norm(read['gradient']-g0),
               'cosine_candidate_g0':cosine(read['gradient'],g0),
               'actual_parameter_distance':read['distance']['parameter_norm'],
               'effective_distance':read['distance'].get('effective'),
               'fp32_distinguishable':read['distance']['fp32_distinguishable'],
               'gradient_sha256':read['gradient_sha256']}
        row.update(logits_kl(base['logits'],read['logits'],class_slice)); points.append(row)
    heldout_base,_ = engine.loss(heldout_batch)
    counterfactuals = []
    for eta in ([.02] if smoke else [.02,.002]):
        target = engine.virtual_step_distance(g0,eta)['parameter_norm']
        for name,read in directions.items():
            direction = read['gradient']
            match = engine.match_parameter_length(direction,eta,target)
            for mode,coefficient in [('natural',1.),('matched_actual_parameter_length',match['coefficient'])]:
                ce,geo = engine.loss(batch,direction,-eta*coefficient)
                heldout_ce,_ = engine.loss(heldout_batch,direction,-eta*coefficient)
                counterfactuals.append({'category':'virtual_one_step','point':name,
                    'eta':eta,'mode':mode,'coefficient':coefficient,
                    'train_probe_ce_before':base['loss'],'train_probe_ce_after':ce,
                    'train_probe_ce_change':ce-base['loss'],
                    'heldout_train_ce_before':heldout_base,'heldout_train_ce_after':heldout_ce,
                    'heldout_train_ce_change':heldout_ce-heldout_base,
                    'actual_parameter_step':geo['parameter_norm'],
                    'effective_step':geo.get('effective'),
                    'target_g0_actual_parameter_step':target,
                    'actual_parameter_match_relative_error':abs(geo['parameter_norm']-target)/target if target>EPS else None,
                    'match':match if mode!='natural' else None,
                    'not_committed':True,'not_trained_accuracy':True})
    checks = engine.entry.check()
    return {'points':points,'counterfactuals':counterfactuals,
            'g0_sha256':base['gradient_sha256'],'g0_layer_norm_shares':engine.gradient_shares(g0),
            'nominal_lr':nominal_lr,'h_reference':h,'base_loss':base['loss'],
            'gradient_reads':engine.gradient_reads,'loss_reads':engine.loss_reads,
            'committed_updates':engine.committed_updates,'restore_count':engine.restore_count,
            'checks':checks,'normalization_scope':'all_trainable_LoRA_and_complete_seen_FC',
            'probe_dtype':'native_float32','summary_reductions':'float64',
            'random_direction_generator':'independent_local_CPU_torch_Generator',
            'all_probes_share_same_entry_R0':True,'no_R1_advance':True,
            'heldout_scope':'disjoint_current_task_TRAIN_minibatch_not_dev_or_test'}


def deterministic_batches(dataset, raw_ids, batch_size=128, probe_batches=4, seed=20261010):
    """Fixed disjoint indices; materialization and augmentations restore RNG."""
    if len(dataset) != len(raw_ids): raise ValueError('Dataset/raw ID length mismatch')
    need = (probe_batches+1)*batch_size
    if len(dataset)<need: raise ValueError(f'{need} disjoint train examples required; got {len(dataset)}')
    generator = torch.Generator().manual_seed(seed)
    chosen = torch.randperm(len(dataset),generator=generator)[:need].tolist()
    original_rng = rng_get()
    batches, receipts = [], []
    try:
        from torch.utils.data._utils.collate import default_collate
        for j in range(probe_batches+1):
            local_ids = chosen[j*batch_size:(j+1)*batch_size]
            realized = default_collate([dataset[i] for i in local_ids])
            if len(realized)!=3: raise ValueError('Native dataset must return index,input,target')
            _,inputs,targets = realized
            batches.append((inputs,targets))
            receipts.append({'relative_indices':local_ids,'raw_train_indices':[int(raw_ids[i]) for i in local_ids],
                             'input_sha256':tensor_sha(inputs),'target_sha256':tensor_sha(targets),
                             'role':'shared_heldout_train' if j==probe_batches else 'train_probe',
                             'transform_mode':'test_deterministic_on_train_source'})
    finally: rng_set(original_rng)
    flattened = [i for receipt in receipts for i in receipt['raw_train_indices']]
    if len(flattened)!=len(set(flattened)): raise AssertionError('Audit minibatches overlap')
    return batches[:-1],batches[-1],receipts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source','reference-output-dir','output-dir','dev-split'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    cli=parser.parse_args()
    source=Path(cli.source).resolve();reference=Path(cli.reference_output_dir).resolve()
    out=Path(cli.output_dir).resolve();split=Path(cli.dev_split).resolve()
    epochs=[0] if cli.smoke else [0,1,10,25,49]
    if cli.dry_run:
        print(json.dumps({'status':'valid','source':str(source),'reference_output_dir':str(reference),
            'epochs':epochs,'probe_batches_per_state':1 if cli.smoke else 4,
            'committed_updates':0,'data_scope':'current_task_train_excludes_dev_and_test',
            'normal_gradient_reads_per_batch':8 if cli.smoke else 20,
            'eta_cross':[.02] if cli.smoke else [.02,.002],
            'independent_inner_radii':[.025,.05] if cli.smoke else [.025,.05,.1,'eta*norm(g0)'],
            'outer_radii':[.125,.25] if cli.smoke else [.025,.05,.125,.25,.5,1.]}));return
    if out.exists() and any(out.iterdir()): raise FileExistsError('Audit output must be fresh')
    out.mkdir(parents=True,exist_ok=True)
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8': raise RuntimeError('Deterministic workspace required')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    sys.path.insert(0,str(source));os.chdir(source)
    from probe_study_train import load_snapshot
    from probe_study_data import build_data_manager,current_task_train_dataset
    config=json.loads((reference/'effective_config.json').read_text())
    reference_summary=json.loads((reference/'final_summary.json').read_text())
    if reference_summary.get('status')!='completed' or reference_summary.get('study_condition_id')!='B0':
        raise AssertionError('Local diagnostics require the completed shared-parent SGD Task1 B0 reference')
    if bool(reference_summary.get('smoke'))!=cli.smoke:
        raise AssertionError('Smoke and formal snapshot/reference budgets must not be mixed')
    if config.get('study_condition_id')!='B0' or config.get('response_method')!='sgd':
        raise AssertionError('Frozen snapshots must belong to the pre-registered SGD reference')
    dm=build_data_manager(str(source),config,str(split))
    dataset,raw_ids=current_task_train_dataset(dm,1,config)
    if not set(map(int,raw_ids)).issubset(set(dm.study_train_indices)) or set(map(int,raw_ids))&set(dm.study_dev_indices):
        raise AssertionError('Audit examples must come only from the training subset, excluding development')
    pb,heldout,receipts=deterministic_batches(dataset,raw_ids,probe_batches=1 if cli.smoke else 4)
    device=torch.device('cuda:0');pb=[tuple(v.to(device) for v in b) for b in pb]
    heldout=tuple(v.to(device) for v in heldout)
    if any(not bool(((targets>=20)&(targets<40)).all()) for _,targets in pb+[heldout]):
        raise AssertionError('Offline gradients/loss must use only the current Task1 labels')
    result={'schema':'LocalProbeAudit_v1','status':'running','smoke':cli.smoke,
        'committed_updates':0,'model_selection_eligible':False,'trained_method_result':False,
        'source':str(source),'reference_output_dir':str(reference),'dev_split_sha256':file_sha(split),
        'reference_final_summary_sha256':file_sha(reference/'final_summary.json'),
        'audit_sha256':file_sha(__file__),'data_receipts':receipts,'snapshots':[],
        'pointset_preregistered':True,'snapshots_from_formal_sgd_task1_only':True,
        'no_test_or_dev_loss_or_gradient':True,'no_history_fisher_projection_hvp':True}
    atomic_json(out/'final_summary.json',result);started=time.time()
    flat=[]
    try:
        for epoch in epochs:
            snapshot=reference/'snapshots'/f'epoch_{epoch:03d}.pt'
            model,payload=load_snapshot(snapshot,device=str(device),restore_rng=True)
            if payload['task']!=1 or payload['class_slice']!=[20,40]: raise AssertionError('Wrong snapshot task')
            if payload['config'].get('study_condition_id')!='B0' or payload['config'].get('response_method')!='sgd':
                raise AssertionError('Wrong diagnostic trajectory: snapshots must come from SGD B0')
            if payload['dev_split_sha256']!=file_sha(split): raise AssertionError('Snapshot/dev split mismatch')
            initial_rng=rng_get()
            class_slice=[20,40]
            def closure(inputs,targets):
                output=model(inputs)
                logits=output['logits'] if isinstance(output,dict) else output
                return F.cross_entropy(logits[:,20:40],targets-20),logits
            geometry=EffectiveStep.from_model(model)
            lr=payload['current_optimizer_lr']
            snap={'epoch':epoch,'path':str(snapshot),'sha256':file_sha(snapshot),
                  'current_optimizer_lr':lr,'full_model_parameter_sha256':payload['full_model_parameter_sha256'],
                  'frozen_origin_sha256':payload['frozen_origin_sha256'],'batches':[]}
            for index,batch in enumerate(pb):
                rng_set(initial_rng)
                engine=LocalProbeEngine(model,closure,geometry)
                rows=audit_fixed_batch(engine,batch,heldout,lr,class_slice,
                                       random_seed=20261010+index,smoke=cli.smoke)
                rows['probe_batch_index']=index;snap['batches'].append(rows)
                for row in rows['points']+rows['counterfactuals']:
                    flat.append({'epoch':epoch,'probe_batch_index':index,**row})
            if not rng_equal(initial_rng,rng_get()): raise AssertionError('Offline reads consumed shared RNG')
            result['snapshots'].append(snap)
            atomic_json(out/'final_summary.json',result)
            del model
            torch.cuda.empty_cache()
        counts=[b for s in result['snapshots'] for b in s['batches']]
        result.update(status='passed',elapsed_seconds=time.time()-started,
            gradient_reads=sum(b['gradient_reads'] for b in counts),
            loss_reads=sum(b['loss_reads'] for b in counts),
            checked_snapshots=len(result['snapshots']),checked_batches=len(counts),
            checks={'zero_committed_updates':all(b['committed_updates']==0 for b in counts),
                    'all_entry_state_and_rng_restores':all(all(b['checks'].values()) for b in counts),
                    'train_minibatches_disjoint':True,'same_point_R0':True,
                    'no_dev_test_access':True,'not_full_training_performance':True})
        fields=sorted({k for row in flat for k in row})
        with open(out/'local_probe_points.csv','w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
            for row in flat: writer.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in row.items()})
        atomic_json(out/'local_probe_points.json',flat)
        (out/'LOCAL_PROBE_AUDIT.md').write_text(
            '# 同状态局部梯度诊断\n\n'
            f"验收：{result['status']}；{len(counts)} 个固定训练小批次，{result['gradient_reads']} 次普通梯度读取，真实训练更新 **0**。\n\n"
            '内半径比较梯度差/a 的方向与大小；外半径保持响应方向固定，比较半点线性一致性。'
            '同半径响应、SAM、反方向与随机方向均为离线控制。两个学习率在同一快照交叉，'
            '同时报告自然步长和实际 FP32 参数步长匹配的虚拟一步结果。\n\n'
            '所有探针与虚拟更新后恢复参数、梯度、缓冲、模式、缓存和 RNG；无 HVP、Fisher、历史、投影。'
            '损失仅来自当前任务训练集的探针批和独立训练保留批，既不读取 dev/test，也不用于模型选择。'
            '这些数值不是完成训练的 FAA/AAA，不能替代正式算法比较。\n')
    except BaseException as exc:
        result.update(status='failed',error=repr(exc));raise
    finally: atomic_json(out/'final_summary.json',result)


if __name__=='__main__': main()
