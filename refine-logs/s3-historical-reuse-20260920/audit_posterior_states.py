"""CPU-only audit of 20 historical endpoint/retained posterior states."""
from pathlib import Path
import csv
import datetime
import gc
import hashlib
import json
import math
import os
import re
import resource
import statistics
import subprocess
import torch

assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'Explicit CPU-only environment required'
torch.set_num_threads(2)
torch.serialization.add_safe_globals([torch.nn.Linear, set])
ROOT = Path('/data/140-1/users/liying/CopyFlatCV')
OUT = Path(__file__).parent
index = json.loads((OUT / 'reverse_index.json').read_text())
selected = [r for r in index['runs'] if Path(r['log_path']).name in {'mechanism_ours_s2_v2_smigpu4.log','mechanism_baseline_s2_v2_smigpu5.log'}]
assert len(selected) == 2

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def pair_metrics(a, b):
    assert a.keys() == b.keys() and a
    aa = bb = dot = ds = 0.0
    identical = True
    for k, av in a.items():
        bv = b[k]
        assert av.shape == bv.shape
        assert torch.isfinite(av).all() and torch.isfinite(bv).all()
        identical = identical and torch.equal(av, bv)
        x, y = av.double(), bv.double()
        aa += x.square().sum().item()
        bb += y.square().sum().item()
        dot += (x * y).sum().item()
        ds += (x - y).square().sum().item()
    return {'exact_equal': identical, 'l2_endpoint': math.sqrt(aa), 'l2_merge': math.sqrt(bb), 'shift_l2': math.sqrt(ds), 'relative_l2': math.sqrt(ds / aa), 'cosine': dot / math.sqrt(aa * bb)}

def fisher_profile(values):
    # Semantic float64 trace; log profile was computed by concatenated float32 reductions.
    assert values and all(torch.isfinite(v).all() and (v >= 0).all() for v in values.values())
    trace = sum(v.double().sum().item() for v in values.values())
    numel = sum(v.numel() for v in values.values())
    maximum = max(v.max().item() for v in values.values())
    return {'blocks': len(values), 'numel': numel, 'trace_float64': trace, 'mean_float64': trace / numel, 'max': maximum}

def delta_metrics(a, b):
    aa = bb = ds = 0.0
    count = 0
    for k, av in a.items():
        if '.linear_a_' not in k: continue
        bk = k.replace('.linear_a_', '.linear_b_')
        da = a[bk].double() @ av.double()
        db = b[bk].double() @ b[k].double()
        aa += da.square().sum().item()
        bb += db.square().sum().item()
        ds += (db-da).square().sum().item()
        count += 1
    assert count == 24
    return {'blocks': count, 'endpoint_l2': math.sqrt(aa), 'merge_l2': math.sqrt(bb), 'shift_l2': math.sqrt(ds), 'relative_l2': math.sqrt(ds / aa)}

def load_checkpoint_factors(folder, task):
    a_path = folder / f'lora_w_a_{task}.pt'
    b_path = folder / f'lora_w_b_{task}.pt'
    a = torch.load(a_path, map_location='cpu', weights_only=True)
    b = torch.load(b_path, map_location='cpu', weights_only=True)
    assert len(a) == len(b) == 24
    factors = {}
    for block in range(12):
        for offset, part in enumerate(['q','v']):
            idx = block * 2 + offset
            name = f'backbone.lora_vit.blocks.{block}.attn.qkv'
            factors[f'{name}.linear_a_{part}.weight'] = a[idx].weight.detach() if hasattr(a[idx], 'weight') else a[idx]
            factors[f'{name}.linear_b_{part}.weight'] = b[idx].weight.detach() if hasattr(b[idx], 'weight') else b[idx]
    return factors, [a_path, b_path]

rows = []
provenance = []
for run in selected:
    group = 'ours' if 'mechanism_ours_' in run['log_path'] else 'baseline_nodamp_nomerge'
    log_path = Path(run['log_path'])
    metrics_path = Path(run['metrics_path'])
    assert digest(log_path) == run['log_sha256']
    assert digest(metrics_path) == run['metrics_sha256']
    raw = json.loads(metrics_path.read_text())
    log = log_path.read_text(errors='replace')
    posterior_dir = ROOT / 'outputs_logs/step_ablation/posterior_state' / run['config']['prefix']
    checkpoint_dir = ROOT / run['config']['filepath']
    assert len(list(posterior_dir.glob('posterior_state_task*.pt'))) == 10
    provenance.append({k: run[k] for k in ['id','log_path','log_sha256','metrics_path','metrics_sha256','config_path','config','faa_macro','aaa_macro','sample_final']})
    for task in range(10):
        archive = posterior_dir / f'posterior_state_task{task:02d}.pt'
        state = torch.load(archive, map_location='cpu', weights_only=True)
        assert state['task'] == task
        assert set(state) == {'task','theta_end_lora','mu_merge_lora','fisher_endpoint','fisher_merge'}
        theta, mu = state['theta_end_lora'], state['mu_merge_lora']
        fe, fm = state['fisher_endpoint'], state['fisher_merge']
        assert len(theta) == len(mu) == 48
        factor = pair_metrics(theta,mu)
        fp = pair_metrics(fe,fm)
        delta = delta_metrics(theta,mu)
        ep,mp = fisher_profile(fe),fisher_profile(fm)
        step = raw['mechanism']['steps'][str(task)]
        profile_checks = {}
        for name, profile in [('endpoint',ep),('merge',mp)]:
            for key in ['blocks','numel','max']:
                profile_checks[f'{name}_{key}_matches'] = profile[key] == step[f'fisher_{name}_{key}']
            # Float32 log-reduction roundoff is up to ~2e-7 of unit trace.
            profile_checks[f'{name}_trace_matches'] = math.isclose(profile['trace_float64'],step[f'fisher_{name}_total_mass'],rel_tol=5e-7,abs_tol=1e-9)
            profile_checks[f'{name}_mean_matches'] = math.isclose(profile['mean_float64'],step[f'fisher_{name}_mean'],rel_tol=5e-7,abs_tol=1e-14)
        factor_shift_match = math.isclose(factor['shift_l2'],step.get('weight_merge_theta_end_to_merge_l2',0.0),rel_tol=1e-5,abs_tol=1e-8)
        saved_match = re.search(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d+) .*\[PosteriorState\] saved task='+str(task)+r' -> '+re.escape(str(archive))+r'$',log,re.M)
        assert saved_match, (group,task,'missing archive save log')
        logged_save = datetime.datetime.strptime(saved_match.group(1),'%Y-%m-%d %H:%M:%S,%f')
        file_mtime = datetime.datetime.fromtimestamp(archive.stat().st_mtime)
        mtime_delta = (file_mtime-logged_save).total_seconds()
        checkpoint, checkpoint_paths = load_checkpoint_factors(checkpoint_dir,task)
        checkpoint_exact = checkpoint.keys() == mu.keys() and all(torch.equal(v,checkpoint[k]) for k,v in mu.items())
        row = {
            'group': group, 'task': task, 'seed': 2, 'archive_path': str(archive), 'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size,
            'archive_mtime_local': file_mtime.isoformat(), 'v2_log_save_time_naive': logged_save.isoformat(), 'mtime_minus_log_save_seconds': mtime_delta,
            'fisher_cosine': fp['cosine'], 'fisher_relative_l2_to_endpoint': fp['relative_l2'], 'fisher_shift_l2': fp['shift_l2'], 'fisher_exact_equal': fp['exact_equal'],
            'theta_endpoint_equals_mu_merge': factor['exact_equal'], 'factor_shift_l2': factor['shift_l2'], 'factor_relative_shift': factor['relative_l2'],
            'factor_endpoint_l2': factor['l2_endpoint'], 'factor_merge_l2': factor['l2_merge'],
            'delta_w_shift_l2': delta['shift_l2'], 'delta_w_relative_shift': delta['relative_l2'], 'delta_w_endpoint_l2': delta['endpoint_l2'], 'delta_w_merge_l2': delta['merge_l2'],
            'fisher_endpoint_trace': ep['trace_float64'], 'fisher_merge_trace': mp['trace_float64'], 'fisher_endpoint_max': ep['max'], 'fisher_merge_max': mp['max'],
            'profile_matches_v2_metrics': all(profile_checks.values()), 'profile_individual_checks': profile_checks,
            'factor_shift_matches_v2_metrics': factor_shift_match, 'mu_merge_equals_final_checkpoint_factors': checkpoint_exact,
            'archive_mtime_matches_v2_save': abs(mtime_delta)<5,
            'logged_fisher_mode': step['fisher_mode'], 'logged_weight_merge_applied': step['weight_merge_applied'],
            'anchor_endpoint_loss': step.get('anchor_endpoint_loss'), 'anchor_merge_loss': step.get('anchor_merge_loss'),
            'anchor_loss_merge_minus_endpoint': step['anchor_merge_loss']-step['anchor_endpoint_loss'],
            'anchor_acc_merge_minus_endpoint_pp': step['anchor_merge_acc']-step['anchor_endpoint_acc'],
            'checkpoint_paths_and_hashes': [{'path':str(p),'sha256':digest(p)} for p in checkpoint_paths],
            'log_path': str(log_path), 'metrics_path': str(metrics_path),
        }
        for k,v in row.items():
            assert not isinstance(v,float) or math.isfinite(v),(k,v)
        assert row['profile_matches_v2_metrics'] and factor_shift_match and checkpoint_exact and row['archive_mtime_matches_v2_save'],row
        rows.append(row)
        print(json.dumps({k:row[k] for k in ['group','task','theta_endpoint_equals_mu_merge','fisher_cosine','fisher_relative_l2_to_endpoint','delta_w_relative_shift','profile_matches_v2_metrics','mu_merge_equals_final_checkpoint_factors']},ensure_ascii=False),flush=True)
        del state,theta,mu,fe,fm,checkpoint
        gc.collect()

summary = {}
for group in ['ours','baseline_nodamp_nomerge']:
    rr = [r for r in rows if r['group']==group]
    ss = {'n_tasks':len(rr),'n_theta_equals_mu':sum(r['theta_endpoint_equals_mu_merge'] for r in rr),'n_fisher_equals':sum(r['fisher_exact_equal'] for r in rr)}
    for field in ['fisher_cosine','fisher_relative_l2_to_endpoint','factor_relative_shift','delta_w_relative_shift','anchor_loss_merge_minus_endpoint','anchor_acc_merge_minus_endpoint_pp']:
        vals = [r[field] for r in rr]
        ss[field] = {'mean':statistics.mean(vals),'min':min(vals),'max':max(vals),'std_across_tasks':statistics.stdev(vals)}
    summary[group] = ss

historical_source='models_LoRAbasedCL/as1_normfisher_gam_intask_damp_weightmergethenfisher.py'
blob=subprocess.check_output(['git','show','b194f04:'+historical_source],cwd=ROOT)
report = {'created_at':datetime.datetime.now().astimezone().isoformat(),'cpu_only':True,'torch_version':torch.__version__,
          'max_rss_mb':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
          'audit_definition':{'fisher_relative_l2':'||F_merge-F_endpoint||_2 / ||F_endpoint||_2','factor_relative_shift':'||mu_merge-theta_endpoint||_2 / ||theta_endpoint||_2','delta_w_relative_shift':'||concat(B_merge A_merge - B_endpoint A_endpoint)||_2 / ||concat(B_endpoint A_endpoint)||_2; 24 Q/V matrices','statistics_scope':'10 stages of one seed per arm, not ten independent seeds','profile_tolerance':'max/blocks/numel exact; trace and mean rtol5e-7 for float32 log reduction; factors exact against saved checkpoints','topk_overlap':'not computed'},
          'source_evidence':{'repository':str(ROOT),'git_revision':'b194f04','path':historical_source,'sha256':hashlib.sha256(blob).hexdigest(),'interpretation':'_capture_endpoint_posterior_state invokes an additional Fisher pass and restores model mode/gradients, not RNG; _compute_delta_fisher later recomputes at retained factors. This snapshot corroborates runtime logs; exact historical working-tree hash remains unavailable.'},
          'identity_checks':{'archives':20,'all_profile_matches':all(r['profile_matches_v2_metrics'] for r in rows),'all_saved_factor_matches':all(r['mu_merge_equals_final_checkpoint_factors'] for r in rows),'all_factor_shift_log_matches':all(r['factor_shift_matches_v2_metrics'] for r in rows),'all_mtime_save_matches':all(r['archive_mtime_matches_v2_save'] for r in rows)},
          'limits':['Baseline has both damping and fusion removed; it is not an isolated fusion-effect accuracy control.','Endpoint and retained Fisher use separate stochastic passes without matched sampler/transforms/RNG, so their difference includes estimator repeat noise.','No per-snapshot path factors are saved in these archives: cannot reconstruct exact intra-path B@A cross terms.','Historical diagnostics add Fisher/anchor/future passes and can alter RNG trajectories; these accuracy results cannot replace current native S3 exact-protocol arms.'],
          'runs':provenance,'summary':summary,'rows':rows}
(OUT/'posterior_diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
csv_fields=[k for k in rows[0] if k not in {'profile_individual_checks','checkpoint_paths_and_hashes'}]
with open(OUT/'posterior_diagnostics.csv','w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=csv_fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
lines=['# 已有后验存档的 CPU 机制诊断','',
       '**20/20 存档可复用为机制诊断，已核实与 July 5 v2 运行对应。无需重训。** 每个存档的Fisher trace/max/维数与原始JSON一致；融合因子与对应任务最终A/B checkpoint逐元素完全一致；文件mtime与v2日志保存时刻相差少于5秒。',
       '', '|历史分支（各10任务、seed2）|终点与保留权重完全相同|Fisher cosine 均值[范围]|Fisher相对L2差均值[范围]|ΔW相对位移均值[范围]|','|---|---:|---:|---:|---:|']
for group in ['ours','baseline_nodamp_nomerge']:
    s=summary[group]
    def fmt(field,percent=False):
        v=s[field];scale=100 if percent else 1;suffix='%' if percent else ''
        return f"{v['mean']*scale:.4f}{suffix} [{v['min']*scale:.4f}, {v['max']*scale:.4f}]"
    lines.append(f"|{group}|{s['n_theta_equals_mu']}/10|{fmt('fisher_cosine')}|{fmt('fisher_relative_l2_to_endpoint')}|{fmt('delta_w_relative_shift',True)}|")
lines += ['', '**关键结论：不融合的 baseline 在10/10任务具有完全相同的终点/保留权重，但两份 Fisher 仍不同。** 因而不能把融合前后Fisher的cosine偏差全部归因于权重融合改变曲率。这里的Fisher为batch-gradient代理，两次估计的随机批次/增强等执行未配对，具有估计器重复噪声。只能报告描述性差异，不能从ours与baseline两组cosine均值相减得到纯融合效应。',
          '', '历史源码 `b194f04` 的 `_capture_endpoint_posterior_state` 在融合前额外执行一次Fisher，之后常规路径又执行融合后Fisher；函数恢复模型模式和梯度，没有保存/恢复RNG。源代码注释“extra pass never feeds back”只指未进入EWC累积量，不代表随机数轨迹不受影响。',
          '', '已有anchor指标在同批当前任务测试样本、task-local logits下比较终点/保留权重；逐任务loss/acc差已保留CSV。它是旧轨迹上的局部诊断，不能替代当前十任务最终性能的融合单变量对照。两个历史完整运行FAA为ours51.400、baseline52.073，但baseline同时移除damping与融合，且诊断额外前向可能改变随机数，不能据此归因单个组件。',
          '', '存档仅有 `theta_end_lora`、`mu_merge_lora`、`fisher_endpoint`、`fisher_merge`；未保存epoch20/40路径因子，不能完整重建路径内部的B@A交叉项。ΔW位移本次直接由24组Q/V的B@A重算；所有张量有限。表中范围/标准差来自同seed10阶段，不是多seed置信区间。',
          '', f"CPU分析峰值RSS约{report['max_rss_mb']:.1f} MiB；未占用GPU，未写原始存档。",'',
          '逐存档路径/哈希、checkpoint身份验证、计算定义、原始指标来源见 [posterior_diagnostics.json](posterior_diagnostics.json)；20行结果见 [posterior_diagnostics.csv](posterior_diagnostics.csv)。','']
(OUT/'posterior_diagnostics.md').write_text('\n'.join(lines))
print(json.dumps({'summary':summary,'identity':report['identity_checks'],'max_rss_mb':report['max_rss_mb']},ensure_ascii=False,indent=2))
