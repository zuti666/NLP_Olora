"""Targeted reproducibility check for the observed r17/u14 raw-Fisher peak.

No training and no replacement of the original observations. Writes only an
independent diagnostic verification below final_review_20260921.
"""
from pathlib import Path
from argparse import Namespace
import fcntl
import json
import math
import os
import sys

HERE=Path(__file__).resolve().parent
EXT=HERE.parent
BASE=EXT.parent
sys.path.insert(0,str(BASE))
from campaign_runner import gpu_status, now


def main():
    status=gpu_status(6)
    assert status['uuid']=='GPU-b74c0891-02cf-152f-704a-bd7edbd0b50a'
    lock_path=BASE.parents[3]/'.cache/gpu_locks'/f"{status['uuid']}.lock"
    with lock_path.open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status=gpu_status(6)
        assert status['memory_mib']<500 and status['utilization']<5 and not status['has_compute_process']
        os.environ['CUDA_VISIBLE_DEVICES']=status['uuid']
        os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
        sys.path.insert(0,str(EXT))
        from diagnostic_extension import preflight,validate_loaded_identity,validate_origin
        from diagnostic_full import FullRunner
        from rq2_common import atomic_json,random_direction,set_point,fisher,sha_file
        import torch
        cfg=json.loads((EXT/'configs/radius_seed0.json').read_text())
        cfg['output_dir']=str(HERE/'peak_repeat_results/seed0')
        cfg_path=HERE/'peak_repeat_config.json'
        assert not (HERE/'FISHER_PEAK_VERIFICATION.json').exists(), 'Do not silently repeat this check'
        atomic_json(cfg_path,cfg)
        args=Namespace(config=str(cfg_path),method='sgd',job='sparse_train',path_task=9,device='cuda:0')
        _,reference,trajectory,evidence=preflight(args)
        runner=FullRunner(args)
        identity=validate_loaded_identity(runner,reference,trajectory)
        origin=runner.origin_full(['subset'])
        origin_check=validate_origin(origin,reference,'sparse_train')
        original_path=EXT/'results/seed0/sgd/sparse_train/states.jsonl'
        original={r['state_id']:r for r in map(json.loads,original_path.read_text().splitlines())}
        direction,info=random_direction(14,cfg['direction_seed'],runner.device)
        repetitions=[]
        for radius in (16,17,18):
            old=original[f'r{radius}_u014']
            assert old['point']['direction']['sha256']==info['sha256']
            row=runner.measure_full(f'repeat_r{radius}_u014',direction,radius,fisher_scopes=['subset'],
                kind='targeted_peak_reproduction',radius=radius,direction=info)
            trace=row['fisher_by_scope']['subset']['raw_trace']
            prior=old['fisher_by_scope']['subset']['raw_trace']
            matched=math.isclose(trace,prior,rel_tol=1e-5,abs_tol=1e-5)
            assert matched,(radius,trace,prior)
            repetitions.append(dict(radius=radius,original_trace=prior,repeated_trace=trace,
                relative_difference=abs(trace-prior)/prior,passed=matched))
        set_point(runner.params,runner.anchor,direction,17)
        individual=fisher(runner.model,runner.collector,runner.params,runner.x['subset'],runner.y['subset'],
                          {},runner.device,8,individual=True)
        assert math.isclose(individual['raw_trace'],repetitions[1]['original_trace'],rel_tol=1e-5)
        idx=max(range(256),key=lambda i:individual['sample_trace'][i])
        x=runner.x['subset'][idx:idx+1];y=runner.y['subset'][idx:idx+1]
        one=fisher(runner.model,runner.collector,runner.params,x,y,{},runner.device,1,individual=True)
        runner.collector.active=False
        runner.model.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(runner.model(x.to(runner.device)),y.to(runner.device))
        grads=torch.autograd.grad(loss,runner.params)
        direct=sum(float(g.double().square().sum()) for g in grads)
        assert math.isclose(direct,one['raw_trace'],rel_tol=1e-5,abs_tol=1e-5)
        runner.restore();end=runner.check_anchor()
        assert end['effective_sha256']==runner.start_checks['effective_sha256']
        result=dict(status='passed',checked_utc=now(),kind='targeted_verification_not_new_formal_observation',
            gpu_uuid=status['uuid'],script_sha256=sha_file(__file__),original_source=str(original_path),
            original_source_sha256=sha_file(original_path),repeated_radii=repetitions,
            identity_check=identity,origin_check=origin_check,end_restoration=end,
            per_example_trace=individual['sample_trace'],mean_raw_trace=individual['raw_trace'],
            peak_subset_index=idx,peak_sample_id=runner.data['subset_ids'][idx],
            peak_sample_trace_batch8=individual['sample_trace'][idx],
            peak_fraction_of_trace_sum=individual['sample_trace'][idx]/sum(individual['sample_trace']),
            single_example_collector_trace=one['raw_trace'],single_example_direct_autograd_trace=direct,
            single_example_collector_matches_direct_autograd=True,
            protocol_notes='Same formal seed0 SGD anchor, r=17/u14, subset256, fixed old20 head, FP32 TF32 off; no lambda. Original figures retain the original values.')
        atomic_json(HERE/'FISHER_PEAK_VERIFICATION.json',result)
        print(json.dumps({k:v for k,v in result.items() if k not in ['per_example_trace','identity_check','origin_check']},indent=2),flush=True)


if __name__=='__main__':
    main()
