"""Prove new direct native observation and actual-seed independent S references."""
import json,tempfile
from pathlib import Path
import torch
import test_legacy_direction_cpu as base
import legacy_direction as extension
from native_support import HERE,MODULE,DIRECT,verify_config,sha256

def main():
    torch.set_num_threads(1);cases={};original_optimizer=base.optimizer
    with tempfile.TemporaryDirectory(prefix='native_seed01_test_') as tmp:
        root=Path(tmp)
        for seed in (0,1):
            base.seed(seed);state=base.Model().state_dict()
            for condition,beta in DIRECT.items():
                def optimizer(learner,beta=beta):
                    opt=original_optimizer(learner)
                    opt.args.grad_beta_1,opt.args.grad_beta_2,opt.args.grad_beta_3,opt.args.grad_gamma=beta
                    return opt
                base.optimizer=optimizer
                plain,plain_mechanism=base.run(state,'native_beta',False,root/'unused')
                observed,observed_mechanism=base.run(state,'native_beta',True,root/f'{condition}_seed{seed}.jsonl')
                assert plain==observed and plain_mechanism==observed_mechanism
                cases[f'{condition}_seed{seed}']={'four_step_parameters_gradients_losses_RNG_mechanism_exact_vs_unwrapped':True,'beta_gamma':beta}
            base.optimizer=original_optimizer
            matchroot=root/f'match_seed{seed}';matchroot.mkdir()
            proof=base.matching_regression(matchroot,state,matching_seed=seed)
            cases[f'Gmatch_seed{seed}']=proof
        suite=json.loads((HERE/'suite.json').read_text());configs=[]
        for job in suite['jobs']:
            if not job['condition'].startswith('N'):continue
            verify_config(job);configs.append(job['id'])
        assert len(configs)==56
    proof={'passed':True,'cases':cases,'configs_validated':configs,'native_formal_configs':28,'native_smoke_configs':28,
        'scope':'CPU exact canonical decomposition and AST-native training; original beta-gamma observational branch; seed0/1 StepReference with negative cross-seed checks',
        'module_hashes':{p.name:sha256(p) for p in MODULE.glob('*.py')}}
    (MODULE/'CPU_PROOF_SEED01.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({'passed':True,'cases':len(cases),'configs':len(configs)}))

if __name__=='__main__':main()
