import importlib.util, json, sys
from pathlib import Path
import torch
from r0_probe_audit import audited_probes,compare
HERE=Path(__file__).parent
spec=importlib.util.spec_from_file_location('cpu_original',HERE/'paired_gam_diagnostic_v2.py')
A=importlib.util.module_from_spec(spec);sys.modules[spec.name]=A;spec.loader.exec_module(A)
SOURCE='/data/140-1/users/liying/CopyFlatCV/refine-logs/s3-hparam-search-20260921/source'
sys.path.insert(0,SOURCE)
from optimer_PerturabtionType.gam import GAM
torch.manual_seed(17)
model=torch.nn.Sequential(torch.nn.Linear(4,5),torch.nn.Tanh(),torch.nn.Linear(5,3))
class Geo:
    params=list(model.parameters())
    base=[p.detach().clone() for p in params]
    def parameter_delta(self):return A.flatten([p.detach()-v for p,v in zip(self.params,self.base)])
    def displacement_metrics(self):return {'parameter_norm':A.norm(self.parameter_delta())}
geo=Geo();anchor=A.Snapshot(model);x=torch.randn(8,4);y=torch.arange(8)%3
ref=A.isolated_logits(model,x)
out={'status':'running','positive':[],'negative':[]}
for r in (0.,1.):
    grads,stats,shifts=audited_probes(A,model,geo,anchor,x,y,0,3,.05,r,x,ref,GAM)
    out['positive'].append({'r':r,'passed':stats['native_self_spec']['passed'],'restore':anchor.check()})
class BrokenProbe(GAM):
    @torch.no_grad()
    def perturb_weights(self,perturb_idx):
        super().perturb_weights(perturb_idx)
        self.param_groups[0]['params'][0].add_(.01)
class BrokenDirection(GAM):
    @torch.no_grad()
    def gradient_decompose(self,args=None):
        super().gradient_decompose(args)
        self.param_groups[0]['params'][0].grad.add_(.01)
for name,typ in [('corrupt_native_probe',BrokenProbe),('corrupt_native_final_gradient',BrokenDirection)]:
    try:audited_probes(A,model,geo,anchor,x,y,0,3,.05,0.,x,ref,typ)
    except AssertionError:out['negative'].append({'case':name,'rejected':True,'restore':anchor.check()})
    else:raise AssertionError(name+' unexpectedly passed')
for name,xx,yy in [('corrupt_sameprobe_CE',[torch.ones(3)],[torch.ones(3)*1.1]),('nonfinite_gradient',[torch.full((3,),float('nan'))],[torch.ones(3)])]:
    try:compare(A,xx,yy)
    except (AssertionError,FloatingPointError):out['negative'].append({'case':name,'rejected':True})
    else:raise AssertionError(name+' unexpectedly passed')
out['status']='passed'
(HERE/'CPU_TEST_REPORT.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
