"""Independent CPU acceptance negatives for serialized base/final cache semantics."""
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'trainer'))
import torch
from response_step import ResponseStep
from validate_job import validate_cached_state,tensor_digest

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__();self.theta=torch.nn.Parameter(torch.tensor([.731,-.431,.271]))

H=torch.tensor([[2.4,.3,-.1],[.3,1.2,.2],[-.1,.2,.8]])
positive=[];negative=[]
def rejected(state,last,method,label):
    try:
        validate_cached_state(state,last,method,3)
    except (AssertionError,ValueError,KeyError,AttributeError,TypeError) as e:
        negative.append(dict(method=method,label=label,error=str(e)))
    else:
        raise AssertionError('Bad cache passed: '+label)

for method in ('sgd','current_g2','history_fusion','history_g2'):
    model=Model();source='base' if method=='history_fusion' else 'final' if method=='history_g2' else None
    stepper=ResponseStep(model,list(model.parameters()),None,dict(method=method,history_weight=.5 if source else 0.))
    opt=torch.optim.SGD(model.parameters(),lr=.043)
    for step in range(3):
        b=torch.tensor([.11,-.22,.07])+step*.02
        def closure(_x,_y):
            p=model.theta;return .5*torch.dot(p,H@p)+torch.dot(b,p),p
        result=stepper.compute(None,None,closure,.043)
        stepper.apply_update(.043,opt)
    last=copy.deepcopy(result['record']);last['identity']={'indices':[4,5,6],'fixture':True}
    state=dict(algorithm=method,history_source=source,algorithm_state=stepper.state_dict(),previous_batch_identity=copy.deepcopy(last['identity']))
    validate_cached_state(state,last,method,3);positive.append(method)
    bad=copy.deepcopy(state);bad['algorithm_state']['cache_source']='base' if source!='base' else 'final'
    rejected(bad,last,method,'top-level-cache-source-swapped')
    bad=copy.deepcopy(state);bad['previous_batch_identity']={'indices':[99]}
    rejected(bad,last,method,'wrong-final-batch-identity')
    if source:
        h=state['algorithm_state']['history'];assert h is not None
        bad=copy.deepcopy(state);bad['algorithm_state']['history']=None
        rejected(bad,last,method,'missing-history-tensors')
        bad=copy.deepcopy(state);bad['algorithm_state']['history']['cache_source']='base' if source=='final' else 'final'
        rejected(bad,last,method,'historical-vector-source-swapped')
        for label,replacement in (
            ('unit-vector-instead-of-full-amplitude',tuple(v/torch.linalg.vector_norm(v) for v in h['vector'])),
            ('wrong-base-versus-final-vector',result['context']['U'] if source=='base' else result['context']['g0']),
            ('displacement-instead-of-gradient',h['displacement'])):
            bad=copy.deepcopy(state);hb=bad['algorithm_state']['history'];hb['vector']=tuple(v.detach().clone() for v in replacement)
            hb['vector_sha256']=tensor_digest(hb['vector'])
            rejected(bad,last,method,label)
        bad=copy.deepcopy(state);hb=bad['algorithm_state']['history'];hb['displacement']=tuple(-.043*v for v in result['context']['U'])
        assert not all(torch.equal(a,b) for a,b in zip(hb['displacement'],h['displacement']))
        hb['displacement_sha256']=tensor_digest(hb['displacement'])
        rejected(bad,last,method,'ideal-lr-vector-instead-of-actual-FP32-displacement')
        bad=copy.deepcopy(state);bad['algorithm_state']['history']['length']*=2
        rejected(bad,last,method,'wrong-actual-displacement-length')
    else:
        bad=copy.deepcopy(state);bad['algorithm_state']['history']={'vector':(torch.tensor([1.]),)}
        rejected(bad,last,method,'retained-cache-in-no-history-method')

report=dict(status='passed',audited_at_utc=datetime.now(timezone.utc).isoformat(),cpu_only=True,
            positive_methods=positive,negative_fixtures=len(negative),rejections=negative,
            validator_sha256=hashlib.sha256((ROOT/'validate_job.py').read_bytes()).hexdigest(),
            core_sha256=hashlib.sha256((ROOT/'trainer/response_step.py').read_bytes()).hexdigest())
(ROOT/'gates/INDEPENDENT_CACHE_ACCEPTANCE_FIXTURE.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'positive_methods':positive,'negative_fixtures':len(negative),'validator_sha256':report['validator_sha256']},indent=2))
