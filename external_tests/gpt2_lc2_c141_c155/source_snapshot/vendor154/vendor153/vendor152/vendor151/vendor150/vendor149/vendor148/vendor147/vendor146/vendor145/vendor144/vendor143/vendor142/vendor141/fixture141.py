"""Separate archive/worker analytic fixtures; no GPT-2 evidence."""
import contextlib,copy,io,shutil
from core141 import *
from fixture140 import data140,loader as loader140,Toy as Toy140,Analytic as Analytic140,dummy_inputs as inputs140
from fixture138 import capture,metric
from worker140 import run as run140
from fixture133 import pack

def archive140(root):
    root.mkdir();paths,c=inputs140(root);out=root/'C140'
    def oldloader(paths,out,c):
        d=loader140(paths,out,c);dump(out/'ALL36_TRAINING_REGISTRY_FROZEN.json',[dict(r,prepared_sha256=safe.tensor_hash(d['points'][r['state_id']]['prepared']),raw_sha256=safe.tensor_hash(d['points'][r['state_id']]['raw'])) for r in parent.registry()]);return d
    with contextlib.redirect_stdout(io.StringIO()):s=run140(paths,out,c,Toy140,Analytic140,oldloader)
    need(s['complete'] and s['fixed_final_training_qualification_passed'],'Synthetic C140 archive incomplete');shutil.copytree(ROOT/'vendor140',out/'source',ignore=shutil.ignore_patterns('__pycache__'));z=pack(out,root/'C140_SYNTHETIC.zip');c=dict(contract(),production=False,pins=dict(c['pins'],C140=dict(path=str(z),sha256=sha(z))));return dict(paths,C140=z),z,data140(),c

class Toy:
    def __init__(self,out,data,phase,calls,mode='normal'):self.data=data;self.phase=phase;self.calls=calls;self.mode=mode;self.pins=data['modelpins'];self.endpoints=0;calls['model_forwards']+=1;calls['initialisation_model_forwards']=1
    def features(self,point,desired):
        self.phase.consume_feature(point,desired);self.calls['feature_attempts']+=1;self.calls['feature_calls']+=1;self.calls['model_forwards']+=3;x=capture(point,desired)
        sp=next(sp for sp in parent.parent.registry() if (int(sp['case'][-2])^int(sp['case'][-1]))==desired and np.array_equal(np.array(safe.target_bits(sp['case'],False))[:2],(point[6:8]>=0).astype(int)));o=np.array(metric(point,sp['case'],False)['observed'])
        if self.mode=='feature_drift' and self.phase.phase=='controls':x[0]+=.001
        if self.mode=='invalid_preparation' and self.phase.phase=='prewrite':o[0]=.003;x[3077]=.003/.002
        if self.mode=='mutate_candidate' and self.phase.phase=='prewrite':self.data['a140'][0]+=.01
        return x,o
    def metrics(self,point,case,write=True):
        if self.mode=='interrupt' and self.phase.phase=='evaluation' and self.endpoints==5:raise RuntimeError('Synthetic interrupted endpoint')
        self.phase.consume(point,case,write);self.calls['metric_attempts']+=1;self.calls['metric_calls']+=1;self.calls['model_forwards']+=2;m=metric(point,case,write)
        if self.phase.phase=='evaluation':self.endpoints+=1
        drift=(self.mode=='control_drift' and self.phase.phase=='controls') or (self.mode=='repeat_drift' and self.phase.phase=='evaluation' and self.endpoints>144) or (self.mode=='prep_repeat_drift' and self.phase.phase=='preparation' and sum(k[0]=='preparation' for k in self.phase.used)>48)
        invalid=(self.mode=='invalid_preparation' and self.phase.phase=='preparation') or (self.mode=='candidate_failure' and self.phase.phase=='evaluation' and self.endpoints in (3,147))
        if invalid:m['observed'][0]=.003;m['response_drift']=max(abs(x) for x in m['observed'][:6]);m['gates']['response']=False;m['pass']=False
        if drift:m['observed'][0]+=.0003;m['response_drift']=max(abs(x) for x in m['observed'][:6]);m['gates']['response']=m['response_drift']<=.002;m['pass']=all(m['gates'].values())
        return m
    def invariant(self):pass
