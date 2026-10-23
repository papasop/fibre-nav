"""Full analytic archives/runtime for wiring checks, not real model evidence."""
import copy,functools,shutil,tempfile,contextlib,io
from unittest.mock import patch
from core154 import *
from fixture153 import base_fixture as earlier_fixture,pack
from fixture152 import loader as analytic_loader
from fixture138 import metric
from audit153 import audit as prior_audit
from inputs154 import load as production_load
_CACHE=[]

@functools.lru_cache(None)
def inputs_fixture():
    tmp=tempfile.TemporaryDirectory();_CACHE.append(tmp);root=Path(tmp.name);_,paths,pc=earlier_fixture();folder=root/'C153'
    with prior.no_models(),safe.no_solvers(),contextlib.redirect_stdout(io.StringIO()):s=prior_audit(paths,folder,pc,input_loader=analytic_loader)
    dump(folder/'SUMMARY.json',dict(s,errors=[]));shutil.copytree(ROOT/'vendor153',folder/'source',ignore=shutil.ignore_patterns('__pycache__'));archive=root/'C153.zip';pack(folder,archive);paths=dict(paths,C153=archive);c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},audit153=pc);return paths,c

def loader(paths,out,c):
    with patch('inputs154.load152',analytic_loader):return production_load(paths,out,c)

class Native:
    def __init__(self,out,d,phase,calls,mode='normal'):
        self.phase=phase;self.calls=calls;self.mode=mode;self.pins=d['modelpins'];calls['model_forwards']+=1;calls['initialisation_model_forwards']=1
        if mode=='mutate_seal':(Path(out).parent/'STENCIL_PARAMETERS.npz').write_bytes(b'changed')
    def metrics(self,point,case,write=True):
        self.phase.consume(point,case,write);pre=self.phase.phase=='preflight';attempt='preflight_metric_attempts' if pre else 'path_metric_attempts';done='preflight_metrics' if pre else 'path_metrics';self.calls[attempt]+=1;self.calls['model_forwards']+=2;m=metric(point,case,write);self.calls[done]+=1
        if self.mode=='preflight_drift' and pre:m['observed'][0]+=.001
        if self.mode=='interrupt' and self.calls['path_metrics']==1:raise RuntimeError('Synthetic interruption at first probe')
        if self.mode=='mutate_future_point' and not pre and self.calls['path_metrics']==1:
            key=next(k for k in self.phase.points if k[1]=='plus_one');self.phase.points[key][0]+=.1
        if self.mode=='repeat_drift' and not pre and self.calls['path_metrics']>1008:m['observed'][0]+=.0003
        if self.mode in ('preflight_drift','repeat_drift'):
            m['response_drift']=max(abs(x) for x in m['observed'][:6]);m['gates']['response']=m['response_drift']<=.002;m['pass']=all(m['gates'].values())
        return m
