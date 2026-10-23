"""Analytic nonlinear archive fixture and forward adapter; explicitly not GPT2."""
import contextlib,io,shutil
from core134 import *
from fixture133 import make_fixture as fixture132,pack,metric
from worker133 import run as run133

def make_fixture(folder,unavailable=False):
    folder=Path(folder);path,c32=fixture132(folder/'C132',unavailable);root=folder/'C132/parent'
    source=dict(c32['source'])
    for p in sorted((ROOT/'runtime_vendor').glob('*.py')):
        target=root/'source/runtime_vendor'/p.name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(p.read_bytes());source['runtime_vendor/'+p.name]=sha(p)
    dump(root/'model_runtime/MODEL_PINS.json',dict(revision='SYNTHETIC_NONLINEAR_NOT_GPT2',device='cpu',dtype='float32',torch='fixture'))
    path=pack(root,path);c32=dict(c32,source=source,pin=dict(path=str(path),sha256=sha(path)))
    out=folder/'C133'
    with contextlib.redirect_stdout(io.StringIO()):s=run133(path,out,c32)
    need(s['complete'],'Synthetic C133 generation incomplete')
    shutil.copytree(ROOT/'vendor133',out/'source',ignore=shutil.ignore_patterns('__pycache__'));p133=pack(out,folder/'C133_SYNTHETIC.zip')
    c=dict(contract(),production=False,pins=dict(C132=dict(path=str(path),sha256=sha(path)),C133=dict(path=str(p133),sha256=sha(p133))),source132=source)
    return dict(C132=path,C133=p133),c

class Toy:
    def __init__(self,out,data,phase,calls,preflight_drift=0.,path_drift=0.,wrong_pin=False,fail_path_attempt=None):
        self.phase=phase;self.calls=calls;self.preflight_drift=preflight_drift;self.path_drift=path_drift;self.fail_path_attempt=fail_path_attempt;self.pins=dict(data['modelpins']);calls['model_forwards']+=1;calls['initialisation_model_forwards']=1
        if wrong_pin:self.pins['revision']='wrong'
    def metrics(self,point,case,write=True):
        phase=self.phase.phase;self.phase.consume(point,case,write);counter='preflight' if phase=='preflight' else 'path';self.calls[counter+'_metric_attempts']+=1
        if counter=='path' and self.calls['path_metric_attempts']==self.fail_path_attempt:raise RuntimeError('Synthetic interrupted path measurement')
        self.calls['model_forwards']+=2;m=metric(point,case,write);drift=self.preflight_drift if phase=='preflight' else self.path_drift
        if drift:
            m['observed'][0]+=drift;m['response_drift']=float(max(abs(v) for v in m['observed'][:6]));m['gates']['response']=m['response_drift']<=.002;m['pass']=all(m['gates'].values())
        self.calls[counter+'_metrics']+=1;return m
