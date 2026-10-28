"""Separate full-dimension analytic fixtures. They are never GPT-2 evidence."""
import contextlib,copy,io,shutil
from core139 import *
from fixture138 import worker_data,metric,Toy as Toy138
from worker138 import run as run138
from fixture133 import pack

def dummy_inputs(root):
    paths={};pins={}
    for k in INPUT_KEYS:
        p=root/(k+'.zip');p.write_bytes(('synthetic C139 input '+k).encode());paths[k]=p;pins[k]=dict(path=str(p),sha256=sha(p))
    return paths,dict(contract(),production=False,pins=pins)

def data139():
    d=worker_data();fresh,_=previous.fresh_coordinates(d);pre={}
    for sp in previous.registry():
        m=metric(fresh[sp['state_id']],sp['case'],False);pre[sp['state_id']]=dict(preparation=m,preparation_valid=bool(safe.metric_gates(m,sp['case'],fresh[sp['state_id']],False)))
    d.update(C138_fresh=fresh,C138_prewrite=pre,C138_summary=dict(decision='C138_FROZEN_LOCAL_CONFIRMATION_FAILURE'));return d

class Toy(Toy138):
    def __init__(self,out,data,phase,calls,mode='normal'):super().__init__(out,data,phase,calls,'normal');self.test_mode=mode
    def features(self,point,desired):
        x,o=super().features(point,desired)
        if self.test_mode=='feature_drift' and self.phase.phase=='controls':x[0]+=.001
        if self.test_mode=='prep_invalid' and self.phase.phase=='prewrite':o[0]=.003;x[3077]=.003/.002
        if self.test_mode=='mutate_candidate' and self.phase.phase=='prewrite':self.data['a'][0]+=.01
        return x,o
    def metrics(self,point,case,write=True):
        if self.test_mode=='interrupt' and self.phase.phase=='evaluation' and self.measurements==5:raise RuntimeError('Synthetic interrupted endpoint')
        m=super().metrics(point,case,write)
        drift=(self.test_mode=='control_drift' and self.phase.phase=='controls') or (self.test_mode=='full_drift' and self.phase.phase=='full_replay') or (self.test_mode=='repeat_drift' and self.phase.phase=='evaluation' and self.measurements>48) or (self.test_mode=='prep_repeat_drift' and self.phase.phase=='kernel_preparation' and sum(k[0]=='kernel_preparation' for k in self.phase.used)>24)
        if self.test_mode=='prep_invalid' and self.phase.phase=='kernel_preparation':m['observed'][0]=.003;m['response_drift']=max(abs(x) for x in m['observed'][:6]);m['gates']['response']=False;m['pass']=False
        if drift:
            m['observed'][0]+=.0003;m['response_drift']=max(abs(x) for x in m['observed'][:6]);m['gates']['response']=m['response_drift']<=.002;m['pass']=all(m['gates'].values())
        return m

def archive138(root):
    root.mkdir();d=worker_data();paths,c=dummy_inputs(root);paths={k:paths[k] for k in previous.INPUT_KEYS};c=dict(c,pins={k:c['pins'][k] for k in previous.INPUT_KEYS});out=root/'C138'
    def loader(paths,out,c):
        from inputs138 import validate_frozen_inputs
        dd=copy.deepcopy(d)
        dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(paths[k]),sha256=sha(paths[k]),known_before_C138=True) for k in previous.INPUT_KEYS})
        dump(out/'BOUND_HISTORICAL_COORDINATE_HASHES.json',dict(hashes=sorted(dd['historical_hashes']),scope='Available states and endpoints in six bound originals; not exhaustive programme history',exhaustive_historical_catalog_verified=False))
        dump(out/'C137_BOUND_RECEIPT.json',dict(synthetic_fixture=True))
        dump(out/'FROZEN_WRITER_SCHEMA_VALIDATION.json',validate_frozen_inputs(dd['writer'],dd['control']))
        return dd
    with contextlib.redirect_stdout(io.StringIO()):s=run138(paths,out,c,Toy138,loader)
    need(s['complete'],'Synthetic C138 worker failed');shutil.copytree(ROOT/'vendor138',out/'source',ignore=shutil.ignore_patterns('__pycache__'));z=pack(out,root/'C138_SYNTHETIC.zip');return z,d,c
