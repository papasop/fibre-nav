"""Analytic full-size fixture only. No model evidence from these functions."""
import copy,functools,io,shutil,tempfile
from unittest.mock import patch
from core155 import *
from fixture154 import inputs_fixture as old_inputs,loader as old_loader,Native
from fixture153 import pack
from fixture138 import metric as analytic_metric
from worker154 import run as old_run
from inputs155 import load as actual_load
_TEMP=[]
@functools.lru_cache(None)
def inputs_fixture():
    tmp=tempfile.TemporaryDirectory();_TEMP.append(tmp);root=Path(tmp.name);paths,pc=old_inputs();out=root/'C154'
    with contextlib.redirect_stdout(io.StringIO()):s=old_run(paths,out,pc,lambda o,d,p,c:Native(o,d,p,c),old_loader)
    need(s['complete'],'Synthetic parent complete');shutil.copytree(ROOT/'vendor154',out/'source',ignore=shutil.ignore_patterns('__pycache__'));archive=root/'C154.zip';pack(out,archive);paths=dict(paths,C154=archive);c=contract();c.update(production=False,parent=pc,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c
def loader(paths,out,c):return actual_load(paths,out,c,parent_loader=old_loader)
@functools.lru_cache(None)
def bound():
    paths,c=inputs_fixture();tmp=tempfile.TemporaryDirectory();_TEMP.append(tmp)
    with contextlib.redirect_stdout(io.StringIO()):d,h,p,r,pc=loader(paths,Path(tmp.name),c)
    return d,h,p,r,pc
def cached_loader(paths,out,c):return copy.deepcopy(bound())
class Pair:
    def __init__(self,rt,calls,mode='normal'):
        self.calls=calls;self.mode=mode;calls['model_forwards']+=1;calls['functional_FP64_forwards']+=1;calls['FP64_baseline_forwards']+=1
    def functional32(self,point,case):
        if self.mode=='functional_failure':raise RuntimeError('Synthetic functional failure')
        self.calls['model_forwards']+=2;self.calls['functional_FP32_forwards']+=2;self.calls['functional_FP32_metrics']+=1;m=analytic_metric(point,case,True)
        if self.mode=='zero_parity_drift':m=full_metric(np.asarray(m['observed'])+np.r_[.001,np.zeros(21)],m['KL'],point,case)
        return m
    def precision64(self,point,case):
        self.calls['model_forwards']+=2;self.calls['functional_FP64_forwards']+=2;self.calls['FP64_focus_metrics']+=1;m=analytic_metric(point,case,True);o=np.asarray(m['observed']);offset=np.arange(6)*1e-7;cross=o.copy();cross[:6]+=offset
        return dict(point_sha256=safe.tensor_hash(point),observed_local_reference=o.tolist(),observed_FP32_response_reference=cross.tolist(),KL_local_reference=m['KL'],KL_FP32_logits_reference_recomputed64=m['KL'],response_reference_offset=offset.tolist(),response_baseline_decomposition_residual_Linf=0.,precision='analytic fixture, not real FP64',acceptance_evaluated=False,training_qualified=False)
    def verify(self,rehash=False):pass
