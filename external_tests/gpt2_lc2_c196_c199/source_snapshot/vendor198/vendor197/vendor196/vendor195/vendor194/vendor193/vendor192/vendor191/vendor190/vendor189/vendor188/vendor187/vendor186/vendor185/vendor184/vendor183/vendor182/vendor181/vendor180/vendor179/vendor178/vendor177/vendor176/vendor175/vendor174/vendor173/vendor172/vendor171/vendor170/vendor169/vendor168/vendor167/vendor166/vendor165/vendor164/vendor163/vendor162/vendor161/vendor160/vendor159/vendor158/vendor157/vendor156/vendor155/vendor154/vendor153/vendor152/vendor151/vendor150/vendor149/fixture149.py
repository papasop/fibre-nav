"""Test-only original archives with analytic/record-replay doubles; no GPT2 evidence."""
import copy,contextlib,io,os,shutil
from core149 import *
from fixture147 import make_fixture as originals147,rebound,npz_bytes
from fixture133 import pack
from fixture148 import Analytic as BaseAnalytic
from fixture145 import Toy
from fixture138 import metric

class Native(Toy):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.reference={(r['theta_sha256'],tuple(r['bits'])):copy.deepcopy(r) for refs in self.data['refs'].values() for r in refs}
    def metrics(self,point,case,write=True):
        m=super().metrics(point,case,write)
        # Replay original recorded mocks (including deliberately injected C146 failure).
        # These fixtures test archival bindings, not a continuous model or true derivatives.
        key=(safe.tensor_hash(point),tuple(safe.target_bits(case,write)))
        if key in self.reference:m=copy.deepcopy(self.reference[key])
        if self.phase.phase=='paths' and self.calls['path_metrics'] in (4,340):m['observed'][0]=.003;m['response_drift']=.003;m['gates']['response']=False;m['pass']=False
        return m

class Analytic(BaseAnalytic):
    def parity(self,sp,p,arm):
        r=super().parity(sp,p,arm)
        if self.phase.phase=='functional_warm':
            met=self.d['refs'][(sp['state_id'],'C146')][0];r['observed']=copy.deepcopy(met['observed']);r['KL']=met['KL']
        return r

def make_fixture(root):
    cached=os.environ.get('C149_SYNTHETIC_FIXTURE_CACHE','')
    if cached:
        q=clean_json(Path(cached).read_bytes());need(q.get('synthetic_fixture_only') is True and q['contract']['production'] is False,'Explicit synthetic cache only');paths={k:Path(v) for k,v in q['paths'].items()}
        for k,p in paths.items():compare(sha(p),q['contract']['pins'][k]['sha256'],'Synthetic cache original hash')
        return paths,q['contract']
    from audit147 import audit as audit147
    from worker148 import run as run148
    from inputs148 import load as load148
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        pp,pc=originals147(root/'parents')
        # Canonical test-only repack from completed fixture files.
        if not os.environ.get('C147_SYNTHETIC_FIXTURE_CACHE',''):
            canonical=pack(root/'parents/C146',root/'parents/C146_CANONICAL.zip');pp=dict(pp,C146=canonical);pc=copy.deepcopy(pc);pc['pins']['C146']=dict(path=str(canonical),sha256=sha(canonical))
        s147=audit147(pp,root/'C147',pc)
    s147['errors']=[];dump(root/'C147'/'SUMMARY.json',s147);shutil.copytree(ROOT/'vendor148/vendor147',root/'C147'/'source',ignore=shutil.ignore_patterns('__pycache__'));p147=pack(root/'C147',root/'C147.zip');p4=dict(pp,C147=p147);c148=parent.contract();c148.update(production=False,binding147=pc,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in p4.items()})
    with contextlib.redirect_stdout(io.StringIO()):s148=run148(p4,root/'C148',c148,Native,Analytic,load148)
    need(s148['complete'],str(s148['errors']));dump(root/'C148'/'ORIGINAL_INPUTS_UNCHANGED.json',{k:True for k in parent.INPUT_KEYS});shutil.copytree(ROOT/'vendor148',root/'C148'/'source',ignore=shutil.ignore_patterns('__pycache__'));p148=pack(root/'C148',root/'C148.zip');paths=dict(p4,C148=p148);c=contract();c.update(production=False,binding148=c148,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c
