"""Synthetic complete archive, explicitly not GPT2 evidence."""
import contextlib,copy,io,shutil
from unittest.mock import patch
from core144 import *
from fixture143 import archive_inputs,Toy,Analytic,pack
from inputs143 import load
from worker143 import run as run143
import fixture143

def make_fixture(root):
    root=Path(root);paths,binding=archive_inputs(root/'originals');folder=root/'C143';holder={};known={};original_metric=fixture143.metric
    def metric(point,case,write):
        key=(safe.tensor_hash(point),case,write)
        return copy.deepcopy(known[key]) if key in known else original_metric(point,case,write)
    def loader(paths,out,c):
        d=load(paths,out,c);holder['data']=d;holder['receipt']=clean_json((out/'C142_BOUND_RECEIPT.json').read_bytes())
        for sp in parent.registry():
            sid=sp['state_id'];p=d['points'][sid]
            for kind,point,w in [('prepared',p['prepared'],False),('raw',p['raw'],True),('warm',parent.proposal(p['prepared'],p['raw'],d['V'],d['initial_a'])[0],True)]:known[(safe.tensor_hash(point),sp['case'],w)]=d['native_refs'][(sid,kind)][0]
        return d
    with contextlib.redirect_stdout(io.StringIO()),patch.object(fixture143,'metric',metric):
        s=run143(paths,folder,binding,lambda out,d,ph,calls:Toy(out,d,ph,calls,'failed_final'),lambda rt,d,ph,calls:Analytic(rt,d,ph,calls),loader)
    need(s['complete'],'Synthetic C143 must complete: '+repr(s['errors']));need(s['decision']=='C143_FIXED_FINAL_TRAINING_CONSTRAINT_FAILURE','Synthetic failure retained')
    shutil.copytree(ROOT/'vendor143',folder/'source',ignore=shutil.ignore_patterns('__pycache__'))
    z=pack(folder,root/'C143_SYNTHETIC.zip');paths=dict(paths,C143=z);c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},binding143=binding)
    return paths,c,holder,folder

def rebound(original,target,changed):
    import zipfile
    with zipfile.ZipFile(original) as z:members={n:z.read(n) for n in z.namelist() if n!='RESULT_MANIFEST.json'}
    for k,v in changed.items():
        if v is None:members.pop(k,None)
        else:members[k]=v
    members['RESULT_MANIFEST.json']=(json.dumps({n:hashlib.sha256(v).hexdigest() for n,v in members.items()},indent=2)+'\n').encode()
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for n,v in members.items():z.writestr(n,v)
    return target
