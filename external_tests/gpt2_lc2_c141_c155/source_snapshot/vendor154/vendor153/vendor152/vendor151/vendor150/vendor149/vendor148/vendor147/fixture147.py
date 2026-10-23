"""Controlled archive fixture; reference replay is not physical or derivative evidence."""
import contextlib,copy,io,os,shutil
from core147 import *
from fixture133 import pack

def make_fixture(root):
    cached=os.environ.get('C147_SYNTHETIC_FIXTURE_CACHE','')
    if cached:
        q=clean_json(Path(cached).read_bytes());need(q.get('synthetic_fixture_only') is True and q['contract']['production'] is False,'Only explicit synthetic test caches permitted');paths={k:Path(v) for k,v in q['paths'].items()};need(set(paths)==set(INPUT_KEYS),'Cached test inventory')
        for k,p in paths.items():compare(sha(p),q['contract']['pins'][k]['sha256'],'Synthetic cache hash/'+k)
        return paths,q['contract']
    from fixture145 import data145,emit,Toy,Analytic as Grad145
    from fixture146 import Native,Analytic,bindings
    from fixture138 import metric
    from worker145 import run as run145
    from worker146 import run as run146
    root=Path(root);root.mkdir(parents=True,exist_ok=True);d=data145();p143=root/'C143';p143.mkdir()
    for sp in registry():d['refs'][(sp['state_id'],'prepared')]=d['refs'][(sp['state_id'],'prepared')][:1]
    # Minimal C143 binding archive: C147 does not claim to re-audit C143 training.
    dump(p143/'SUMMARY.json',dict(protocol=parent.base.PROTOCOL,complete=True,synthetic_fixture=True,scope='Synthetic C143 tensor/native-reference binding only; no historical training execution claim.'))
    np.savez_compressed(p143/'ALL84_TRAINING_INPUTS.npz',**{sid+'__'+k:p[k] for sid,p in d['points'].items() for k in ('prepared','raw','features')})
    for name,values in [('FROZEN_SHARED_ROW_BASIS.npz',dict(V=d['V'])),('FINAL_SHARED_CONTROL_STEP12.npz',dict(a=d['a'])),('INITIAL_C140_STEP12_CONTROL.npz',dict(a=d['C140_a'])),('FROZEN_ORIGINAL_WRITER.npz',d['writer']),('FROZEN_ORIGINAL_CONTROL.npz',d['control'])]:np.savez_compressed(p143/name,**values)
    dump(p143/'MODEL_PINS_REFERENCE.json',d['modelpins']);pre=[];ends=[]
    for sp in parent.prior.registry():
        sid=sp['state_id'];p=d['points'][sid]
        for kind,point,write in [('prepared',p['prepared'],False),('raw',p['raw'],True),('warm',parent.proposal(p['prepared'],p['raw'],d['V'],d['C140_a'])[0],True)]:pre.append(dict(**sp,kind=kind,metrics=metric(point,sp['case'],write)))
    for rnd in (0,1):
        for sp in parent.prior.registry():
            sid=sp['state_id'];p=d['points'][sid]
            for arm,point in [('raw_control',p['raw']),('C140_frozen_step12',parent.proposal(p['prepared'],p['raw'],d['V'],d['C140_a'])[0]),('C143_shared_step12',parent.proposal(p['prepared'],p['raw'],d['V'],d['a'])[0])]:ends.append(dict(**sp,round=rnd,arm=arm,metrics=metric(point,sp['case'],True)))
    dump(p143/'PREFLIGHT_PROGRESS.json',pre);dump(p143/'ALL504_FINAL_TRAINING_ENDPOINTS.json',ends);shutil.copytree(ROOT/'vendor146/vendor145/vendor144/vendor143',p143/'source',ignore=shutil.ignore_patterns('__pycache__'));paths={'C143':pack(p143,root/'C143_SYNTHETIC.zip')}
    for k in ('C144','C140','C141','C142'):
        p=root/(k+'_UNOPENED_REFERENCE.zip');p.write_bytes(('Synthetic unopened original reference '+k).encode());paths[k]=p
    p145=root/'C145';c145=parent.prior.contract();c145.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()})
    def loader145(paths,out,c):dd=copy.deepcopy(d);emit(out,dd,paths);return dd
    with contextlib.redirect_stdout(io.StringIO()):s145=run145(paths,p145,c145,lambda out,d,ph,calls:Toy(out,d,ph,calls),lambda rt,d,ph,calls:Grad145(rt,d,ph,calls),loader145)
    need(s145['complete'],str(s145['errors']));shutil.copytree(ROOT/'vendor146/vendor145',p145/'source',ignore=shutil.ignore_patterns('__pycache__'));paths['C145']=pack(p145,root/'C145_SYNTHETIC.zip');d['C145_summary']=s145
    c146=parent.contract();c146.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});out=root/'C146'
    def loader146(paths,out,c):dd=copy.deepcopy(d);bindings(out,dd,paths);return dd
    with contextlib.redirect_stdout(io.StringIO()):s=run146(paths,out,c146,lambda out,d,ph,calls:Native(out,d,ph,calls,'failed_final'),lambda rt,d,ph,calls:Analytic(rt,d,ph,calls),loader146)
    need(s['complete'],str(s['errors']));need(not s['fixed_final_training_qualification_passed'],'Synthetic final failure must remain')
    dump(out/'ORIGINAL_INPUTS_UNCHANGED.json',{k:True for k in parent.INPUT_KEYS});shutil.copytree(ROOT/'vendor146',out/'source',ignore=shutil.ignore_patterns('__pycache__'));p=pack(out,root/'C146_SYNTHETIC.zip');paths={k:paths[k] for k in ('C143','C145')};paths['C146']=p
    c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()},binding146=c146,expected_parent_decision=s['decision']);return paths,c

def rebound(path,target,changes):
    from fixture144 import rebound as original
    return original(path,target,changes)

def npz_bytes(**arrays):
    f=io.BytesIO();np.savez_compressed(f,**arrays);return f.getvalue()
