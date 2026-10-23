"""Synthetic full-coordinate archives, explicitly not real model evidence."""
import copy,contextlib,io,shutil
from core142 import *
from fixture140 import data140
from fixture138 import metric
from fixture141 import Toy as BaseToy
from worker141 import run as run141
from fixture133 import pack

class Toy(BaseToy):
    def metrics(self,point,case,write=True):
        evaluation=self.phase.phase=='evaluation';i=self.endpoints;result=super().metrics(point,case,write)
        if evaluation:
            sp=prior.registry()[(i%144)//3];arm=ARMS[i%3];sid=sp['state_id']
            c140={r['state_id'] for r in prior.registry() if (r['case']=='xor_00' and '_p02_n01_d01' in r['state_id']) or (r['case']=='xor_01' and '_p05_n01_d00' in r['state_id']) or (r['case']=='xor_11' and any(x in r['state_id'] for x in ('_p08_n01_d00','_p08_n01_d01','_p09_n01_d00','_p11_n01_d01')))}
            c137=c140|{r['state_id'] for r in prior.registry() if '_p01_n01_d01' in r['state_id'] or '_p10_n01_d00' in r['state_id']}
            raw={r['state_id'] for r in prior.registry()[:16] if '_p01_' in r['state_id']}|{r['state_id'] for r in prior.registry() if '_p05_n01_d00' in r['state_id']}|{r['state_id'] for r in prior.registry()[32:43]}
            failed=sid in ({ARMS[0]:raw,ARMS[1]:c137,ARMS[2]:c140}[arm]);result['observed'][0]=.0023 if failed else .001;result['response_drift']=float(max(abs(x) for x in result['observed'][:6]));result['gates']['response']=not failed;result['pass']=all(result['gates'].values())
        return result

def make_fixture(root):
    root=Path(root);root.mkdir(parents=True);d=data140();d['a140']=d['initial_a']-.002;d['a']=d['initial_a'].copy();d['C140_summary']=dict(decision='C140_FIXED_FINAL_ALL36_TRAINING_GATES_PASSED');d['C140_points']={};d['C140_controls']={};pre=[];final=[]
    for sp in prior.parent.registry():
        sid=sp['state_id'];p=d['points'][sid]
        for kind,point,w in [('prepared',p['prepared'],False),('raw',p['raw'],True),('warm',prior.parent.proposal(p['prepared'],p['raw'],d['V'],d['initial_a'])[0],True)]:pre.append(dict(**sp,kind=kind,metrics=metric(point,sp['case'],w)))
        for arm,a in [(prior.parent.EVAL_ARMS[0],np.zeros(len(d['V']))),(prior.parent.EVAL_ARMS[1],d['initial_a']),(prior.parent.EVAL_ARMS[2],d['a140'])]:
            point=prior.parent.proposal(p['prepared'],p['raw'],d['V'],a)[0];d['C140_points'][(sid,arm)]=point
            for j in (0,1):
                rec=dict(**sp,arm=arm,round=j,metrics=metric(point,sp['case'],True));d['C140_controls'][(sid,arm,j)]=rec;final.append(rec)
    p140=root/'p140';p140.mkdir()
    dump(p140/'SUMMARY.json',dict(protocol=prior.parent.PROTOCOL,complete=True,decision='C140_FIXED_FINAL_ALL36_TRAINING_GATES_PASSED',synthetic_fixture=True,real_model_execution_claim=False))
    np.savez_compressed(p140/'ALL36_TRAINING_INPUTS.npz',**{sid+'__'+k:p[k] for sid,p in d['points'].items() for k in ('prepared','raw','features')})
    np.savez_compressed(p140/'FROZEN_SHARED_ROW_BASIS.npz',V=d['V']);np.savez_compressed(p140/'INITIAL_C137_STEP24_CONTROL.npz',a=d['initial_a']);np.savez_compressed(p140/'FINAL_SHARED_CONTROL_STEP12.npz',a=d['a140']);np.savez_compressed(p140/'FROZEN_ORIGINAL_WRITER.npz',**d['writer']);np.savez_compressed(p140/'FROZEN_ORIGINAL_CONTROL.npz',**d['control'])
    (p140/'training/step12').mkdir(parents=True);np.savez_compressed(p140/'training/step12/UPDATE.npz',after_a=d['a140'])
    dump(p140/'STEP12_FREEZE_BEFORE_EVALUATION.json',dict(completed_steps=12,checkpoint_choice='Fixed C140 step12 only',no_best_or_recovery_selection=True,control_digest=row_digest(d['a140']),control_sha256=sha(p140/'FINAL_SHARED_CONTROL_STEP12.npz')))
    dump(p140/'PREFLIGHT_PROGRESS.json',pre);dump(p140/'ALL216_FINAL_TRAINING_ENDPOINTS.json',final);z140=pack(p140,root/'C140_SYNTHETIC.zip')
    paths={k:root/(k+'_DUMMY.zip') for k in prior.INPUT_KEYS if k!='C140'}
    for k,p in paths.items():p.write_bytes(('synthetic '+k).encode())
    paths['C140']=z140;c141=prior.contract();c141.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()})
    out=root/'p141'
    def loader(paths,out,c):
        dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p),known_before_C141=True) for k,p in paths.items()});dump(out/'BOUND_HISTORICAL_COORDINATE_HASHES.json',dict(hashes=sorted(d['historical_hashes']),exhaustive_programme_history_verified=False));dump(out/'C140_BOUND_RECEIPT.json',dict(synthetic_fixture=True));return d
    with contextlib.redirect_stdout(io.StringIO()):s=run141(paths,out,c141,Toy,loader)
    need(s['complete'] and not s['local_confirmation_passed'],'Synthetic negative confirmation required')
    shutil.copytree(ROOT/'vendor141',out/'source',ignore=shutil.ignore_patterns('__pycache__'));z141=pack(out,root/'C141_SYNTHETIC.zip')
    c=contract();c.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in {'C141':z141,'C140':z140}.items()})
    return dict(C141=z141,C140=z140),c
