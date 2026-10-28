"""Fixed eight-step FP64 coefficient training; original physical FP32 acceptance."""
from pathlib import Path
import sys,json,time,hashlib,contextlib
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor160'))
import core160 as previous
from core160 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,safe,DIM,array_digest,point_hash,parameter_hash,registry,objective,aggregate,loss,native_summary,archive155,arithmetic
PROTOCOL='GPT2_LC2_C161_FULL84_FP64_TRAIN_CPU_V1'
INPUT_KEYS=previous.INPUT_KEYS+('C160',)
N=84;STEPS=8;RATE=1e-5;TIMEOUT=10800
REQUESTED_FORWARDS=2187;REQUESTED_BACKWARDS=1344;FUNCTIONAL_FORWARDS=1514
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=previous.contract(),source160=clean_json((ROOT/'trusted/C160_SOURCE_PINS.json').read_bytes()))
def design():
 return dict(protocol=PROTOCOL,training_registry=registry(),initial_head='Exact C160 alpha0 = C155 small_step12 BEFORE FP64 3x2x36 tensor; never initialize from an observed best panel.',training_states=84,exposed_training_only=True,optimizer=dict(name='manual projected FP64 Adam',updates=STEPS,learning_rate=RATE,betas=[.9,.999],epsilon=1e-8,global_gradient_L2_clip=1.,coordinate_box=[-.125,.125],moments='Fresh explicit zeros; not resumed C152 optimizer state',schedule='Constant; no stopping, search, repair, backtracking or best checkpoint selection'),final_checkpoint='Exact step8 only, fixed before model load; intermediate heads are audit history, not alternatives.',router='All84 original frozen prewrite routing weights, V, prepared and raw tensors unchanged; no refit or feature recomputation.',continuous_map=previous.design()['map'],references=previous.design()['FP64_reference'],native_reference=previous.design()['native_reference'],loss=previous.design()['loss'],acceptance_budgets=previous.design()['acceptance_budgets'],gradient='Two scalar reverse calls per state: response and summed guards; aggregate all84 with exact maximizing-tie weights. Only coefficient arrays updated; model parameters remain frozen.',seal_order='Initial head/constants/objective references and1092-request registry before model; each current head and all84 continuous points before its gradient batch; final step8 head and all84 original NumPy-to-FP32 proposals before final evaluation.',measurement_order='Native initial round0; eight complete84 gradient/update batches; final84 FP64 objective; native final round0; native initial round1; native final round1.',counts=dict(model_initialization=3,training_records=672,final_FP64_records=84,native_FP32_records=336,total_records=1092,model_forwards=REQUESTED_FORWARDS,functional_FP64_forwards=FUNCTIONAL_FORWARDS,scalar_backwards=REQUESTED_BACKWARDS,coefficient_updates=STEPS),within_run_repeat_tolerances=previous.design()['native_repeat_tolerances'],worker_budget_seconds=TIMEOUT,confirmation_states=0,excluded_input_states=0,thresholds_changed=False,LC2_closed=False,scope='All84 exposed training states only. Fixed final step8 produces a frozen head for two repeated native FP32 tests with unchanged original gates. Full training success is neither fresh seen-class confirmation nor excluded-input generalisation. No convergence, global error bound or LC2 closure claim. All earlier decisions retained. Preparation provenance inherited, not re-prepared.')
def update(p,g,m,v,step):
 need(type(step) is int and 1<=step<=STEPS,'Registered optimizer step')
 for a in (p,g,m,v):need(a.dtype==np.float64 and a.shape==p.shape and np.isfinite(a).all(),'Finite FP64 optimizer arrays')
 need(np.all(v>=0) and np.max(abs(p))<=.125,'Legal optimizer state')
 norm=float(np.linalg.norm(g));need(np.isfinite(norm),'Finite gradient norm');factor=1. if norm<=1. else 1./norm;clipped=g*factor
 mm=.9*m+.1*clipped;vv=.999*v+.001*clipped*clipped
 unconstrained=p-RATE*(mm/(1-.9**step))/(np.sqrt(vv/(1-.999**step))+1e-8)
 new=np.clip(unconstrained,-.125,.125)
 return new,mm,vv,dict(step=step,gradient_L2=norm,clip_factor=factor,clipped_gradient_L2=float(np.linalg.norm(clipped)),projected_coordinates=int(np.count_nonzero(new!=unconstrained)),update_Linf=float(np.max(abs(new-p))))
def mapped(c,p,sid):
 farray(p,c['P0'].shape);need(p.dtype==np.float64 and np.max(abs(p))<=.125,'Finite fixed-box coefficient head')
 w=c[sid+'__weights'];e=(w[0]*p[0]+w[1]*p[1])+w[2]*p[2];u=c['V']@(c[sid+'__raw']-c[sid+'__prepared']);pt=c[sid+'__raw']+c['V'].T@(e[0]*u+e[1]*np.roll(u,1));return pt,e
def seal_head(c,p,out,tag,physical=False):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);points={};native={};rows=[]
 for sp in registry():
  sid=sp['state_id'];pt,e=mapped(c,p,sid);points[(sid,'zero')]=pt
  row=dict(state_id=sid,continuous_FP64_sha256=point_hash(pt))
  if physical:
   q=arithmetic.proposal(c[sid+'__prepared'].astype(np.float32),c[sid+'__raw'].astype(np.float32),c['V'],e);need(q.dtype==np.float32 and np.isfinite(q).all(),'Original physical proposal map');native[(sid,'zero')]=q;row['physical_FP32_sha256']=safe.tensor_hash(q)
  rows.append(row)
 np.savez_compressed(out/'HEAD.npz',P=p)
 if physical:np.savez_compressed(out/'PHYSICAL_FP32_POINTS.npz',**{sid:a for (sid,_),a in native.items()})
 dump(out/'POINT_BINDINGS.json',rows);receipt=dict(tag=tag,head_digest=parameter_hash(p),state_ids=[s['state_id'] for s in registry()],before_measurement=True,artifact_seals={f.name:sha(f) for f in out.iterdir() if f.is_file()});dump(out/'SEALED.json',receipt)
 return points,native,receipt
def requests():
 rr=[dict(stage='native_initial0',state_id=s['state_id'],step=0) for s in registry()]
 rr += [dict(stage='training',state_id=s['state_id'],step=t) for t in range(STEPS) for s in registry()]
 rr += [dict(stage='final_FP64',state_id=s['state_id'],step=STEPS) for s in registry()]
 for stage,t in [('native_final0',STEPS),('native_initial1',0),('native_final1',STEPS)]:rr.extend(dict(stage=stage,state_id=s['state_id'],step=t) for s in registry())
 return rr
def paired(initial,final):
 a=native_summary(initial);b=native_summary(final);rescued=[];destroyed=[]
 for x,y in zip(initial,final):
  compare(x['state_id'],y['state_id']);u=x['qualification']['complete_pass'];v=y['qualification']['complete_pass']
  if v and not u:rescued.append(x['state_id'])
  if u and not v:destroyed.append(x['state_id'])
 return dict(initial=a,final=b,rescued=len(rescued),destroyed=len(destroyed),rescued_ids=rescued,destroyed_ids=destroyed)
