"""Eight fixed offline relinearized LP coefficient updates; final step8 only."""
from pathlib import Path
from fractions import Fraction as Q
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor169'))
import core169 as prior
from core169 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses
source168=prior.prior
control=source168.old.parent
seal_head=control.seal_head
native_summary=control.native_summary
paired=control.paired
PROTOCOL='GPT2_LC2_C170_RELINEARISED_LP_TRAIN_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C169',)
STEPS=8;STEP=1e-4;TIMEOUT=10800
RECORDS=1092;REQUESTED_FORWARDS=2187;FUNCTIONAL_FORWARDS=1514;BACKWARDS=1344
SOLVER_OPTIONS=dict(presolve=True,time_limit=60.,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,threads=1,random_seed=0)
SCOPE='New offline eight-step response-only common-descent LP fit from exact C163 fixed-final step8. Fresh full84 FP64 response and guard scalar gradients at each current head; one LP per active batch, fixed1e-4 coefficient step, exact recorded-gradient sign check before each update, final step8 only. Guard parts/gradients are measured and archived but are not constraints of the response-only LP; actual native gates decide final training qualification. No online correction-free or autonomous-control claim for this fitting procedure. All84 remain exposed training; no fresh confirmation, excluded input, checkpoint/step selection, deployment, convergence, global feasibility or LC2 closure claim. All earlier decisions retained.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source169=clean_json((ROOT/'trusted/C169_SOURCE_PINS.json').read_bytes()))
def design():
 return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),training_registry=source168.design()['training_registry'],anchor='Exact C163 original fixed-final local step8; never initialize from C168 probes or a selected historical head.',training=dict(steps=STEPS,step=STEP,parameter_shape=[3,2,36],free_mask='abs(current coefficient)<=0.125-1e-4; excluded coordinates have direction0; original two-sided coefficient box unchanged',active_rule='Every current full84 response_loss>0 row, including zero-gradient rows; no failed-state-only subset',LP='max t subject to A_i d+t<=0 for all active rows; A_i is current projected binary64 response gradient divided by its binary64 L2 norm (zero row if norm0); |d_j|<=1 on free coordinates, excluded coordinates0.',solver=dict(backend='scipy.optimize.linprog',method='highs-ds',options=SOLVER_OPTIONS,max_calls=STEPS,one_call_per_active_batch=True,no_dual_LP=True),witness='Use exact rational contractions of every binary64 raw response/guard gradient with the bounded direction. Every active raw dot must be strictly negative before update; report min normalized exact lower bound. This certifies recorded inputs only, not true derivatives, solver optimality, infeasibility or finite acceptance.',update='current + fixed1e-4*direction in binary64. Solver vector clipped only to direction bounds and fixed subspace before exact verification; no head clipping, step search, halving, best checkpoint or line search. Empty-active batch has registered zero direction/no LP; unresolved/nonpositive witness stops incomplete and packages, without fallback.',guards='Measure/record every current read, KL, radius and regularizer loss and its summed gradient. These do not constrain the response-only LP; no guard-preservation guarantee. Original native hard gates retained at fixed final evaluation.',fresh_optimizer_state='No Adam/momentum state; no resumed optimizer'),frozen='Original model/revision, LoRA-A, full84 routes, V, prepared/raw coordinates, cached C155 response/logits references and all native thresholds. Only3x2x36 head coefficients updated offline.',references=source168.design()['references'],acceptance_budgets=source168.design()['acceptance_budgets'],measurement_order='Native initial84 round0; eight full84 FP64 gradient batches, update after each complete batch; final84 FP64 no-grad; native final round0; native initial round1; native final round1.',sealing='Original constants, initial head, all1092 requests before model initialization. Each current head and84 actual FP64 points before its gradient batch; final step8 before final measurement. Store756 actual FP64 arrays and168 physical FP32 arrays.',replay='Initial native two rounds versus original C168 anchor with unchanged native replay/gate checks; initial FP64 versus original C168 anchor within2e-9 for observed and KL, before first LP/update.',counts=dict(records=RECORDS,model_forwards=REQUESTED_FORWARDS,functional_FP64_forwards=FUNCTIONAL_FORWARDS,scalar_backwards=BACKWARDS,actual_FP64_point_arrays=756,native_point_arrays=168,native_endpoint_records=336,native_repeat_pairs=168,coefficient_updates=STEPS,max_LP_calls=STEPS),worker_budget_seconds=TIMEOUT,scope=SCOPE,LC2_closed=False)
def requests():
 r=[dict(stage='native_initial0',step=0,state_id=s) for s in IDS]
 r += [dict(stage='training',step=t,state_id=s) for t in range(STEPS) for s in IDS]
 r += [dict(stage='final_FP64',step=STEPS,state_id=s) for s in IDS]
 for name,t in [('native_final0',STEPS),('native_initial1',0),('native_final1',STEPS)]:r += [dict(stage=name,step=t,state_id=s) for s in IDS]
 compare(len(r),RECORDS);return r
def objective(rows):
 compare([r['state_id'] for r in rows],IDS);R=np.array([r['objective_parts']['response_loss'] for r in rows]);G=np.array([sum(r['objective_parts'][k] for k in PARTS[1:]) for r in rows]);return dict(n=84,active_response_states=int(np.count_nonzero(R>0)),maximum_response_loss=float(max(R)),mean_response_loss=float(np.mean(R)),mean_guard_loss=float(np.mean(G)),original_hardmax_objective=float(.9*max(R)+.1*np.mean(R)+np.mean(G)),descriptive_objective_not_LP_objective=True)
def problem(rows,p):
 compare([r['state_id'] for r in rows],IDS);farray(p,p.shape);need(p.dtype==np.float64 and p.ndim==3 and p.shape[:2]==(3,2) and max(abs(p.ravel()))<=.125,'Exact legal current coefficient head');free=(abs(p)<=.125-STEP).ravel();R=[];G=[];active=[];L=[]
 for i,r in enumerate(rows):
  value=float(r['objective_parts']['response_loss']);need(math.isfinite(value) and value>=0,'Finite current response loss');L.append(value);R.append(farray(np.asarray(r['response_gradient'],np.float64),p.shape).ravel());G.append(farray(np.asarray(r['guard_gradient'],np.float64),p.shape).ravel())
  if value>0:active.append(i)
 raw=np.array(R);guards=np.array(G);norms=np.linalg.norm(raw*free,axis=1);need(np.isfinite(norms).all(),'Finite projected row norms');A=np.array([raw[i]*free/norms[i] if norms[i]>0 else np.zeros(p.size) for i in active],np.float64).reshape(len(active),p.size)
 return dict(anchor=p.copy(),raw=raw,guards=guards,free=free,active=np.array(active,np.int64),norms=norms,A=A,loss=np.array(L))
def enc(q):return dict(numerator=str(q.numerator),denominator=str(q.denominator),approximate=float(q))
def verify_direction(pr,d):
 m=pr['anchor'].size
 for name in ('raw','guards'):farray(pr[name],(84,m));need(pr[name].dtype==np.float64,'Binary64 current gradient matrix')
 farray(pr['norms'],(84,));need(pr['norms'].dtype==np.float64 and np.all(pr['norms']>=0),'Finite nonnegative binary64 norms');need(pr['free'].dtype==np.bool_ and pr['free'].shape==(m,),'Boolean current subspace');need(pr['active'].dtype.kind in 'iu' and pr['active'].ndim==1 and pr['active'].tolist()==sorted(set(pr['active'].tolist())) and all(0<=int(i)<84 for i in pr['active']),'Canonical active registry')
 d=np.asarray(d);need(d.dtype==np.float64 and d.shape==(pr['anchor'].size,) and np.isfinite(d).all(),'Finite binary64 direction');need(max(abs(d),default=0)<=1 and np.all(d[~pr['free']]==0),'Direction bounds/subspace');D=[Q.from_float(float(v)) for v in d];rows=[];normalized=[]
 for i,sid in enumerate(IDS):
  r=sum((Q.from_float(float(x))*y for x,y in zip(pr['raw'][i],D)),Q(0));g=sum((Q.from_float(float(x))*y for x,y in zip(pr['guards'][i],D)),Q(0));is_active=i in pr['active'];rows.append(dict(state_id=sid,active=bool(is_active),raw_response_dot=enc(r),guard_dot=enc(g),strict_raw_response_descent=bool(r<0)))
  if is_active:
   norm=Q.from_float(float(pr['norms'][i]));normalized.append(r/norm if norm>0 else Q(0))
 if not len(pr['active']):need(not np.any(d),'Empty-active direction is zero');lower=None;strict=False
 else:lower=-max(normalized);strict=bool(lower>0 and all(r['strict_raw_response_descent'] for r in rows if r['active']))
 return dict(active_count=len(pr['active']),active_state_ids=[IDS[i] for i in pr['active']],strict_recorded_common_descent=strict,normalized_margin_lower=enc(lower) if lower is not None else None,all84_exact_contractions=rows,scope='Exact arithmetic on current recorded binary64 gradients and direction only; no derivative truth, optimality, infeasibility or finite-write certificate.')
def update(p,d):
 need(p.dtype==d.dtype==np.float64 and p.shape==d.shape and np.isfinite(d).all() and np.max(abs(d))<=1,'Fixed bounded FP64 update');new=p+STEP*d;need(np.isfinite(new).all() and np.max(abs(new))<=.125,'Actual unprojected head must fit original box');return new,dict(step_size=STEP,update_Linf=float(max(abs((new-p).ravel()))),head_before=parameter_hash(p),head_after=parameter_hash(new),head_projection_performed=False)
