"""C167: bounded LP on recorded gradients; exact rational witness verification."""
from pathlib import Path
import sys,json,hashlib,time,contextlib,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor166'))
import core166 as prior
from core166 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N
PROTOCOL='GPT2_LC2_C167_RECORDED_GRADIENT_FEASIBILITY_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C166',)
TIMEOUT=1800
SOLVER_TIME=60.
SOLVER_OPTIONS=dict(presolve=True,time_limit=SOLVER_TIME,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,threads=1,random_seed=0)
SCOPE='Offline two-LP direction diagnostic on exact binary64 archived scalar gradients. Rational verification certifies only the stated linear problem on recorded gradients in the original C165 two-sided coefficient subspace. It does not certify true model derivatives, finite-step endpoint acceptance, guard preservation, excluded-input generalisation, global feasibility or infeasibility, chronology or LC2 closure. No model, backward, new head, finite probe, training, confirmation or excluded input; original C165/C166 decisions retained.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source166=clean_json((ROOT/'trusted/C166_SOURCE_PINS.json').read_bytes()))
def design():return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),training_registry=prior.registry(),anchor='Exact C163 fixed-final head already bound by independently rerun C166 arithmetic.',active_rule='All84 retained; every original C165 R>0 row active (safety0.0018), including zero gradients. No failed-state-only subset.',subspace='Original C165 symmetric-probe free mask abs(anchor)<=0.125-1e-5; 216 coordinates; no new one-sided or full-parameter claim.',linear_problem='max t subject to (raw_projected_gradient_i / stored_positive_L2_norm_i) dot d + t <= 0 for all active i; ||d||infinity<=1, excluded coordinates zero. Zero gradients are zero rows. Solver uses binary64 normalization; independent certificate uses exact rational division of original binary64 gradient by original binary64 norm.',dual='min sum(z) subject to -z<=A.T lambda<=z, lambda>=0,sum(lambda)=1,z>=0. One separate fixed dual LP; no head or checkpoint search.',solver=dict(backend='scipy.optimize.linprog',method='highs-ds',calls_if_active=2,options=SOLVER_OPTIONS,handling='Final primal direction clipped to declared [-1,1] bounds only; no head clipping. Final dual weights clipped nonnegative then normalized exactly as rational numbers. Both raw solver results retained. No fallback/restart/solver selection.'),certificate='Fractions from exact binary64 values. Verify all84 raw response/guard dot products; exact lower=min(-B_i dot d), exact upper=sum_j abs(sum_i lambda_i*B_ij). These bound the optimal common margin for B=raw_i/exact_norm_i. Positive lower proves strict common descent for recorded active gradients; exactly zero dual upper proves no strict common descent in this subspace; otherwise unresolved. Interval width is reported, no near-zero infeasibility threshold.',outputs='All84 gradients, both LPs, exact normalized problem bounds and raw row contractions, one direction witness only, matrix/anchor/free-mask hashes. No parameter head, physical proposal or model evaluation.',worker_budget_seconds=TIMEOUT,scope=SCOPE,LC2_closed=False)

def problem(rows,p):
 compare([r['state_id'] for r in rows],IDS,'Exact ordered full84 gradient registry')
 farray(p,p.shape);need(p.dtype==np.float64 and p.ndim==3 and p.shape[:2]==(3,2) and np.max(abs(p))<=.125,'Legal binary64 anchor')
 free=(abs(p)<=.125-max(prior.H)).ravel();raw=[];guards=[];active=[];loss=[]
 for i,r in enumerate(rows):
  v=float(r['objective_parts']['response_loss']);need(math.isfinite(v) and v>=0,'Finite response loss');loss.append(v)
  raw.append(farray(np.asarray(r['response_gradient'],dtype=np.float64),p.shape).ravel());guards.append(farray(np.asarray(r['guard_gradient'],dtype=np.float64),p.shape).ravel())
  if v>0:active.append(i)
 raw=np.asarray(raw);guards=np.asarray(guards);projected=raw*free;norms=np.linalg.norm(projected,axis=1);need(np.isfinite(norms).all(),'Finite norm')
 A=np.array([projected[i]/norms[i] if norms[i]>0 else np.zeros(p.size) for i in active],dtype=np.float64).reshape(len(active),p.size)
 need(np.isfinite(A).all(),'Finite normalized solver matrix')
 return dict(ids=list(IDS),anchor=p.copy(),free=free,raw=raw,guards=guards,loss=np.array(loss),active=np.array(active,dtype=np.int64),norms=norms,A=A)
