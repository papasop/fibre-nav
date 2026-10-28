"""Read-only full C170 training trajectory, recorded witnesses and native gates."""
from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor170'))
import core170 as prior
from core170 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses
def exact(a,b,msg='C171 exact tensor comparison'):return prior.exact(a,b,msg)
PROTOCOL='GPT2_LC2_C171_READONLY_LP_TRAIN_AUDIT_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C170',)
TIMEOUT=1800
DECISION='C171_READONLY_LP_TRAIN_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'
INCOMPLETE='C171_INCOMPLETE_PARENT_DECISION_NOT_RECLASSIFIED'
SCOPE='Pure NumPy sixteen-original read-only archive, tensor, update, rational-contraction and native-gate audit. No model/Torch/HF/SciPy import, model forward, backward, training, fit, new LP, linear solve, Jacobian, SVD, new preparation, confirmation or excluded input. Reconstructs recorded binary64 gradients, original solver vectors and finite observations only; does not independently evaluate derivatives, logits or KL. Exact contractions do not certify true derivatives, solver optimality, infeasibility, curvature bounds, convergence, causal explanations, chronology, non-access or generalisation. All84 states, eight updates, fixed final step8, failures and both native rounds retained, without checkpoint/step selection or deployment. Original C170 decision retained; LC2 remains open.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source170=clean_json((ROOT/'trusted/C170_SOURCE_PINS.json').read_bytes()))
def design():return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),training_registry=prior.design()['training_registry'],parent_protocol=prior.PROTOCOL,C170_original_hash_known_before_runtime=True,all84_exposed_training=True,arithmetic='Bind all sixteen originals. Recompute all1092 ordered C170 records,672 gradient records,eight current active masks/normalisations/LP direction-vector bindings,eight exact recorded direction witnesses,eight fixed1e-4 updates,756 actual FP64 points,168 native proposal arrays,336 native endpoint gates and168 native repeat pairs. No solver/model import or new optimization.',diagnostics='All756 per-state frame records and672 consecutive transitions: response and guard parts, original/safety slack, actual coefficient-update dot recorded gradient, planned-direction versus actual-update arithmetic, finite-minus-linear residuals, active state/coordinate/sign changes, all9 worst-response state sets, all8 margin/bottleneck changes and336 FP64/native comparisons. Report all84 plus fixed initial failure IDs, without alternative candidate selection or extrapolated closure time.',counts=dict(ordered_records=1092,gradient_records=672,training_updates=8,actual_FP64_point_arrays=756,native_FP32_point_arrays=168,native_endpoint_records=336,native_repeat_pairs=168,frame_records=756,transition_records=672),thresholds='Unchanged source-pinned native and same-anchor replay/map tolerances. No new diagnostic pass threshold for loss trends, secants or gradient residuals.',worker_budget_seconds=TIMEOUT,scope=SCOPE,LC2_closed=False)
def no_compute():return dict(new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_direction_optimization_iterations=0,new_linear_program_calls=0,linear_systems_solved=0,new_Jacobians=0,new_SVD_calls=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,best_checkpoint_selected=False,writer_deployed=False,thresholds_changed=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False)
def encoded(q):return dict(numerator=str(q.numerator),denominator=str(q.denominator),approximate=float(q))
def rational(v):
 x=float(v);need(math.isfinite(x),'Finite exact arithmetic scalar');a,b=x.as_integer_ratio();return Fraction(a,b)
def contraction(row,delta):return sum((rational(x)*rational(y) for x,y in zip(np.asarray(row).ravel(),np.asarray(delta).ravel())),Fraction(0))
def frame_objective(rows):
 compare([r['state_id'] for r in rows],IDS);rv=np.array([float(r['objective_parts']['response_loss']) for r in rows]);gv=np.array([sum(float(r['objective_parts'][k]) for k in PARTS[1:]) for r in rows]);return dict(n=84,active_response_states=int(np.count_nonzero(rv>0)),maximum_response_loss=float(np.max(rv)),mean_response_loss=float(np.mean(rv)),mean_guard_loss=float(np.mean(gv)),original_hardmax_objective=float(.9*np.max(rv)+.1*np.mean(rv)+np.mean(gv)),descriptive_objective_not_LP_objective=True)
def recorded_problem(rows,p):
 compare([r['state_id'] for r in rows],IDS,'Canonical84 gradient order');farray(p,p.shape);need(p.dtype==np.float64 and p.ndim==3 and p.shape[:2]==(3,2) and np.max(abs(p))<=.125,'Legal original coefficient head');raw=np.vstack([farray(np.asarray(r['response_gradient'],np.float64),p.shape).reshape(1,-1) for r in rows]);guards=np.vstack([farray(np.asarray(r['guard_gradient'],np.float64),p.shape).reshape(1,-1) for r in rows]);values=np.array([float(r['objective_parts']['response_loss']) for r in rows]);need(np.isfinite(values).all() and np.all(values>=0),'Finite losses');free=(np.abs(p).reshape(-1)<=.125-1e-4);active=np.flatnonzero(values>0).astype(np.int64);projected=raw*free[None,:];norms=np.linalg.norm(projected,axis=1);need(np.isfinite(norms).all(),'Finite norms');A=np.zeros((len(active),p.size),np.float64)
 for j,i in enumerate(active):
  if norms[i]>0:A[j]=projected[i]/norms[i]
 return dict(anchor=p.copy(),raw=raw,guards=guards,free=free,active=active,norms=norms,A=A,loss=values)
def certificate(pr,d):
 need(d.dtype==np.float64 and d.shape==(pr['anchor'].size,) and np.isfinite(d).all(),'Saved finite binary64 direction');need(np.max(abs(d))<=1 and not np.any(d[~pr['free']]),'Original fixed direction subspace/bounds');rows=[];margin=[];active=set(int(i) for i in pr['active'])
 for i,sid in enumerate(IDS):
  r=contraction(pr['raw'][i],d);g=contraction(pr['guards'][i],d);rows.append(dict(state_id=sid,active=i in active,raw_response_dot=encoded(r),guard_dot=encoded(g),strict_raw_response_descent=r<0))
  if i in active:margin.append(-r/rational(pr['norms'][i]) if pr['norms'][i]>0 else Fraction(0))
 if active:lower=min(margin);strict=bool(lower>0 and all(r['strict_raw_response_descent'] for r in rows if r['active']))
 else:need(not np.any(d),'No arbitrary empty-active direction');lower=None;strict=False
 return dict(active_count=len(active),active_state_ids=[IDS[i] for i in pr['active']],strict_recorded_common_descent=strict,normalized_margin_lower=encoded(lower) if lower is not None else None,all84_exact_contractions=rows,scope='Exact arithmetic on current recorded binary64 gradients and direction only; no derivative truth, optimality, infeasibility or finite-write certificate.')
