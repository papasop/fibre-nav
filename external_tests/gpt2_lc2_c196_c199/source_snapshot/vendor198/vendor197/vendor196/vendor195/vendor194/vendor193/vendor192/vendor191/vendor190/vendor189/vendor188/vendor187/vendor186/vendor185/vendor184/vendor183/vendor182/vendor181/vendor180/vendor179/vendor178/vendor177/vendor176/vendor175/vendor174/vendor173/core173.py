"""Read-only C172 signed finite-step and response-coordinate audit."""
from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor172'))
import core172 as prior
from core172 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses
control=prior.control
H=prior.H
ARM_NAMES=prior.ARM_NAMES
PROTOCOL='GPT2_LC2_C173_READONLY_FINITE_STEP_COORDINATE_AUDIT_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C172',)
TIMEOUT=1800
DECISION='C173_READONLY_FINITE_STEP_COORDINATE_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'
INCOMPLETE='C173_INCOMPLETE_PARENT_DECISION_NOT_RECLASSIFIED'
SCOPE='Pure NumPy eighteen-original read-only C172 archive, saved-point, gradient-row, one-LP-vector, exact recorded-contraction, six-head and native-gate arithmetic audit. Adds all84/all6 paired response-coordinate and max-coordinate/sign-switch descriptions from archived observables only. No model/Torch/HF/SciPy import, forward, backward, training, head fit, new direction/LP/linear solve, Jacobian, SVD, new preparation, confirmation or excluded input. Does not certify true gradients, model measurements, KL accuracy, solver optimality/infeasibility, chronology, non-access, actual consumption of saved points, statistical/physical causality, curvature or convergence. A max-coordinate switch or finite secant does not supply an inactive-coordinate gradient or explain a loss reversal. No head/step selection, deployment, new acceptance threshold or LC2 claim; C172 decision retained.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source172=clean_json((ROOT/'trusted/C172_SOURCE_PINS.json').read_bytes()))
def no_compute():return dict(new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_direction_optimization_iterations=0,new_linear_program_calls=0,linear_systems_solved=0,new_Jacobians=0,new_SVD_calls=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,best_checkpoint_selected=False,writer_deployed=False,thresholds_changed=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False)
def design():return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),parent_protocol=prior.PROTOCOL,C172_zip_hash_known_before_runtime=True,training_registry=prior.design()['training_registry'],arithmetic='Reopen all18 original ZIPs and trusted delivered source. Reconstruct C172 all1848 requests,84 fresh gradient records,one current free-mask/normalisation/problem,original primal-vector mapping and exact recorded witness,six fixed signed heads,672 actual FP64 and588 native points,1176 native gates,588 native repeat pairs,84 FP64 repeats,252 scalar central differences and504 signed scalar-loss changes. Original LP internal solve count remains null; new LP/solve counts0.',coordinates='All504 state/arm pairs and3024 response-coordinate transitions,1512 coordinate central secants,all7 frame worst-state sets and1176 native-vs-FP64 response-only comparisons. Exact equality defines archived maxima/ties; coordinate indexing is zero-based. Report signs,hard/safety slacks,anchor-max-coordinate envelope and its gap to actual endpoint maximum. Coordinate switches and observed secants are descriptive, not derivatives, causal or curvature evidence.',counts=dict(original_inputs=18,ordered_records=1848,gradient_records=84,temporary_probe_heads_audited=6,actual_FP64_point_arrays=672,native_FP32_point_arrays=588,native_endpoint_records=1176,native_repeat_pairs=588,FP64_anchor_repeat_pairs=84,scalar_central_differences=252,signed_scalar_loss_changes=504,paired_state_rows=504,response_coordinate_transition_rows=3024,response_coordinate_central_secant_rows=1512,native_vs_FP64_response_only_rows=1176),thresholds='All source-pinned original response/KL/read/radius/map/replay thresholds retained. No switch,secant or loss-trend pass gate added.',worker_budget_seconds=TIMEOUT,scope=SCOPE,LC2_closed=False)
def guard(row):return sum(float(row['objective_parts'][k]) for k in PARTS[1:])
def stripped(rows):return [{k:v for k,v in r.items() if k not in ('stage','arm')} for r in rows]
def rational(x):
 v=float(x);need(math.isfinite(v),'Finite binary64 rational input');return Fraction(*v.as_integer_ratio())
def encoded(q):return dict(numerator=str(q.numerator),denominator=str(q.denominator),approximate=float(q))
def problem(rows,p):
 compare([r['state_id'] for r in rows],IDS);farray(p,p.shape);need(p.dtype==np.float64 and p.ndim==3 and p.shape[:2]==(3,2) and np.max(np.abs(p))<=.125,'Legal complete binary64 coefficient head');raw=np.stack([farray(np.asarray(r['response_gradient'],np.float64),p.shape).ravel() for r in rows]);guards=np.stack([farray(np.asarray(r['guard_gradient'],np.float64),p.shape).ravel() for r in rows]);values=np.array([float(r['objective_parts']['response_loss']) for r in rows]);need(np.isfinite(values).all() and np.all(values>=0),'Finite response losses');free=np.abs(p).ravel()<=.125-1e-3;active=np.flatnonzero(values>0).astype(np.int64);norms=np.linalg.norm(raw*free,axis=1);need(np.isfinite(norms).all(),'Finite projected norms');A=np.zeros((len(active),p.size),np.float64)
 for j,i in enumerate(active):
  if norms[i]>0:A[j]=(raw[i]*free)/norms[i]
 return dict(anchor=p.copy(),raw=raw,guards=guards,free=free,active=active,norms=norms,A=A,loss=values)
def panel(p,d):
 farray(d,p.shape);need(d.dtype==p.dtype==np.float64 and np.max(np.abs(d))<=1,'Finite bounded binary64 direction');need(not np.any(d[np.abs(p)>.125-1e-3]),'Full six-arm free subspace');arms=[dict(name=('minus_' if sign<0 else 'plus_')+'s'+str(i+1),alpha=float(sign*h)) for i,h in enumerate((1e-4,3e-4,1e-3)) for sign in (-1,1)];heads={a['name']:np.add(p,np.multiply(a['alpha'],d)) for a in arms}
 for n,q in heads.items():need(np.isfinite(q).all() and np.max(np.abs(q))<=.125,'Actual fixed-panel head legal/'+n)
 re=dict(positive_steps=[1e-4,3e-4,1e-3],arms=arms,anchor_head_digest=parameter_hash(p),direction_digest=array_digest(d),head_digests={n:parameter_hash(q) for n,q in heads.items()},zero_direction_noop=not bool(np.any(d)),head_projection_performed=False,step_shrinking_performed=False,postwrite_selection=False);return heads,re
def certificate(pr,d):
 from core171 import certificate as independent_certificate
 return independent_certificate(pr,d.ravel())
