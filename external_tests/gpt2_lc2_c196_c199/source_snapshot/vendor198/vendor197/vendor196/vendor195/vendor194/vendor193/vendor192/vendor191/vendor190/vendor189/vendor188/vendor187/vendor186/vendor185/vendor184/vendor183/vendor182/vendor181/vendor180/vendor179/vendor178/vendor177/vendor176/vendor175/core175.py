"""C175: independent read-only all-coordinate LP and finite-probe arithmetic."""
from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor174'))
import core174 as prior
from core174 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses
control=prior.control
H=(1e-4,3e-4,1e-3)
ETA=1e-3
SAFETY=.0018
ARM_NAMES=prior.ARM_NAMES
PROTOCOL='GPT2_LC2_C175_READONLY_ALL_COORDINATE_AUDIT_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C174',)
TIMEOUT=3600
DECISION='C175_READONLY_ALL_COORDINATE_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'
INCOMPLETE='C175_INCOMPLETE_PARENT_DECISION_NOT_RECLASSIFIED'
SCOPE='Pure NumPy twenty-original C174 archival arithmetic audit. Independently rebuild all84 six-coordinate Jacobian blocks from archived rows, the full1008-row LP, raw-primal direction mapping, scalar-gradient chain arithmetic, exact rational recorded-affine certificates for the direction and all six actual heads, saved physical points, all1848 measurements and original native gates/repeats, and all3024 coordinate-affine residual rows. Derivatives, model outputs and KL are archived inputs, not independently recomputed. No model/Torch/HF/SciPy import, new forward/backward/training/fit/LP/linear solve/Jacobian/SVD, new states, confirmation, excluded input, head/step selection or deployment. Exact arithmetic certifies recorded binary64 affine expressions only; does not certify true derivatives, finite error bounds, solver optimality/infeasibility, curvature, convergence, measurement accuracy, chronology, non-access, actual saved-point consumption, causality or generalisation. C174 decision and original thresholds retained; LC2 remains open.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source174=clean_json((ROOT/'trusted/C174_SOURCE_PINS.json').read_bytes()))
def no_compute():return dict(new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_direction_optimization_iterations=0,new_linear_program_calls=0,linear_systems_solved=0,new_Jacobians=0,new_SVD_calls=0,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,candidate_selected=False,best_checkpoint_selected=False,writer_deployed=False,thresholds_changed=False,parent_decision_reclassified=False,LC2_closed=False,LC2_single_fold_passed=False)
def design():return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),parent_protocol=prior.PROTOCOL,C174_zip_hash_known_before_runtime=True,training_registry=prior.design()['training_registry'],anchor_head_digest='cc06ae379e9c86be4b4f6e4eb79181d088a56b59d8c5b5d0eddbfa1de571236a',arithmetic='Reopen all20 fixed original ZIPs; immutable bundled C174 source; separately reconstruct C174 problem, raw primal clipping/free-mask mapping, scalar AD chain, one direction and six actual-head exact rational certificates; no solver call.',counts=dict(training_states=84,ordered_records=1848,archived_response_Jacobian_blocks=84,archived_response_coordinate_rows=504,archived_scalar_backwards=672,LP_signed_rows=1008,exact_affine_certificates=7,exact_signed_certificate_rows=7056,temporary_probe_heads=6,actual_FP64_point_arrays=672,native_FP32_point_arrays=588,native_endpoint_records=1176,native_repeat_pairs=588,FP64_anchor_repeat_pairs=84,scalar_central_differences=252,signed_scalar_loss_changes=504,response_coordinate_affine_rows=3024),thresholds='C174 native response0.002/KL/read/radius and source-pinned replay/map thresholds unchanged. max(0.0018,current maximum response) is a training linearisation cap, not native acceptance. No new derivative/cap/secant/coordinate-switch acceptance gate.',measurements='All84 exposed training states, six signed fixed probes and both native rounds retained; no best arm or historical checkpoint chosen. Scalar-gradient chain consistency is not independent AD validation.',worker_budget_seconds=TIMEOUT,scope=SCOPE,LC2_closed=False)
def stripped(rows):return [{k:v for k,v in r.items() if k not in ('stage','arm')} for r in rows]
def guard(r):return sum(float(r['objective_parts'][k]) for k in PARTS[1:])
def rational(x):
 x=float(x);need(math.isfinite(x),'Finite binary64 rational input');return Fraction(*x.as_integer_ratio())
def encoded(q):return dict(numerator=str(q.numerator),denominator=str(q.denominator),approximate=float(q))
def chain_check(row,shape):
 r=farray(np.asarray(row['observed'],np.float64),(22,))[:6];J=farray(np.asarray(row['response_coordinate_gradients'],np.float64),(6,)+shape);g=farray(np.asarray(row['response_gradient'],np.float64),shape)
 maximum=float(np.max(np.abs(r)));indices=np.flatnonzero(np.abs(r)==maximum);factor=2*max(maximum/SAFETY-1,0)/SAFETY
 terms=np.stack([J[j]*float(np.sign(r[j])) for j in indices]);expected=factor*np.mean(terms,axis=0);err=float(np.max(abs(expected-g)))
 return dict(maximum_Linf=err,within_declared_chain_tolerance=bool(np.allclose(expected,g,atol=1e-10,rtol=1e-8)),absolute_tolerance=1e-10,relative_tolerance=1e-8,not_an_AD_accuracy_certificate=True)
def problem(rows,p):
 compare([r['state_id'] for r in rows],IDS,'All84 archived Jacobian rows');farray(p,p.shape);need(p.dtype==np.float64 and p.ndim==3 and p.shape[:2]==(3,2) and np.max(abs(p))<=.125,'Legal binary64 head')
 r=np.stack([farray(np.asarray(x['observed'],np.float64),(22,))[:6] for x in rows]);J=np.stack([farray(np.asarray(x['response_coordinate_gradients'],np.float64),(6,)+p.shape).reshape(6,-1) for x in rows]);g=np.stack([farray(np.asarray(x['response_gradient'],np.float64),p.shape).ravel() for x in rows]);G=np.stack([farray(np.asarray(x['guard_gradient'],np.float64),p.shape).ravel() for x in rows]);caps=np.maximum(SAFETY,np.max(abs(r),axis=1));active=np.flatnonzero(np.max(abs(r),axis=1)>SAFETY).astype(np.int64);free=np.abs(p).ravel()<=.125-ETA
 for row in rows:need(chain_check(row,p.shape)['within_declared_chain_tolerance'],'Independent scalar-gradient chain in response units')
 registry=np.array([(i,j,sgn) for i in range(84) for j in range(6) for sgn in (-1,1)],np.int64);A=np.empty((1008,p.size));rhs=np.empty(1008)
 for k,(i,j,sgn) in enumerate(registry):A[k]=(float(sgn)*ETA/SAFETY)*(J[i,j]*free);rhs[k]=(caps[i]-int(sgn)*r[i,j])/SAFETY
 need(np.isfinite(A).all() and np.isfinite(rhs).all(),'Finite full LP');return dict(anchor=p.copy(),response=r,J=J,response_loss_gradients=g,guards=G,caps=caps,free=free,active=active,A=A,rhs=rhs,registry=registry)
def certificate(pr,d,head=None,budget=lambda:None):
 p=pr['anchor'];farray(d,p.shape);need(d.dtype==np.float64 and np.max(abs(d))<=1 and not np.any(d.ravel()[~pr['free']]),'Recorded bounded free-mask direction')
 if head is None:delta=[rational(ETA)*rational(x) for x in d.ravel()];mode='exact eta times recorded direction'
 else:
  farray(head,p.shape);need(head.dtype==np.float64 and np.max(abs(head))<=.125,'Actual binary64 head');delta=[rational(y)-rational(x) for x,y in zip(p.ravel(),head.ravel())];mode='exact difference of actual binary64 head endpoints'
 if not len(pr['active']):need(not np.any(d),'Empty active zero direction only')
 rows=[];minimum=None
 for i,sid in enumerate(IDS):
  budget()
  for j in range(6):
   shift=sum((rational(v)*step for v,step in zip(pr['J'][i,j],delta)),Fraction(0));prediction=rational(pr['response'][i,j])+shift
   for sgn in (-1,1):
    slack=(rational(pr['caps'][i])-sgn*prediction)/rational(SAFETY);rows.append(dict(state_id=sid,coordinate=j,sign=sgn,recorded_linear_signed_response=encoded(sgn*prediction),normalized_recorded_cap_slack=encoded(slack)));minimum=slack if minimum is None else min(minimum,slack)
 return dict(signed_rows=1008,head_difference_mode=mode,minimum_normalized_recorded_cap_slack=encoded(minimum),all_recorded_affine_caps_strictly_satisfied=bool(minimum>0),strict_recorded_active_margin=bool(len(pr['active']) and minimum>0),active_state_ids=[IDS[i] for i in pr['active']],all1008_signed_rows=rows,scope='Exact recorded binary64 response/Jacobian affine arithmetic only; not a derivative, finite endpoint, optimality/infeasibility, curvature or generalisation certificate.')
def panel(p,d):
 farray(d,p.shape);need(p.dtype==d.dtype==np.float64 and np.max(abs(d))<=1,'Bounded binary64 direction');need(not np.any(d[np.abs(p)>.125-ETA]),'All six heads share safe free mask')
 arms=[dict(name=('minus_' if s<0 else 'plus_')+'s'+str(i+1),alpha=float(s*h)) for i,h in enumerate(H) for s in (-1,1)];heads={a['name']:np.add(p,np.multiply(a['alpha'],d)) for a in arms}
 for n,q in heads.items():need(np.isfinite(q).all() and np.max(abs(q))<=.125,'Legal actual signed head/'+n)
 return heads,dict(positive_steps=list(H),arms=arms,anchor_head_digest=parameter_hash(p),direction_digest=array_digest(d),head_digests={n:parameter_hash(q) for n,q in heads.items()},zero_direction_noop=not bool(np.any(d)),head_projection_performed=False,step_shrinking_performed=False,postwrite_selection=False)
