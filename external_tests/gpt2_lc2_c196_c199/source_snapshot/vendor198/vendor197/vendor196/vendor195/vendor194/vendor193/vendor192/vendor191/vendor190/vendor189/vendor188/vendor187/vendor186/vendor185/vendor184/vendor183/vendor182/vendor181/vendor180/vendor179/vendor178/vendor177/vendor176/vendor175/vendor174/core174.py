"""Offline all-coordinate response-budget LP and fixed signed probe contract."""
from pathlib import Path
from fractions import Fraction
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor173'))
import core173 as prior
from core173 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses,rational,encoded
source172=prior.prior
control=source172.control
seal_head=control.seal_head
native_summary=control.native_summary
paired=control.paired
H=(1e-4,3e-4,1e-3)
ETA=1e-3
SAFETY=.0018
ARM_NAMES=source172.ARM_NAMES
PROTOCOL='GPT2_LC2_C174_ALL_COORDINATE_DIRECTION_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C173',)
TIMEOUT=10800
RECORDS=1848
FORWARDS=3699
FUNCTIONAL=1346
BACKWARDS=672
SOLVER_OPTIONS=dict(control.SOLVER_OPTIONS)
SCOPE='Offline training-only all-coordinate direction diagnostic at the original C170 fixed final step8, independently rebound through C173. One fresh full84 FP64 batch: six response-coordinate gradients plus response-loss and guard-loss gradients per state, 84 coefficient-response Jacobian blocks (504x216 response rows in production),672 scalar backwards. One LP uses both signs of all504 response coordinates with frozen per-state cap max(0.0018,current response Linf); cap is a training linearisation constraint, not a changed native acceptance budget. Guard gradients recorded but not constrained. Six fixed signed probes sealed before their measurements; all84/all6/two native rounds retained. No optimizer/training update, head or step selection, deployment, fresh preparation, confirmation or excluded input. Exact checks certify recorded binary64 affine predictions only, not true derivatives, finite acceptance, solver optimality/infeasibility, curvature, convergence, causality or LC2. Derivative construction is offline development cost and is incompatible with an online correction-free LC1 execution claim.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source173=clean_json((ROOT/'trusted/C173_SOURCE_PINS.json').read_bytes()))
def design():
 return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),training_registry=source172.design()['training_registry'],anchor='Exact original C170 FIXED_FINAL_HEAD_STEP8.npz; never a C172 probe or selected historical head.',anchor_head_digest=source172.design()['initial_head_digest'],gradients='At each of all84 anchor states: AD of all six signed response coordinates, response scalar loss, and summed guard scalar loss with respect to the3x2x36 coefficient head. Model/backbone/LoRA-A/routes/V/data/references frozen. This explicitly constructs84 response Jacobian blocks; no full-model-parameter Jacobian.',scalar_gradient_chain_check=dict(absolute_tolerance=1e-10,relative_tolerance=1e-8,scope='Coordinate AD versus scalar response-loss AD chain in original response units; not an AD accuracy certificate.'),linear_problem=dict(eta=ETA,response_scale=SAFETY,cap='b_i=max(0.0018,max_j abs(r_ij)); r is the fresh anchor signed response drift.',constraints='For every i=0..83,j=0..5,sign in(-1,+1): sign*(r_ij+eta*J_ij*d)/0.0018 + t <= b_i/0.0018. Maximise t, t>=0; |d_j|<=1 on abs(p_j)<=0.125-0.001, other direction coordinates0. All1008 signed rows retained without active/near-threshold selection or per-row L2 normalisation.',solver=dict(backend='scipy.optimize.linprog',method='highs-ds',options=SOLVER_OPTIONS,max_calls=1),witness='Exact rational slack of all1008 signed affine constraints at eta*d and at each actual binary64 probe-head displacement. Nonempty active set requires positive minimum recorded slack for every positive probe before measurements; otherwise stop incomplete without retry/fallback. Empty active set uses zero direction/no LP and makes no strict-descent claim.',guard='Read/KL/radius/regularizer and guard gradients are recorded; LP constrains only response coordinates. Original native gates decide finite endpoints.'),positive_steps=list(H),arms=source172.design()['arms'],head_rule='Same six p+alpha*d binary64 heads as C172; no head projection, shrinking, line search, optimizer, best checkpoint or deployment.',sealing='Constants,anchor,all1848 requests before model; actual anchor/repeat points before anchor AD; all6 NumPy/actual FP64 probe heads and exact affine certificates after the gradient/LP but before any probe measurement. Gradient batch explicitly precedes probe sealing.',measurement_order=source172.design()['measurement_order'],replay='Initial native two rounds and anchor FP64 against original C172 anchor same-round records under original tolerances; no baseline recentering.',references=source172.design()['references'],acceptance_budgets=source172.design()['acceptance_budgets'],counts=dict(records=RECORDS,model_forwards=FORWARDS,functional_FP64_forwards=FUNCTIONAL,scalar_backwards=BACKWARDS,response_coordinate_backwards=504,response_loss_backwards=84,guard_loss_backwards=84,coefficient_response_Jacobian_blocks=84,coefficient_response_Jacobian_rows=504,production_coefficient_columns=216,max_LP_calls=1,LP_signed_rows=1008,temporary_probe_heads=6,actual_FP64_point_arrays=672,native_FP32_point_arrays=588,native_endpoint_records=1176,native_repeat_pairs=588,FP64_anchor_repeat_pairs=84,new_training_steps=0),worker_budget_seconds=TIMEOUT,scope=SCOPE,LC2_closed=False)
def requests():return source172.requests()
def loss_chain_check(row,shape):
 r=farray(np.asarray(row['observed'],np.float64),(22,))[:6];J=farray(np.asarray(row['response_coordinate_gradients'],np.float64),(6,)+shape);g=farray(np.asarray(row['response_gradient'],np.float64),shape);m=float(np.max(np.abs(r)));top=np.abs(r)==m;factor=2*max(m/SAFETY-1,0)/SAFETY;want=factor*np.mean(J[top]*np.sign(r[top]).reshape((-1,)+(1,)*len(shape)),axis=0);err=float(np.max(abs(want-g)));return dict(maximum_Linf=err,within_declared_chain_tolerance=bool(np.allclose(want,g,atol=1e-10,rtol=1e-8)),absolute_tolerance=1e-10,relative_tolerance=1e-8,not_an_AD_accuracy_certificate=True)
def problem(rows,p):
 compare([r['state_id'] for r in rows],IDS,'All84 Jacobian rows');farray(p,p.shape);need(p.dtype==np.float64 and p.ndim==3 and p.shape[:2]==(3,2) and np.max(np.abs(p))<=.125,'Legal binary64 coefficient head');r=np.stack([farray(np.asarray(x['observed'],np.float64),(22,))[:6] for x in rows]);J=np.stack([farray(np.asarray(x['response_coordinate_gradients'],np.float64),(6,)+p.shape).reshape(6,-1) for x in rows]);g=np.stack([farray(np.asarray(x['response_gradient'],np.float64),p.shape).ravel() for x in rows]);G=np.stack([farray(np.asarray(x['guard_gradient'],np.float64),p.shape).ravel() for x in rows]);caps=np.maximum(SAFETY,np.max(np.abs(r),axis=1));active=np.flatnonzero(np.max(np.abs(r),axis=1)>SAFETY).astype(np.int64);free=(np.abs(p).ravel()<=.125-ETA);registry=[];A=[];rhs=[]
 for row in rows:need(loss_chain_check(row,p.shape)['within_declared_chain_tolerance'],'Recorded scalar loss AD must match six-coordinate chain in declared units')
 for i,sid in enumerate(IDS):
  for j in range(6):
   for sign in (-1,1):registry.append((i,j,sign));A.append((sign*ETA/SAFETY)*(J[i,j]*free));rhs.append((caps[i]-sign*r[i,j])/SAFETY)
 A=np.asarray(A,np.float64);rhs=np.asarray(rhs,np.float64);need(np.isfinite(A).all() and np.isfinite(rhs).all(),'Finite complete coordinate LP');return dict(anchor=p.copy(),response=r,J=J,response_loss_gradients=g,guards=G,caps=caps,free=free,active=active,A=A,rhs=rhs,registry=np.asarray(registry,np.int64))
def exact_certificate(pr,d,head=None):
 p=pr['anchor'];farray(d,p.shape);need(d.dtype==np.float64 and np.max(np.abs(d))<=1 and not np.any(d.ravel()[~pr['free']]),'Fixed bounded direction/free mask')
 if head is None:delta=[rational(ETA)*rational(v) for v in d.ravel()];mode='exact eta times recorded direction'
 else:farray(head,p.shape);need(head.dtype==np.float64 and np.max(np.abs(head))<=.125,'Actual legal binary64 head');delta=[rational(y)-rational(x) for x,y in zip(p.ravel(),head.ravel())];mode='exact difference of actual binary64 head endpoints'
 if not len(pr['active']):need(not np.any(d),'Empty active zero direction only')
 rows=[];minslack=None
 for i,sid in enumerate(IDS):
  for j in range(6):
   shift=sum((rational(g)*v for g,v in zip(pr['J'][i,j],delta)),rational(0));pred=rational(pr['response'][i,j])+shift
   for sign in (-1,1):
    slack=(rational(pr['caps'][i])-sign*pred)/rational(SAFETY);rows.append(dict(state_id=sid,coordinate=j,sign=sign,recorded_linear_signed_response=encoded(sign*pred),normalized_recorded_cap_slack=encoded(slack)));minslack=slack if minslack is None else min(minslack,slack)
 return dict(signed_rows=1008,head_difference_mode=mode,minimum_normalized_recorded_cap_slack=encoded(minslack),all_recorded_affine_caps_strictly_satisfied=bool(minslack>0),strict_recorded_active_margin=bool(len(pr['active']) and minslack>0),active_state_ids=[IDS[i] for i in pr['active']],all1008_signed_rows=rows,scope='Exact recorded binary64 response/Jacobian affine arithmetic only; not a derivative, finite endpoint, optimality/infeasibility, curvature or generalisation certificate.')
def step_panel(p,d):return source172.step_panel(p,d)
