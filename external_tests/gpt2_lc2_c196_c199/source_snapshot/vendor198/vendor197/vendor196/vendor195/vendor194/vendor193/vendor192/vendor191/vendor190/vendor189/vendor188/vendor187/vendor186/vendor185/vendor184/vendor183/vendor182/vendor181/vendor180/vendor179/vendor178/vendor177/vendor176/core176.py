"""Eight fixed offline all-coordinate relinearised coefficient updates."""
from pathlib import Path
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor175'))
import core175 as prior
from core175 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses,rational,encoded
source174=prior.prior
control=source174.control
seal_head=control.seal_head
native_summary=control.native_summary
paired=control.paired
objective=control.objective
problem=source174.problem
exact_certificate=source174.exact_certificate
PROTOCOL='GPT2_LC2_C176_ALL_COORDINATE_RELINEARISED_TRAIN_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C175',)
STEPS=8
STEP=1e-3
TIMEOUT=14400
RECORDS=1176
FORWARDS=2355
FUNCTIONAL=1682
BACKWARDS=5376
SCOPE='Fixed eight-step offline full84 all-six-response-coordinate relinearised LP training from the original C170 fixed final step8 head. Every current batch records six signed coordinate gradients and scalar response/guard gradients,672 scalar backwards and84 coefficient-response Jacobian blocks. Per-step one1008-row LP at fixed eta0.001, all504 coordinates including zero scalar-loss states; exact recorded-affine direction and actual-head endpoint-difference certificates before committing current+0.001*d. Read/KL/radius/regularizer guards measured but not constrained by LP. All eight slots retained; empty active uses zero direction/no LP, nonpositive witness stops incomplete without fallback or step selection. Only predetermined final step8 gets original native full84 two-round qualification; final FP64 repeat and all failures retained. No online correction-free execution, fresh confirmation, excluded input, best checkpoint, deployment, derivative truth, curvature, convergence, solver optimality/infeasibility, chronology, non-access or LC2 claim.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source175=clean_json((ROOT/'trusted/C175_SOURCE_PINS.json').read_bytes()))
def design():
 return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),training_registry=source174.design()['training_registry'],anchor='Exact original C170 fixed-final step8, not a C174 probe.',anchor_head_digest='cc06ae379e9c86be4b4f6e4eb79181d088a56b59d8c5b5d0eddbfa1de571236a',training=dict(steps=8,step=STEP,parameter_shape=[3,2,36],linear_problem=dict(eta=STEP,response_scale=.0018,cap='b_i=max(0.0018,max_j abs(r_ij)) from every current full84 batch',constraints='For all84 states,all6 response coordinates,both signs: sign*(r_ij+0.001*J_ij*d)/0.0018+t<=b_i/0.0018; maximise t>=0; abs(d_j)<=1 on abs(current p_j)<=0.125-0.001, all other coordinates0. All1008 rows retained without row normalisation.',witness='Exact rational recorded-affine cap slack for0.001*d and for exact difference of actual binary64 next/current heads; nonempty active requires positive minimum of all1008 signed rows before this update. Empty active uses registered zero/no LP.'),fresh_gradients='Each of all84 current states: six signed response-coordinate gradients plus response-loss and summed guard-loss gradients. All504 response coordinates constrain every step; no per-row L2 normalisation.',update='current+fixed0.001*direction in binary64; no head projection or step search. Exact certificate of both eta*d and the exact difference of actual binary64 endpoints before committing. Empty-active slot is registered zero/no LP. Unresolved certificate stops incomplete and retains partial work; no fallback or recovery checkpoint qualification.',solver='Original pinned SciPy1.17.0 highs-ds/options, at most one LP per active batch, at most8 total; solver internal linear-system count unknown/null.',guard='Read/KL/radius/regularizer and summed guard gradients recorded; response-only LP supplies no finite guard guarantee.'),frozen='Model/revision,LoRA-A,prepared/raw states,V,routes,original C155 cached response/logit references and all native thresholds. Only shared coefficient head changes offline.',references=source174.design()['references'],acceptance_budgets=source174.design()['acceptance_budgets'],measurement_order='Native initial84 round0; eight full84 coordinate+loss AD batches and fixed updates; frozen final84 FP64; frozen final84 FP64 repeat; native final84 round0; original initial84 round1; final84 round1.',sealing='Design,initial head,constants,all1176 requests before model. Each current head and84 actual FP64 points before its own AD batch. Direction and actual next-head affine certificates before each update; each completed update sealed and rehashed thereafter. Fixed final step8 and both final FP64 point batches sealed before final measurement.',replay='Initial native both rounds and first full84 FP64 versus original C174 anchor under original tolerances/identities; no baseline recentering.',counts=dict(records=RECORDS,model_forwards=FORWARDS,functional_FP64_forwards=FUNCTIONAL,scalar_backwards=BACKWARDS,response_coordinate_backwards=4032,response_loss_backwards=672,guard_loss_backwards=672,coefficient_response_Jacobian_blocks=672,coefficient_response_Jacobian_rows=4032,max_LP_calls=8,LP_signed_rows_each=1008,max_LP_signed_rows_total=8064,actual_FP64_point_arrays=840,native_FP32_point_arrays=168,native_endpoint_records=336,native_repeat_pairs=168,FP64_final_repeat_pairs=84,coefficient_update_slots=8),worker_budget_seconds=TIMEOUT,scope=SCOPE,LC2_closed=False)
def requests():
 r=[dict(stage='native_initial0',step=0,state_id=s) for s in IDS]
 r += [dict(stage='training',step=t,state_id=s) for t in range(STEPS) for s in IDS]
 for kind in ('final_FP64','final_FP64_repeat'):r += [dict(stage=kind,step=8,state_id=s) for s in IDS]
 for kind,t in [('native_final0',8),('native_initial1',0),('native_final1',8)]:r += [dict(stage=kind,step=t,state_id=s) for s in IDS]
 compare(len(r),RECORDS);return r
def update(p,d):
 farray(p,p.shape);farray(d,p.shape);need(p.dtype==d.dtype==np.float64 and np.max(abs(p))<=.125 and np.max(abs(d))<=1,'Finite bounded binary64 update');need(not np.any(d[np.abs(p)>.125-STEP]),'Original full fixed-step free mask');q=np.add(p,np.multiply(STEP,d));need(np.isfinite(q).all() and np.max(abs(q))<=.125,'Actual unprojected head fits original box')
 return q,dict(step_size=STEP,update_Linf=float(np.max(abs(q-p))),head_before=parameter_hash(p),head_after=parameter_hash(q),head_projection_performed=False,step_shrinking_performed=False)
