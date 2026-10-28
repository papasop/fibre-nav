"""Fixed signed finite-step probes at the original C170 fixed final head."""
from pathlib import Path
import sys,json,hashlib,contextlib,time,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor171'))
import core171 as prior
from core171 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N,DIM,PARTS,parameter_hash,array_digest,point_hash,safe,mapping,losses
control=prior.prior
seal_head=control.seal_head
native_summary=control.native_summary
paired=control.paired
PROTOCOL='GPT2_LC2_C172_FINAL_HEAD_FINITE_STEPS_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C171',)
H=(1e-4,3e-4,1e-3)
ARM_NAMES=[('minus_' if s<0 else 'plus_')+'s'+str(i+1) for i in range(3) for s in (-1,1)]
TIMEOUT=10800
RECORDS=1848
FORWARDS=3699
FUNCTIONAL=1346
BACKWARDS=168
SCOPE='Offline training-only fixed signed finite-step diagnostic from the original C170 fixed final step8, independently rebound through C171. One fresh full84 FP64 response/guard gradient batch, at most one response-only LP and exact recorded-gradient witness. Six fixed +/- coefficient probes at1e-4,3e-4,1e-3, sealed together after the gradient and direction but before any probe measurement; all84 and all6 retained. No optimizer update/resumed training, step/head selection, deployment, new preparation, confirmation or excluded input. Guard gradients are recorded but not constrained by the LP. Native hard gates remain unchanged; no derivative truth, solver optimality/infeasibility, curvature, convergence, causal explanation or LC2 closure claim.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source171=clean_json((ROOT/'trusted/C171_SOURCE_PINS.json').read_bytes()))
def design():
 solver=dict(control.design()['training']['solver'],max_calls=1)
 return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),training_registry=control.design()['training_registry'],anchor='Exact original C170 FIXED_FINAL_HEAD_STEP8.npz; no C171-selected head or historical best checkpoint.',initial_head_digest='cc06ae379e9c86be4b4f6e4eb79181d088a56b59d8c5b5d0eddbfa1de571236a',direction=dict(batch='Fresh complete84 FP64 response and guard gradients at the fixed final head; response_loss>0 includes every active state, including zero gradient.',free_mask='abs(p_j)<=0.125-0.001; excluded coefficients have direction0, reserving all six symmetric fixed probes without shrinking their step sizes.',LP=control.design()['training']['LP'],solver=solver,max_LP_calls=1,witness='All active raw recorded-gradient contractions strictly negative in exact rational arithmetic. Empty active set registers zero direction/no LP; nonempty unresolved direction stops incomplete, without retry or alternate solver.',guard='Measured and recorded, not included as an LP constraint.'),positive_steps=list(H),arms=[dict(name=n,alpha=float(s*h)) for h,i in zip(H,range(3)) for s,n in [(-1,ARM_NAMES[2*i]),(1,ARM_NAMES[2*i+1])]],head_rule='Each actual binary64 q=p+alpha*d is independently checked against the original coefficient box. No head clipping, shrinking, line search, retry, extrapolation or head selection.',seals='Freeze constants, anchor and all1848 measurement requests before model initialization; actual anchor points before gradients. Freeze one direction and all six NumPy and actual Torch FP64 probe batches together after the full gradient and LP, before any probe observation. Gradient measurements precede probe sealing and are explicitly counted.',measurement_order='Native anchor round0; FP64 anchor with two scalar AD per state; six FP64 probe batches; FP64 anchor repeat without AD; six native probe batches round0; native anchor round1; six native probe batches round1.',replay='Anchor native two rounds against original C170 final same rounds; anchor FP64 against original C170 final within original2e-9 observed/KL tolerance. No recentering. Repeat anchor before final native rounds.',references=control.design()['references'],acceptance_budgets=control.design()['acceptance_budgets'],counts=dict(records=RECORDS,model_forwards=FORWARDS,functional_FP64_forwards=FUNCTIONAL,scalar_backwards=BACKWARDS,max_LP_calls=1,actual_FP64_point_arrays=672,native_FP32_point_arrays=588,native_endpoint_records=1176,native_repeat_pairs=588,FP64_anchor_repeat_pairs=84,temporary_probe_heads=6,new_training_steps=0),worker_budget_seconds=TIMEOUT,scope=SCOPE,LC2_closed=False)
def requests():
 rr=[dict(stage='native_anchor0',arm='anchor',state_id=s) for s in IDS]+[dict(stage='FP64_anchor',arm='anchor',state_id=s) for s in IDS]
 rr += [dict(stage='FP64_probe',arm=a,state_id=s) for a in ARM_NAMES for s in IDS]
 rr += [dict(stage='FP64_anchor_repeat',arm='anchor',state_id=s) for s in IDS]
 rr += [dict(stage='native_probe0',arm=a,state_id=s) for a in ARM_NAMES for s in IDS]
 rr += [dict(stage='native_anchor1',arm='anchor',state_id=s) for s in IDS]
 rr += [dict(stage='native_probe1',arm=a,state_id=s) for a in ARM_NAMES for s in IDS]
 compare(len(rr),RECORDS);return rr
def problem(rows,p):
 pr=control.problem(rows,p);pr['free']=(np.abs(p).ravel()<=.125-H[-1]);pr['norms']=np.linalg.norm(pr['raw']*pr['free'],axis=1);need(np.isfinite(pr['norms']).all(),'Finite projected row norms');pr['A']=np.array([pr['raw'][i]*pr['free']/pr['norms'][i] if pr['norms'][i]>0 else np.zeros(p.size) for i in pr['active']],np.float64).reshape(len(pr['active']),p.size);return pr
def step_panel(p,d):
 farray(p,p.shape);farray(d,p.shape);need(p.dtype==d.dtype==np.float64 and p.ndim==3 and p.shape[:2]==(3,2),'Full binary64 coefficient tensors');need(np.max(np.abs(p))<=.125 and np.max(np.abs(d))<=1,'Original coefficient and direction bounds');free=(np.abs(p)<=.125-H[-1]);need(not np.any(d[~free]),'Direction uses only full-panel safe subspace');arms=design()['arms'];heads={a['name']:np.add(p,np.multiply(a['alpha'],d)) for a in arms}
 for n,q in heads.items():need(np.isfinite(q).all() and np.max(np.abs(q))<=.125,'Actual symmetric head outside original box/'+n)
 return heads,dict(positive_steps=list(H),arms=arms,anchor_head_digest=parameter_hash(p),direction_digest=array_digest(d),head_digests={n:parameter_hash(q) for n,q in heads.items()},zero_direction_noop=not bool(np.any(d)),head_projection_performed=False,step_shrinking_performed=False,postwrite_selection=False)
