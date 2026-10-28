"""Fixed finite probes of the exact C167 recorded-gradient direction."""
from pathlib import Path
from fractions import Fraction as Q
import sys,json,hashlib,time,contextlib,math
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor167'))
import core167 as prior
from core167 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,archive155,IDS,N
old=prior.prior.parent # trusted core165
safe=prior.prior.safe
DIM=prior.prior.DIM
PARTS=prior.prior.PARTS
parameter_hash=prior.prior.parameter_hash
array_digest=prior.prior.array_digest
point_hash=prior.prior.point_hash
PROTOCOL='GPT2_LC2_C168_CERTIFIED_DIRECTION_PROBES_CPU_V1'
INPUT_KEYS=prior.INPUT_KEYS+('C167',)
TIMEOUT=10800
RECORDS=1848
REQUESTED_FORWARDS=3699
FUNCTIONAL_FORWARDS=1346
REQUESTED_BACKWARDS=0
ARM_NAMES=[('minus_' if sign<0 else 'plus_')+'s'+str(i) for i in (1,2,3) for sign in (-1,1)]
SCOPE='Training-only finite model probes of the exact C167 direction at the original C163 fixed-final anchor. All84 exposed training states and all6 registered arms retained. No new direction solve, backward, training, head fit, selection, deployment, confirmation or excluded-input evaluation. Recorded-gradient certificates do not certify actual derivatives, guards or finite writes; original native gates are measured separately. No LC2 closure, causal or global feasibility claim. C165-C167 decisions retained.'
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=prior.contract(),source167=clean_json((ROOT/'trusted/C167_SOURCE_PINS.json').read_bytes()))
def design():return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),training_registry=prior.prior.registry(),anchor='Exact C163 fixed final step8; no refit or recovered checkpoint.',direction='Exact serialized C167 DIRECTION_CERTIFICATE_ARRAYS.npz direction, independently reopen originalC167 and reverify rational certificate against originalC165 gradients; require strictly positive exact lower bound. No solver import or call.',step_rule='Before model initialization: exact rational r=min_{d_j!=0} (0.125-|p_j|)/|d_j|; h3=min(0.001,r/2), converted downward to binary64; h1=h3/100,h2=h3/10. Register all six +/- arms, check every actual binary64 head in the original box, no clipping/retry/search or measured-data dependence.',arm_names=ARM_NAMES,all84_all6_retained=True,seal_rule='All six NumPy heads/points and anchor/repeat before model initialization. All eight actual Torch FP64 batches after zero-chart baseline initialization but before first state measurement. Rehash all receipts and direction; model must consume exact saved point bytes.',measurements='Native anchor round0; full84 no-grad FP64 anchor; six full84 FP64 probe arms; full84 anchor repeat; native six arms round0; native anchor round1; native six arms round1. All original native gates retained, failed endpoints remain denominator failures and do not stop later registered arms.',references='Original cached C155 FP32 response/logit references, fixed routes/basis/writer/state coordinates, model/revision and zero LoRA-B; original same-anchor replay checks unchanged. No new baseline selection, recentering or calibration.',derivative_comparison='C165 archived response/guard scalar-gradient dot C167 direction versus same-function C168 central differences at all3h, plus signed one-sided loss changes for all84 states. No new AD or derivative-accuracy claim.',acceptance_budgets=old.parent.design()['acceptance_budgets'],replay=old.parent.design()['replay'],counts=dict(records=RECORDS,model_forwards=REQUESTED_FORWARDS,functional_FP64_forwards=FUNCTIONAL_FORWARDS,scalar_backwards=0,actual_FP64_point_arrays=672,native_endpoint_records=1176,native_repeat_pairs=588,temporary_probe_heads=6,new_training_steps=0,new_direction_solves=0),scope=SCOPE,worker_budget_seconds=TIMEOUT,LC2_closed=False)
def step_panel(p,d):
 farray(p,p.shape);farray(d,p.shape);need(p.dtype==d.dtype==np.float64 and p.ndim==3 and p.shape[:2]==(3,2),'Exact full coefficient tensors');need(np.max(abs(p))<=.125 and 0<np.max(abs(d))<=1,'Legal nonzero original direction')
 radii=[(Q(1,8)-Q.from_float(abs(float(x))))/Q.from_float(abs(float(y))) for x,y in zip(p.ravel(),d.ravel()) if y!=0];r=min(radii);need(r>0,'Strict two-sided coefficient slack required');h3q=min(Q.from_float(.001),r/2);h3=float(h3q)
 if Q.from_float(h3)>h3q:h3=np.nextafter(h3,0.)
 H=[float(h3/100),float(h3/10),float(h3)];need(all(np.isfinite(H)) and 0<H[0]<H[1]<H[2],'Three positive finite fixed steps')
 arms=[dict(name=('minus_' if sign<0 else 'plus_')+'s'+str(i+1),alpha=float(sign*h)) for i,h in enumerate(H) for sign in (-1,1)];heads={a['name']:p+a['alpha']*d for a in arms}
 for name,q in heads.items():need(np.isfinite(q).all() and np.max(abs(q))<=.125,'All6 actual heads fit original box; no clipping/'+name)
 receipt=dict(rule=design()['step_rule'],symmetric_radius_exact=dict(numerator=str(r.numerator),denominator=str(r.denominator),approximate=float(r)),largest_step_exact_before_downward_rounding=dict(numerator=str(h3q.numerator),denominator=str(h3q.denominator),approximate=float(h3q)),positive_steps=H,arms=arms,actual_head_max_abs={n:float(np.max(abs(q))) for n,q in heads.items()},computed_from_prewrite_anchor_and_fixed_direction_only=True,postwrite_selection=False)
 return heads,receipt

def requests():
 rr=[dict(stage='native_anchor0',arm='anchor',state_id=s) for s in IDS]+[dict(stage='FP64_anchor',arm='anchor',state_id=s) for s in IDS]
 rr += [dict(stage='FP64_probe',arm=a,state_id=s) for a in ARM_NAMES for s in IDS]
 rr += [dict(stage='FP64_anchor_repeat',arm='anchor',state_id=s) for s in IDS]
 rr += [dict(stage='native_probe0',arm=a,state_id=s) for a in ARM_NAMES for s in IDS]
 rr += [dict(stage='native_anchor1',arm='anchor',state_id=s) for s in IDS]
 rr += [dict(stage='native_probe1',arm=a,state_id=s) for a in ARM_NAMES for s in IDS]
 compare(len(rr),RECORDS);return rr
