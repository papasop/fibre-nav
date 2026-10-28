"""Fixed full84 common-response-descent diagnostic and symmetric finite probes."""
from pathlib import Path
import sys,json,hashlib,contextlib,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor164'))
import core164 as previous
parent=previous.parent
from core164 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,safe,DIM,array_digest,point_hash,parameter_hash,registry,archive155,IDS,N,PARTS
PROTOCOL='GPT2_LC2_C165_COMMON_DESCENT_PROBES_CPU_V1'
INPUT_KEYS=previous.INPUT_KEYS+('C164',)
TIMEOUT=10800;ITERATIONS=4096;H=(1e-6,5e-6,1e-5);BOUND=.125
REQUESTED_FORWARDS=3699;FUNCTIONAL_FORWARDS=1346;REQUESTED_BACKWARDS=168;RECORDS=1848
ARMS=[dict(name=('minus_' if sign<0 else 'plus_')+label,alpha=sign*h) for label,h in zip(('1e-6','5e-6','1e-5'),H) for sign in (-1,1)]
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=previous.contract(),source164=clean_json((ROOT/'trusted/C164_SOURCE_PINS.json').read_bytes()))
def design():return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),training_registry=registry(),all84_exposed_training=True,anchor='Exact C163 fixed final local step8 head; C164 original audit independently reproduced. No best or recovered checkpoint.',gradients='Recompute response-loss R and summed-guard G scalar gradients with respect to all216 coefficient coordinates at the fixed anchor, for every84 state (168 scalar reverse calls). No model-parameter gradients or full response Jacobian.',direction=dict(method='Fixed4096-iteration Frank-Wolfe minimum-norm convex combination of normalized projected response gradients for every state with original R>0.',target_rule='R>0 means response above original0.0018 training safety level, not a selected subset of failed gates. All84 gradients and state IDs retained. Zero projected gradient rows retained as zero.',coordinate_subspace='Only coordinates with abs(anchor)<=.125-max(H) are free, so all registered positive and negative probes fit the original coefficient box. This restricted two-sided subspace cannot establish general or one-sided infeasibility.',initial_weights='Uniform across all R>0 rows.',step='argmin row dot current mixture, first canonical tie; exact clipped quadratic line step; no early stopping; all4096 iterations retained.',normalization='Normalize each projected nonzero response gradient by itsL2 norm. Negative mixture normalized to unitLinf; norm<=1e-12 becomes zero. Zero direction remains a fully measured descriptive control.',interpretation='Computed strict common descent only if each active normalized gradient dot direction<-1e-10; finite floating-point arithmetic and AD records, not a rigorous feasibility/infeasibility certificate. Guards and inactive response rows are not guaranteed by this search; all original full84 finite gates must still be measured.'),arms=ARMS,measurement_order='Native anchor round0; full84 anchor response/guard gradients and C163 FP64 replay; derive one registered direction; seal all6 symmetric heads and their physical/actual FP64 points before any probe; full84 FP64 for all6; full84 FP64 anchor repeat; all6 native round0; native anchor round1; all6 native round1.',acceptance_budgets=parent.design()['acceptance_budgets'],replay=parent.design()['replay'],counts=dict(records=1848,model_forwards=3699,functional_FP64_forwards=1346,scalar_backwards=168,actual_FP64_point_arrays=672,native_endpoint_records=1176,native_repeat_pairs=588,temporary_probe_heads=6,new_model_training_steps=0,new_head_fits=0),scope='Training-only local direction diagnostic and fixed symmetric finite probes. New offline direction optimization uses measured anchor gradients; probes are diagnostic temporary heads, not a correction-free writer or prospective acceptance claim. All84 states and all6 arms retained. No head/step selection, deployment, new preparation, confirmation or excluded-input evaluation. No causal, derivative-error, feasibility, convergence or LC2-closure proof. Original parent decisions retained.',worker_budget_seconds=TIMEOUT,LC2_closed=False)
def direction(rows,p):
 compare([r['state_id'] for r in rows],IDS,'Exact complete84 direction gradient registry');farray(p,p.shape);need(p.ndim==3 and p.shape[:2]==(3,2),'Full coupled coefficient shape');need(p.dtype==np.float64 and np.max(abs(p))<=BOUND,'Legal fixed FP64 anchor')
 free=abs(p)<=BOUND-max(H);active=[];raw=[];guards=[];loss=[]
 for i,r in enumerate(rows):
  R=float(r['objective_parts']['response_loss']);need(np.isfinite(R) and R>=0,'Finite nonnegative original response loss');a=farray(np.asarray(r['response_gradient'],float),p.shape);g=farray(np.asarray(r['guard_gradient'],float),p.shape);raw.append(a);guards.append(g);loss.append(R)
  if R>0:active.append(i)
 raw=np.asarray(raw);projected=(raw*free[None]).reshape(N,-1);norms=np.linalg.norm(projected,axis=1);need(np.isfinite(norms).all(),'Finite projected row norms');A=np.asarray([projected[i]/norms[i] if norms[i]>0 else np.zeros(p.size) for i in active]);history=[];weights=np.full(len(active),1/len(active)) if active else np.empty(0);u=weights@A if active else np.zeros(p.size)
 if active:
  for t in range(ITERATIONS):
   dots=A@u;j=int(np.argmin(dots));q=A[j]-u;den=float(np.dot(q,q));gamma=0. if den==0 else float(np.clip(-np.dot(u,q)/den,0.,1.));gap=max(0.,float(np.dot(u,u)-dots[j]));weights*=1-gamma;weights[j]+=gamma;u=weights@A;history.append(dict(iteration=t+1,vertex_state_id=IDS[active[j]],line_step=gamma,FW_gap_before=gap,mixture_norm_after=float(np.linalg.norm(u))))
 norm=float(np.linalg.norm(u));d=np.zeros(p.size) if norm<=1e-12 else -u/np.max(abs(u));d=d.reshape(p.shape);need(np.isfinite(d).all() and np.max(abs(d))<=1.,'Finite unitLinf direction');need(np.all(d[~free]==0),'Fixed interior subspace');normalized=A@d.ravel() if active else np.empty(0);strict=bool(active and np.all(normalized< -1e-10));dots=np.sum(raw*d[None],axis=tuple(range(1,raw.ndim)));gdots=np.sum(np.asarray(guards)*d[None],axis=tuple(range(1,raw.ndim)))
 summary=dict(active_state_ids=[IDS[i] for i in active],active_count=len(active),zero_projected_gradient_state_ids=[IDS[i] for i in active if norms[i]==0],free_coefficient_coordinates=int(free.sum()),total_coefficient_coordinates=p.size,iterations_completed=len(history),computed_strict_common_response_descent=strict,direction_zero=bool(not np.any(d)),mixture_L2=norm,maximum_active_normalized_gradient_dot_direction=float(max(normalized)) if active else None,all84_response_gradient_dot_direction=dots.tolist(),all84_guard_gradient_dot_direction=gdots.tolist(),all84_projected_response_gradient_norms=norms.tolist(),normalization_weights=weights.tolist(),no_infeasibility_or_finite_acceptance_claim=True)
 return d,summary,history

def probe_heads(p,d):
 out={}
 for r in ARMS:
  q=p+r['alpha']*d;need(np.isfinite(q).all() and np.max(abs(q))<=BOUND,'All symmetric probes inside original box; no clipping or search');out[r['name']]=q
 return out

def requests():
 rr=[dict(stage='native_anchor0',arm='anchor',state_id=sid) for sid in IDS]+[dict(stage='anchor_gradient',arm='anchor',state_id=sid) for sid in IDS]
 rr += [dict(stage='FP64_probe',arm=a['name'],state_id=sid) for a in ARMS for sid in IDS]
 rr += [dict(stage='FP64_anchor_repeat',arm='anchor',state_id=sid) for sid in IDS]
 rr += [dict(stage='native_probe0',arm=a['name'],state_id=sid) for a in ARMS for sid in IDS]
 rr += [dict(stage='native_anchor1',arm='anchor',state_id=sid) for sid in IDS]
 rr += [dict(stage='native_probe1',arm=a['name'],state_id=sid) for a in ARMS for sid in IDS]
 compare(len(rr),RECORDS);return rr
