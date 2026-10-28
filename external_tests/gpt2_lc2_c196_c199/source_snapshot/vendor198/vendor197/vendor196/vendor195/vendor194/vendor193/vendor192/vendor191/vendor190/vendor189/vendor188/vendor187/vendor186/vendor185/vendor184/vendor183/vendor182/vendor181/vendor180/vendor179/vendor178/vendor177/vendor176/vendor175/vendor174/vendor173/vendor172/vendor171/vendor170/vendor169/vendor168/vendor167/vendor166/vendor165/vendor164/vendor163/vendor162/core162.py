"""Independent arithmetic and bounded diagnostics for archived C161 training."""
from pathlib import Path
import sys,json,hashlib,contextlib,time
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor161'))
import core161 as parent
from core161 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,safe,DIM,array_digest,point_hash,parameter_hash,registry,archive155
PROTOCOL='GPT2_LC2_C162_READONLY_FP64_TRAIN_AUDIT_CPU_V1'
INPUT_KEYS=parent.INPUT_KEYS+('C161',);N=84;STEPS=8;TIMEOUT=1200
IDS=[s['state_id'] for s in registry()]
FOCUS=['xor_00__c141_confirmation_p02_n01_d01','xor_11__c141_confirmation_p08_n01_d00','xor_11__c141_confirmation_p08_n01_d01','xor_11__c141_confirmation_p09_n01_d00','xor_11__c141_confirmation_p11_n01_d01']
PARTS=('response_loss','read_loss','KL_loss','radius_loss','regularizer')
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=parent.contract(),source161=clean_json((ROOT/'trusted/C161_SOURCE_PINS.json').read_bytes()))
def design():return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),training_registry=registry(),fixed_diagnostic_focus=FOCUS,focus_is_retrospective_exposed_training_only=True,arithmetic='Independently reconstruct all672 full-loss records, all8 max/mean gradient aggregations and projected Adam updates; bind original alpha0 and exact fixed-final step8; recompute336 native gates,168 native repeats, all84 signed coordinate margins and672 recorded transition diagnostics.',optimizer=dict(steps=8,rate=1e-5,betas=[.9,.999],epsilon=1e-8,global_L2_clip=1.,coordinate_box=[-.125,.125],moments_start_at_zero=True,final_checkpoint=8),thresholds=dict(response=.002,response_training_safety=.0018,KL=.005,read_margin=.05,proposal_radius=20.,radius_rounding_allowance=1e-5),precision_comparison='Compare archived continuous FP64 observables with physical FP32 endpoint observables for all84 at initial/final heads and both native repeats. Classifications concern response gate only; no FP64 full acceptance score. Combined path discrepancy does not isolate rounding, forward precision, reductions, reference arithmetic or hardware.',point_binding='Reconstruct and hash archived NumPy continuous maps, bind actual saved FP32 point arrays and coefficient bytes. Per-forward Torch FP64 coordinates were not archived in C161, so their opaque hashes and reported map errors cannot be independently rehashed or validated as physical measurements.',gradient_scope='Archived response and guard coefficient gradients are finite recorded inputs, not independently recomputed derivatives. Dot products, cosines and observed-minus-predicted changes are descriptive, not causal explanations, curvature bounds, stationarity or convergence certificates.',parent_decision='Retain fixed C161 decision, full denominator and all failures; no best checkpoint, alternative step, head, rate or route selected.',worker_budget_seconds=TIMEOUT,new_model_forwards=0,new_backward_calls=0,new_training_steps=0,new_head_fits=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,LC2_closed=False,scope='Pure CPU NumPy and read-only seven-original archive audit. No Torch/HF/model import, archived-source execution, model replay, preparation replay or new endpoint. Archive integrity cannot establish measurement accuracy, chronology, non-access, derivative truth, optimizer causality, feasibility, convergence or generalisation. LC2 remains open.')
def losses(observed,kl,point,prepared,e,case):
 o=farray(np.asarray(observed,np.float64),(22,));need(case in ('xor_00','xor_01','xor_11'),'Training classes only');need(np.isfinite(kl) and kl>=0,'Finite nonnegative archived KL');x,y=int(case[-2]),int(case[-1]);sign=2*np.tile([x,y,1,x^y],4)-1;r=max(0.,float(np.max(np.abs(o[:6])))/.0018-1.);a=max(0.,float(kl)/.0045-1.);b=max(0.,float(np.linalg.norm(point-prepared))/20.-1.)
 return dict(response_loss=r*r,read_loss=float(np.square(np.maximum(0.,1.-sign*o[6:]/.075)).mean()),KL_loss=a*a,radius_loss=b*b,regularizer=float(.001*np.square(e).sum()/e.shape[-1]))
def objective(rows):
 compare([r['state_id'] for r in rows],IDS,'All84 ordered objective rows');R=[];G=[]
 for r in rows:
  compare(set(r['objective_parts']),set(PARTS),'Exact loss terms');a=[float(r['objective_parts'][k]) for k in PARTS];need(np.isfinite(a).all() and min(a)>=0,'Finite nonnegative loss parts');R.append(a[0]);G.append(sum(a[1:]))
 R=np.asarray(R);G=np.asarray(G);top=R==R.max();count=int(top.sum());weights=np.full(N,.1/N);weights[top]+=.9/count
 return dict(objective=float(.9*R.max()+.1*R.mean()+G.mean()),maximum_response_loss=float(R.max()),mean_response_loss=float(R.mean()),mean_guard_loss=float(G.mean()),response_weights=weights.tolist(),guard_weights=[1/N]*N,maximizing_state_ids=[IDS[i] for i in np.flatnonzero(top)],exact_max_tie_count=count)
def aggregate(rows,shape):
 a=objective(rows);rr=np.asarray([farray(np.asarray(r['response_gradient'],np.float64),shape).ravel() for r in rows]);gg=np.asarray([farray(np.asarray(r['guard_gradient'],np.float64),shape).ravel() for r in rows]);weights=np.asarray(a['response_weights']);g=(weights[:,None]*rr+gg/N).sum(axis=0).reshape(shape);need(np.isfinite(g).all(),'Finite aggregate gradient');return g,a
def adam(p,g,m,v,step):
 need(type(step) is int and 1<=step<=8,'Fixed eight-step schedule');shape=p.shape;need(len(shape)==3 and shape[:2]==(3,2),'Full coupled head shape')
 for a in (p,g,m,v):need(a.dtype==np.float64,'FP64 optimizer state');farray(a,shape)
 need(np.all(v>=0) and np.max(abs(p))<=.125,'Legal optimizer state');norm=float(np.linalg.norm(g));need(np.isfinite(norm),'Finite gradient norm');clip=1. if norm<=1. else 1./norm
 pp=p.ravel();grad=g.ravel()*clip;mm=np.empty_like(grad);vv=np.empty_like(grad);result=np.empty_like(grad);unrestricted=np.empty_like(grad)
 for j in range(len(grad)):
  mm[j]=.9*m.ravel()[j]+.1*grad[j];vv[j]=.999*v.ravel()[j]+.001*grad[j]*grad[j];unrestricted[j]=pp[j]-1e-5*(mm[j]/(1-.9**step))/(np.sqrt(vv[j]/(1-.999**step))+1e-8);result[j]=min(.125,max(-.125,unrestricted[j]))
 need(np.isfinite(result).all(),'Finite projected head');return result.reshape(shape),mm.reshape(shape),vv.reshape(shape),dict(step=step,gradient_L2=norm,clip_factor=clip,clipped_gradient_L2=float(np.linalg.norm(grad)),projected_coordinates=int(np.count_nonzero(result!=unrestricted)),update_Linf=float(np.max(abs(result-pp))))
def mapping(c,p,sid):
 farray(p,c['P0'].shape);need(p.dtype==np.float64 and np.max(abs(p))<=.125,'Fixed-box FP64 head');w=c[sid+'__weights'];u=c['V']@(c[sid+'__raw']-c[sid+'__prepared']);e=(w[0]*p[0]+w[1]*p[1])+w[2]*p[2];shift=np.concatenate((u[-1:],u[:-1]));point=c[sid+'__raw']+c['V'].T@(e[0]*u+e[1]*shift);return point,e
def native_point(c,p,sid):
 point,_=mapping(c,p,sid);return point.astype(np.float32)
def cosine(a,b):
 a=np.asarray(a,float).ravel();b=np.asarray(b,float).ravel();n=float(np.linalg.norm(a)*np.linalg.norm(b));return None if n==0 else float(np.clip(np.dot(a,b)/n,-1.,1.))
def response_class(fp64,native):
 a=bool(np.max(abs(np.asarray(fp64)[:6]))<=.002);b=bool(np.max(abs(np.asarray(native)[:6]))<=.002)
 return ('BOTH_RESPONSE_PASS' if a and b else 'FP64_RESPONSE_ONLY_FAIL' if b else 'NATIVE_RESPONSE_ONLY_FAIL' if a else 'BOTH_RESPONSE_FAIL')
