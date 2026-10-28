"""Independent read-only arithmetic contract for the fixed C163 result."""
from pathlib import Path
import sys,json,hashlib,contextlib,time
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor163'))
import core163 as parent
old=parent.previous
from core163 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,safe,DIM,array_digest,point_hash,parameter_hash,registry,archive155
N=84;STEPS=8;TAU=.01;TIMEOUT=1800
PROTOCOL='GPT2_LC2_C164_READONLY_SMOOTH_TRAIN_AUDIT_CPU_V1'
INPUT_KEYS=parent.INPUT_KEYS+('C163',)
IDS=parent.IDS;FOCUS=old.FOCUS;PARTS=old.PARTS
losses=old.losses;mapping=old.mapping;cosine=old.cosine;response_class=old.response_class

def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=parent.contract(),source163=clean_json((ROOT/'trusted/C163_SOURCE_PINS.json').read_bytes()))
def design():return dict(protocol=PROTOCOL,inputs=list(INPUT_KEYS),training_registry=registry(),retrospective_focus_ids=FOCUS,all84_retained=True,arithmetic='Nine-original binding; all672 recorded training losses,8 smooth coefficient-gradient aggregations and projected Adam updates,84 final FP64 losses,336 native gates and168 repeat pairs. Exact C161 final step8 restart, zero initial moments, C163 fixed final local step8 only.',smooth_objective='tau=.01; J=.9*tau*log(mean(exp(R/tau)))+.1*mean(R)+mean(G). Independent shifted scalar implementation; original hardmax objective also reconstructed.',actual_points='Rehash all756 saved actual Torch FP64 coordinate arrays, nine batch receipts and every row binding; independently compare actual point versus reconstructed NumPy map within original1e-12 and recompute recorded map_Linf. Also bind168 saved FP32 proposal arrays. Receipts/counters are archive evidence, not independent proof of real chronology or model consumption.',diagnostics='All672 per-state observed changes, recorded gradient dot actual update and linear residuals; all8 full84 response-gradient cosine matrices; smooth versus original hardmax and average/tail losses; all84 response margins at all9 heads and336 initial/final FP64-versus-native comparisons. Retrospective fixed five failures shown in addition to all states; no candidate selection.',thresholds=old.design()['thresholds'],gradient_scope='Recorded response/guard gradients are inputs, not independently evaluated derivatives. KL is an archived scalar. No causal attribution, curvature/error bound, feasibility, stationarity, convergence or optimizer superiority claim.',worker_budget_seconds=TIMEOUT,LC2_closed=False,scope='Pure NumPy read-only nine-original archival audit. No Torch/HF/model import, forward, backward, fitting, solving, SVD, new preparation, confirmation or excluded-input evaluation. Saved actual FP64 points become independently rehashable; model outputs, gradient truth, KL accuracy, chronology and non-access remain unproved. Parent decisions retained; LC2 remains open.')
def objective(rows):
 base=old.objective(rows);R=np.array([float(r['objective_parts']['response_loss']) for r in rows]);top=float(R.max());e=np.array([__import__('math').exp((float(x)-top)/TAU) for x in R]);den=float(__import__('math').fsum(e));risk=max(0.,top+TAU*__import__('math').log(den/N));weights=.1/N+.9*e/den
 return dict(objective=float(.9*risk+.1*base['mean_response_loss']+base['mean_guard_loss']),smooth_response_risk=float(risk),temperature=TAU,original_hardmax_objective=base['objective'],maximum_response_loss=base['maximum_response_loss'],mean_response_loss=base['mean_response_loss'],mean_guard_loss=base['mean_guard_loss'],response_weights=weights.tolist(),guard_weights=[1/N]*N,hardmax_maximizing_state_ids=base['maximizing_state_ids'],hardmax_exact_tie_count=base['exact_max_tie_count'])
def aggregate(rows,shape):
 a=objective(rows);g=np.zeros(int(np.prod(shape)),np.float64)
 for i,r in enumerate(rows):
  rg=farray(np.asarray(r['response_gradient'],np.float64),shape);gg=farray(np.asarray(r['guard_gradient'],np.float64),shape);g+=a['response_weights'][i]*rg.ravel()+gg.ravel()/N
 need(np.isfinite(g).all(),'Finite independent smooth aggregate');return g.reshape(shape),a

def adam(p,g,m,v,step):
 """Independent vector equations, distinct from C163's scalar C162 updater."""
 need(type(step) is int and 1<=step<=8,'Fixed local Adam step')
 for a in (p,g,m,v):
  need(a.dtype==np.float64,'FP64 Adam state');farray(a,p.shape)
 need(p.ndim==3 and p.shape[:2]==(3,2) and np.all(v>=0) and np.max(abs(p))<=.125,'Legal coupled Adam state')
 norm=float(np.linalg.norm(g));need(np.isfinite(norm),'Finite gradient norm');factor=1./max(1.,norm);clipped=g*factor
 mm=.9*m+.1*clipped;vv=.999*v+.001*np.square(clipped);bias_m=mm/(1.-.9**step);bias_v=vv/(1.-.999**step);unrestricted=p-1e-5*bias_m/(np.sqrt(bias_v)+1e-8);new=np.minimum(.125,np.maximum(-.125,unrestricted))
 need(np.isfinite(new).all(),'Finite independent Adam update')
 return new,mm,vv,dict(step=step,gradient_L2=norm,clip_factor=factor,clipped_gradient_L2=float(np.linalg.norm(clipped)),projected_coordinates=int(np.count_nonzero(new!=unrestricted)),update_Linf=float(np.max(abs(new-p))))
