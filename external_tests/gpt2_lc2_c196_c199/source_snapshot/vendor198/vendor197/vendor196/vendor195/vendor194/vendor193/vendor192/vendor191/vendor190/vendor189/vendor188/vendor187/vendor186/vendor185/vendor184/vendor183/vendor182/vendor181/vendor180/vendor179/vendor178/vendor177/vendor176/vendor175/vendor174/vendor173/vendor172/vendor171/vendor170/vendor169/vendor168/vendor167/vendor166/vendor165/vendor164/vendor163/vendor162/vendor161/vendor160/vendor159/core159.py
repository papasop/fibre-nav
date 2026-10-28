"""C159 fixed continuous precommit coefficient-chain diagnostic; no learning."""
from pathlib import Path
import contextlib,hashlib,json,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'vendor158'))
import core158 as previous
from core158 import need,sha,dump,clean_json,Archive,farray,exact,close,compare,safe,DIM,point_hash,describe,response_loss,analyse
PROTOCOL='GPT2_LC2_C159_COEFFICIENT_CHAIN_FP64_CPU_V1'
INPUT_KEYS=('C155','C156','C157','C158');FOCUS=previous.FOCUS;LABELS=previous.LABELS;SCALES=previous.SCALES;REFERENCES=previous.REFERENCES
TIMEOUT=7200;WITHIN_RUN_TOL=2e-9;MAP_TOL=1e-12
REQUESTED_FORWARDS=27;REQUESTED_BACKWARDS=18
def contract():return dict(production=True,pins=clean_json((ROOT/'INPUT_PINS.json').read_bytes()),parent=previous.contract(),source158=clean_json((ROOT/'trusted/C158_SOURCE_PINS.json').read_bytes()))
def design():return dict(protocol=PROTOCOL,input_archives=list(INPUT_KEYS),all84_remain_exposed_training=True,focus_state_ids=list(FOCUS),head_anchor='C155 small_step12 cyclic_rate_1e4 before/after; all3x2x36 coefficients',path='P(alpha)=P0+alpha*(P1-P0), left-associated e=(w0*P[0]+w1*P[1])+w2*P[2], u=V@(raw-prepared), theta=raw+V.T@(e[0]*u+e[1]*roll(u,+1)). All constants exact FP64 casts, no final FP32 cast in differentiable path.',rounding_boundary='This continuous precommit FP64 path is NOT the exact FP32-commit function or the C158 rounded-endpoint chord. Both differences and coefficient alpha1 rounding are retained. No straight-through derivative is used or certified for the FP32 cast.',routing='Original archived prewrite convex weights stay fixed; no route recomputation, new features or routing selection.',labels=LABELS,scales=[h for h,_,_ in SCALES],references=previous.design()['references'],reference_rule=previous.design()['reference_rule'],cross_run=previous.design()['cross_run'],AD='Six response-coordinate scalar-alpha derivatives and three scalar loss reverse calls per state. Each loss reverse call jointly returns gradients with respect to alpha, effective2xrank controls and complete3x2xrank coefficients. All other tensors frozen. No model-parameter gradients or parameter Jacobian.',chain='Independently form w*g_effective, its contraction with deltaP, and compare to direct coefficient gradient and alpha AD; retain signed residuals without a derivative-agreement pass threshold. Independent NumPy precommit construction and exact original FP32 map are checked before model.',response_loss=previous.design()['response_loss'],counts=previous.design()['counts'],map_reconstruction_tolerance=MAP_TOL,within_run_tolerance=WITHIN_RUN_TOL,LC2_acceptance_budgets=previous.design()['LC2_acceptance_budgets'],LC2_acceptance_thresholds_changed=False,worker_budget_seconds=TIMEOUT,scope='Training-only two-state same-run FP64 precommit coefficient-chain diagnostic. Original C155-C158 decisions retained. No derivative certification for discrete FP32 commit, accuracy/error/curvature bound, optimizer causality, feasibility or generalisation. No training, fitting, head/route/step selection, deployment, preparation, fresh confirmation, excluded-input testing or full84 FP64 qualification.',LC2_closed=False,LC2_single_fold_passed=False,candidate_selected=False)
def array_digest(a):
 a=np.asarray(a);need(a.dtype.kind in 'fiu' and np.isfinite(a).all(),'Finite numeric constants');return hashlib.sha256((a.dtype.str+str(a.shape)).encode()+a.tobytes()).hexdigest()
def constants(data):
 d=data['d'];a=farray(data['before'],data['before'].shape);b=farray(data['after'],a.shape);rank=d['V'].shape[0]
 need(a.shape==(3,2,rank) and a.dtype==b.dtype==np.float64 and 1<rank<=36,'Full coupled coefficient shape');need(np.any(b!=a),'Nonzero archived coefficient update');V=farray(d['V'],(rank,DIM));need(V.dtype in (np.float32,np.float64),'Floating original shared V')
 out=dict(P0=a.copy(),deltaP=b-a,V=V.astype(np.float64))
 for sid in FOCUS:
  q=d['points'][sid];w=farray(d['weights'][sid],(3,));need(np.all(w>=0) and abs(w.sum()-1)<1e-14,'Original fixed convex route')
  need(q['raw'].dtype==q['prepared'].dtype==np.float32,'Original FP32 physical inputs')
  for key in ('raw','prepared'):out[sid+'__'+key]=farray(q[key],(DIM,)).astype(np.float64)
  out[sid+'__weights']=w.astype(np.float64).copy()
 return out
def numpy_map(c,sid,alpha):
 need(type(alpha) in (int,float) and alpha in LABELS.values(),'Registered scalar alpha');p=c['P0']+float(alpha)*c['deltaP'];need(np.max(abs(p))<=.125+1e-15,'Original coefficient box');w=c[sid+'__weights'];e=(w[0]*p[0]+w[1]*p[1])+w[2]*p[2];u=c['V']@(c[sid+'__raw']-c[sid+'__prepared']);return c[sid+'__raw']+c['V'].T@(e[0]*u+e[1]*np.roll(u,1)),p,e

def freeze(data,out):
 from map159 import torch_constants,torch_map
 import torch
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.use_deterministic_algorithms(True)
 out=Path(out);c=constants(data);tc=torch_constants(c);points={};tangents={};rows=[]
 for sid in FOCUS:
  w=c[sid+'__weights'];de=(w[0]*c['deltaP'][0]+w[1]*c['deltaP'][1])+w[2]*c['deltaP'][2];u=c['V']@(c[sid+'__raw']-c[sid+'__prepared']);tangents[sid]=c['V'].T@(de[0]*u+de[1]*np.roll(u,1))
  need(np.any(tangents[sid]!=0),'Nonzero actual physical tangent')
  for n,alpha in LABELS.items():
   want,p,e=numpy_map(c,sid,alpha)
   with torch.no_grad():point,pp,ee=torch_map(tc,sid,torch.tensor(float(alpha),dtype=torch.float64))
   point=point.numpy().copy();err=float(np.max(abs(point-want)));need(err<=MAP_TOL,'Independent NumPy/Torch precommit map disagreement');points[(sid,n)]=point
   original=data['points'][n+'__'+sid];q=data['d']['points'][sid];oldp=data['coefs'][n];legacy=previous.previous.parent.parent.prior.proposal(q['prepared'],q['raw'],data['d']['V'],previous.previous.parent.parent.prior.effective(oldp,data['d']['weights'][sid],previous.previous.parent.ARM));exact(legacy,original,'Exact legacy FP32 coefficient-map reconstruction')
   chord=previous.previous.chord(data['points']['zero__'+sid],data['points']['plus_one__'+sid],alpha)
   rows.append(dict(state_id=sid,label=n,alpha=alpha,FP64_point_sha256=point_hash(point),independent_numpy_map_Linf=err,original_FP32_commit_Linf=float(max(abs(point-original.astype(float)))),rounded_new_map_vs_original_FP32_Linf=float(max(abs(point.astype(np.float32).astype(float)-original.astype(float)))),rounded_new_map_coordinates_different=int(np.count_nonzero(point.astype(np.float32)!=original)),C158_physical_chord_Linf=float(max(abs(point-chord))),coefficient_vs_archived_Linf=float(max(abs(p-oldp).reshape(-1))),effective_vs_archived_Linf=float(max(abs(e-previous.previous.parent.parent.prior.effective(oldp,data['d']['weights'][sid],previous.previous.parent.ARM)).reshape(-1)))))
 requests=[dict(stage='preflight',state_id=sid,label=n) for sid in FOCUS for n in ('zero','plus_one')]
 for sid in FOCUS:requests.append(dict(stage='AD',state_id=sid,label='zero'));requests.extend(dict(stage='finite',state_id=sid,label=n) for n in LABELS);requests.extend(dict(stage='repeat',state_id=sid,label=n) for n in ('zero','plus_one'))
 np.savez_compressed(out/'FROZEN_CHAIN_CONSTANTS.npz',**c);np.savez_compressed(out/'ALL14_FP64_CHAIN_POINTS.npz',**{sid+'__'+n:p for (sid,n),p in points.items()});np.savez_compressed(out/'TWO_PHYSICAL_CHAIN_TANGENTS.npz',**tangents);dump(out/'POINT_BINDINGS_AND_PATH_DIFFERENCES.json',rows);dump(out/'REQUESTS_FROZEN.json',requests)
 names=['DESIGN_FROZEN_BEFORE_MODEL.json','FROZEN_CHAIN_CONSTANTS.npz','ALL14_FP64_CHAIN_POINTS.npz','TWO_PHYSICAL_CHAIN_TANGENTS.npz','POINT_BINDINGS_AND_PATH_DIFFERENCES.json','REQUESTS_FROZEN.json'];seals={n:sha(out/n) for n in names};dump(out/'SEALED_BEFORE_MODEL.json',dict(artifact_seals=seals,model_forwards_so_far=0,torch_map_recomputed_before_model=True));seals['SEALED_BEFORE_MODEL.json']=sha(out/'SEALED_BEFORE_MODEL.json');return points,tangents,requests,seals
