"""Pure NumPy C161 producer for audit tests; no measured-model evidence."""
import copy,io,shutil,tempfile,zipfile
from core162 import *
from fixture161 import context as simple_context
from baseline158 import record
from worker161 import run as produce161
from fixture156 import pack
_TEMP=[]
def context():
 ctx=simple_context();fp32=ctx['baseline_logits']['FP32_logits_cast64'];fp64=fp32.copy();fp64[:,:6]+=np.arange(6)*1e-7;ctx['baseline_logits']['FP64_logits']=fp64;ctx['baseline']['FP64_response_reference']=fp64[:,:6].mean(0).tolist();ctx['modelpins']=dict(response_ids=list(range(6)));return ctx
class Analytic:
 def __init__(self,out,ctx,c,points,native,coefs,calls):
  self.c=c;self.ctx=ctx;self.calls=calls;calls['model_forwards']+=3;calls['functional_FP64_forwards']+=2;self.ids={n:array_digest(a) for n,a in c.items()};out=Path(out);out.mkdir(parents=True,exist_ok=True);first=ctx['baseline_logits']['FP64_logits'];R=dict(current_FP64=first[:,ctx['modelpins']['response_ids']].mean(0),archived_FP64=np.asarray(ctx['baseline']['FP64_response_reference'],float),archived_FP32=np.asarray(ctx['baseline']['FP32_cached_response_reference'],float))
  for name,a,b,role in [('NATIVE_FP32_BASELINE_LOGITS',ctx['baseline_logits']['FP32_logits_cast64'].astype(np.float32),ctx['baseline_logits']['FP32_logits_cast64'],'cross_run_descriptive_only'),('NATIVE_FP32_CACHED_RESPONSE',R['archived_FP32'].astype(np.float32),R['archived_FP32'],'cross_run_descriptive_only'),('FUNCTIONAL_FP64_BASELINE_LOGITS',first,ctx['baseline_logits']['FP64_logits'],'cross_run_descriptive_only'),('WITHIN_RUN_FP64_ZERO_BASELINE_REPEAT',first,first,'within_run_repeat')]:record(out,name,a,b,role)
  dump(out/'REFERENCES_FROZEN_BEFORE_FOCUS.json',dict(protocol=parent.PROTOCOL,references={n:a.tolist() for n,a in R.items()},reference_tensor_sha256={n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in R.items()},model_forwards_so_far=3,state_probes_so_far=0))
 def observed(self,sid,point):
  x,y=int(sid[4]),int(sid[5]);return np.r_[point[:6]-np.asarray(self.ctx['baseline']['FP32_cached_response_reference']),.1*(2*np.tile([x,y,1,x^y],4)-1)]
 def coefficient_record(self,sid,p,expected,gradient):
  self.calls['model_forwards']+=2;self.calls['functional_FP64_forwards']+=2;pt,e=parent.mapped(self.c,p,sid);exact(pt,expected,'Synthetic fixed map');o=self.observed(sid,pt);parts=parent.loss(o,0.,pt,self.c[sid+'__prepared'],e,sid.split('__')[0]);row=dict(state_id=sid,case=sid.split('__')[0],parameters_digest=parameter_hash(p),FP64_point_sha256=point_hash(pt),sealed_numpy_point_sha256=point_hash(expected),map_Linf=0.,observed=o.tolist(),KL=0.,objective_parts=parts,FP64_acceptance_evaluated=False)
  if gradient:
   u=self.c['V']@(self.c[sid+'__raw']-self.c[sid+'__prepared']);a=o[:6];top=abs(a)==max(abs(a));factor=2*max(max(abs(a))/.0018-1,0)/.0018;dout=factor*np.sign(a)*top/top.sum();V=self.c['V'];ge=np.stack([(V[:,:6]@dout)*u,(V[:,:6]@dout)*np.roll(u,1)]);guard=2*.001*e/e.shape[-1];norm=np.linalg.norm(pt-self.c[sid+'__prepared'])
   if norm>20:
    dp=2*(norm/20-1)/(20*norm)*(pt-self.c[sid+'__prepared']);guard+=np.stack([(V@dp)*u,(V@dp)*np.roll(u,1)])
   w=self.c[sid+'__weights'];row['response_gradient']=(w[:,None,None]*ge[None]).tolist();row['guard_gradient']=(w[:,None,None]*guard[None]).tolist();self.calls['backward_calls']+=2;self.calls['backward_attempts']+=2
  return row
 def native_candidate(self,sid,p,digest):
  compare(safe.tensor_hash(p),digest);self.calls['model_forwards']+=2;o=self.observed(sid,p.astype(float));m=archive155.full_metric(o,0.,p,sid.split('__')[0]);q=archive155.endpoint_gates(m,sid.split('__')[0],p,self.c[sid+'__prepared'].astype(np.float32),True);return dict(state_id=sid,case=sid.split('__')[0],metrics=m,qualification=q,actual_physical_FP32_commit=False,synthetic_fixture=True)
 def verify(self,*args):compare({n:array_digest(a) for n,a in self.c.items()},self.ids,'Synthetic fixed inputs')
def produce(ctx,paths,pc,out):
 out=Path(out)
 with contextlib.redirect_stdout(io.StringIO()):s=produce161(paths,out,pc,Analytic,lambda *a:copy.deepcopy(ctx))
 need(s['complete'],'Synthetic C161 complete: '+str(s['errors']));shutil.copytree(ROOT/'vendor161',out/'source',ignore=shutil.ignore_patterns('__pycache__'));dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});z=out.parent/'C161.zip';pack(out,z);return z
def fixture(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);ctx=context();paths={}
 for k in parent.INPUT_KEYS:p=root/(k+'.zip');p.write_bytes(b'Synthetic injected parent binding marker');paths[k]=p
 pc=parent.contract();pc.update(production=False,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});z=produce(ctx,paths,pc,root/'C161');paths['C161']=z;c=contract();c.update(production=False,parent=pc,pins={k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()});return paths,c,ctx
def changed(original,output,name,change,rebind=False):
 with zipfile.ZipFile(original) as z:members={n:z.read(n) for n in z.namelist()}
 members[name]=change(members[name])
 if rebind:
  for key,raw in list(members.items()):
   if key.endswith('/SEALED.json') or key=='SEALED_BEFORE_MODEL.json':
    r=clean_json(raw);changed=False
    for n in r['artifact_seals']:
     full=(key.rsplit('/',1)[0]+'/' if '/' in key else '')+n
     if full==name:r['artifact_seals'][n]=hashlib.sha256(members[name]).hexdigest();changed=True
    if changed:members[key]=(json.dumps(r)+'\n').encode()
  # The whole result manifest is always rebound, to exercise independent checks.
 m={n:hashlib.sha256(v).hexdigest() for n,v in members.items() if n!='RESULT_MANIFEST.json'};members['RESULT_MANIFEST.json']=(json.dumps(m)+'\n').encode()
 with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
  for n,v in members.items():z.writestr(n,v)
 return output
def json_change(fn):
 def edit(raw):a=clean_json(raw);fn(a);return (json.dumps(a,allow_nan=False)+'\n').encode()
 return edit
def array_change(fn):
 def edit(raw):
  with np.load(io.BytesIO(raw),allow_pickle=False) as z:a={n:z[n] for n in z.files}
  fn(a);f=io.BytesIO();np.savez_compressed(f,**a);return f.getvalue()
 return edit
