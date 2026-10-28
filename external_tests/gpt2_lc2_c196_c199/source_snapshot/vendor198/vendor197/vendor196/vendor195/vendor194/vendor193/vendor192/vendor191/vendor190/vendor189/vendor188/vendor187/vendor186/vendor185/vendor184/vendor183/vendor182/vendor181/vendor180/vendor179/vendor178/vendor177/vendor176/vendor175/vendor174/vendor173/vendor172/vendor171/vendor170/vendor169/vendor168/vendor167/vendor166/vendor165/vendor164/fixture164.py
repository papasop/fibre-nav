"""Analytic NumPy fixtures only; never scientific model evidence."""
import copy,io,shutil,zipfile
from core164 import *
from fixture162 import Analytic,fixture as fixture161,pack,json_change,array_change
from audit162 import audit as audit161
from worker163 import run as produce163

class Analytic163(Analytic):
 def __init__(self,*args):
  super().__init__(*args);out=Path(args[0]);r=clean_json((out/'REFERENCES_FROZEN_BEFORE_FOCUS.json').read_bytes());r['protocol']=parent.PROTOCOL;dump(out/'REFERENCES_FROZEN_BEFORE_FOCUS.json',r);self.point_tensors_saved=0;self.batch=None
 def prepare_batch(self,p,points,folder,tag):
  folder=Path(folder);arrays={sid:points[(sid,'zero')].copy() for sid in IDS};bindings=[dict(state_id=sid,actual_Torch_FP64_sha256=point_hash(arrays[sid]),NumPy_point_sha256=point_hash(arrays[sid]),Torch_vs_NumPy_Linf=0.) for sid in IDS];np.savez_compressed(folder/'ACTUAL_TORCH_FP64_POINTS.npz',**arrays);dump(folder/'ACTUAL_TORCH_FP64_POINT_BINDINGS.json',bindings);receipt=dict(tag=tag,head_digest=parameter_hash(p),all84_state_ids=IDS,model_forwards_so_far=self.calls['model_forwards'],backward_calls_so_far=self.calls['backward_calls'],sealed_before_this_batch=True,artifact_seals={n:sha(folder/n) for n in ('HEAD.npz','POINT_BINDINGS.json','ACTUAL_TORCH_FP64_POINTS.npz','ACTUAL_TORCH_FP64_POINT_BINDINGS.json')});dump(folder/'ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json',receipt);self.batch=folder;self.point_tensors_saved+=84;return dict(folder=folder.name,**receipt,receipt_sha256=sha(folder/'ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json'))
 def coefficient_record(self,*args):
  r=super().coefficient_record(*args);r['actual_FP64_point_artifact']=self.batch.name+'/ACTUAL_TORCH_FP64_POINTS.npz';return r
 def verify_batch(self,p):pass

def make163(ctx,paths,pc,out):
 out=Path(out);pp=Path(paths['C161'])
 with contextlib.redirect_stdout(io.StringIO()):a=audit161(pp,ctx,out.parent/'audit161',pc)
 ctx=copy.deepcopy(ctx);ctx.update(initial_head=a['heads'][8],C161_audit=a,C162_receipt=dict(decision='C162_READONLY_TRAINING_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'),parent_final_FP64_records=a['batches'][-1],parent_final_native_records={str(t):a['natives']['native_final'+str(t)] for t in (0,1)})
 # pc is a core162 contract (seven inputs); construct core163's eight-input contract.
 paths=dict(paths)
 if 'C162' not in paths:
  marker=out.parent/'C162_marker.zip';marker.write_bytes(b'Standalone synthetic injected C162 binding');paths['C162']=marker
 cc=parent.contract();cc.update(production=False,parent=pc,pins={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()})
 with contextlib.redirect_stdout(io.StringIO()):s=produce163(paths,out,cc,Analytic163,lambda *a:copy.deepcopy(ctx))
 need(s['complete'],'Synthetic C163: '+str(s['errors']));shutil.copytree(ROOT/'vendor163',out/'source',ignore=shutil.ignore_patterns('__pycache__'));dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});p=out.parent/'C163.zip';pack(out,p);paths['C163']=p;c=contract();c.update(production=False,parent=cc,pins={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});return paths,c,ctx

def fixture(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);paths,pc,ctx=fixture161(root/'C161_fixture');return make163(ctx,paths,pc,root/'C163')

def changed(original,output,name,fn,rebind=True):
 with zipfile.ZipFile(original) as z:members={n:z.read(n) for n in z.namelist()}
 members[name]=fn(members[name])
 if rebind:
  # Rebind ordinary receipts, actual batch receipts, and their top-level duplicate.
  for key,raw in list(members.items()):
   if key.endswith('/SEALED.json') or key=='SEALED_BEFORE_MODEL.json' or key.endswith('/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json'):
    r=clean_json(raw);prefix=key.rsplit('/',1)[0]+'/' if '/' in key else ''
    for n in r['artifact_seals']:
     full=prefix+n
     if full in members:r['artifact_seals'][n]=hashlib.sha256(members[full]).hexdigest()
    members[key]=(json.dumps(r)+'\n').encode()
  key='ALL9_ACTUAL_FP64_POINT_SEALS.json'
  if key in members:
   rr=clean_json(members[key]);rr=[dict(folder=r['folder'],**clean_json(members[r['folder']+'/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json']),receipt_sha256=hashlib.sha256(members[r['folder']+'/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json']).hexdigest()) for r in rr];members[key]=(json.dumps(rr)+'\n').encode()
 manifest={n:hashlib.sha256(v).hexdigest() for n,v in members.items() if n!='RESULT_MANIFEST.json'};members['RESULT_MANIFEST.json']=(json.dumps(manifest)+'\n').encode()
 with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
  for n,v in members.items():z.writestr(n,v)
 return output
