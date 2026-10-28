"""Complete C165 analytic NumPy fixture, never real-model evidence."""
import copy,io,shutil
from core166 import *
from fixture164 import fixture as fixture163,Analytic163,pack,changed,json_change,array_change
from audit164 import audit as audit163
from worker165 import run as produce165
class Analytic165(Analytic163):
 def __init__(self,*args):
  super().__init__(*args);out=Path(args[0]);r=clean_json((out/'REFERENCES_FROZEN_BEFORE_FOCUS.json').read_bytes());r['protocol']=parent.PROTOCOL;dump(out/'REFERENCES_FROZEN_BEFORE_FOCUS.json',r);self.registered={}
 def preseal(self,p,points,folder,tag):
  r=self.prepare_batch(p,points,folder,tag);self.registered[Path(folder).name]=Path(folder);return r
 def activate(self,name,p):self.batch=self.registered[name]

def make165(ctx,paths,c164,out,ctx_is_C163_final=False):
 out=Path(out)
 if not ctx_is_C163_final:
  with contextlib.redirect_stdout(io.StringIO()):a=audit163(paths['C163'],ctx,out.parent/'C163_audit_for_fixture',c164)
  ctx=copy.deepcopy(ctx);ctx.update(initial_head=a['heads'][8],C163_audit=a,parent_final_FP64_records=a['batches'][-1],parent_final_native_records={str(t):a['natives']['native_final'+str(t)] for t in (0,1)},C164_receipt=dict(decision='C164_READONLY_TRAINING_AUDIT_COMPLETE_PARENT_DECISION_RETAINED'))
 else:ctx=copy.deepcopy(ctx)
 paths=dict(paths)
 if 'C164' not in paths:p=out.parent/'C164_marker.zip';p.write_bytes(b'Standalone synthetic injected original');paths['C164']=p
 cc=parent.contract();cc.update(production=False,parent=c164,pins={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()})
 with contextlib.redirect_stdout(io.StringIO()):s=produce165(paths,out,cc,Analytic165,lambda *a:copy.deepcopy(ctx))
 need(s['complete'],'Synthetic C165 complete: '+str(s['errors']));shutil.copytree(ROOT/'vendor165',out/'source',ignore=shutil.ignore_patterns('__pycache__'));dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});p=out.parent/'C165.zip';pack(out,p);paths['C165']=p;c=contract();c.update(production=False,parent=cc,pins={k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});return paths,c,ctx

def fixture(root):
 root=Path(root);root.mkdir(parents=True,exist_ok=True);paths,pc,ctx=fixture163(root/'C163_fixture');return make165(ctx,paths,pc,root/'C165')

# Rebind all C165 receipt duplicates, so negative tests reach independent arithmetic.
def changed(original,output,name,fn,rebind=True):
 import zipfile
 with zipfile.ZipFile(original) as z:members={n:z.read(n) for n in z.namelist()}
 members[name]=fn(members[name])
 if rebind:
  for key,raw in list(members.items()):
   if key.endswith('/SEALED.json') or key=='SEALED_BEFORE_MODEL.json' or key.endswith('/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json'):
    r=clean_json(raw);prefix=key.rsplit('/',1)[0]+'/' if '/' in key else ''
    for n in r['artifact_seals']:
     full=prefix+n
     if full in members:r['artifact_seals'][n]=hashlib.sha256(members[full]).hexdigest()
    members[key]=(json.dumps(r)+'\n').encode()
  key='ALL8_ACTUAL_FP64_POINT_SEALS.json';rr=clean_json(members[key]);rr=[dict(folder=r['folder'],**clean_json(members[r['folder']+'/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json']),receipt_sha256=hashlib.sha256(members[r['folder']+'/ACTUAL_FP64_POINTS_SEALED_BEFORE_MEASUREMENTS.json']).hexdigest()) for r in rr];members[key]=(json.dumps(rr)+'\n').encode();key='ALL6_PROBES_SEALED_BEFORE_FIRST_PROBE.json';r=clean_json(members[key]);r['actual_point_seals']=rr
  for folder in r['head_seals']:r['head_seals'][folder]=clean_json(members[folder+'/SEALED.json'])
  for n in r['direction_artifact_seals']:r['direction_artifact_seals'][n]=hashlib.sha256(members[n]).hexdigest()
  members[key]=(json.dumps(r)+'\n').encode()
 m={n:hashlib.sha256(v).hexdigest() for n,v in members.items() if n!='RESULT_MANIFEST.json'};members['RESULT_MANIFEST.json']=(json.dumps(m)+'\n').encode()
 with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
  for n,v in members.items():z.writestr(n,v)
 return output
