"""Full C36 verification, immutable evidence copy and minimal frozen execution context."""
from c37_protocol import *
from c36_protocol import SETTINGS as C36_SETTINGS
def probe(z,root,full=False):
 require('identity.json' in z.namelist(),'Choose C36 RESULTS ZIP, not source ZIP')
 i=zread(z,'identity.json');s=zread(z,'SUMMARY.json')
 require(i=={'protocol':PARENT_PROTOCOL,'settings':C36_SETTINGS,'source_hashes':read(Path(root)/'C36_SOURCE_PINS.json')},'C36 FIX1 source/settings mismatch')
 require(s['protocol']==PARENT_PROTOCOL and s['complete'] and s['decision']==SETTINGS['parent_expected_decision'],'Completed passing C36 required')
 require(hashlib.sha256(z.read('C36_FROZEN.json')).hexdigest()==s['C36_frozen_sha256']==SETTINGS['expected_parent_freeze_sha256'],'Not the recorded C36 freeze')
 require(hashlib.sha256(z.read('c35/COEFFICIENTS_FROZEN.json')).hexdigest()==s['C35_coefficients_frozen_sha256']==SETTINGS['expected_writer_freeze_sha256'],'C35 writer identity differs')
 require(hashlib.sha256(z.read('DEPLOYED_WRITER.json')).hexdigest()==s['deployed_writer_identity_sha256'],'C36 deployed tensor record differs')
 require(s['branch_complete_acceptance']=={'geometry6':True,'selected6':True} and s['new_training_steps']==0,'C36 scope differs')
 if full:
  for n,h in i['source_hashes'].items():require(hashlib.sha256(z.read('source/'+n)).hexdigest()==h,'C36 source member differs: '+n)
 return s

def prepare(path,out,root):
 out=Path(out);pr=out/'parent_c36';pr.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(path) as z:
  m=verify(z);s=probe(z,root,True);copied={}
  for n,h in m.items():
   if n.startswith('source/'):continue
   p=pr/n;p.parent.mkdir(parents=True,exist_ok=True)
   with z.open(n) as a,p.open('wb') as b:shutil.copyfileobj(a,b,8<<20)
   require(sha(p)==h,'Parent copy changed');copied[n]=h
  dump(out/'PARENT_MANIFEST.json',m)
 dump(out/'PARENT_PROVENANCE.json',{'parent_zip_sha256':sha(path),'members':copied,'manifest_sha256':sha(out/'PARENT_MANIFEST.json'),'parent_modified':False,'parent_zip_not_embedded':True})
 setup_context(out)
 validate(out,root);dump(out/'PREPARED.json',{'complete':True,'parent_decision':s['decision']})

def setup_context(out):
 out=Path(out);src=out/'parent_c36/c35';dest=out/'c35'
 names=['COEFFICIENTS_FROZEN.json','PARENT_PROVENANCE.json','WRITER_FROZEN.json','FIT_RUNTIME.json','RUNTIME.json','parent/C34_BASES_FROZEN.json']
 names+=list(read(src/'COEFFICIENTS_FROZEN.json')['files'])+list(read(src/'parent/C34_BASES_FROZEN.json')['files'])
 names+=['parent/'+n for n in ('MODEL_PINS.json','C25_FINAL_CHECKPOINT.pt','C9_CHECKPOINT.pt','C27_BRANCH.pt','C27_FEATURE_MAP.pt')]
 copied={}
 for n in sorted(set(names)):
  if n=='RUNTIME.json' and not (src/n).exists():continue
  p=dest/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src/n,p);copied[n]=sha(p)
 dump(out/'CONTEXT_MEMBERS.json',copied)

def validate(out,root):
 out=Path(out);p=read(out/'PARENT_PROVENANCE.json');m=read(out/'PARENT_MANIFEST.json')
 require(sha(out/'PARENT_MANIFEST.json')==p['manifest_sha256'],'Parent manifest changed')
 require(p['members']=={n:h for n,h in m.items() if not n.startswith('source/')},'Parent member coverage changed')
 for n,h in p['members'].items():require(sha(out/'parent_c36'/n)==h,'Original C36 evidence changed: '+n)
 require(read(out/'parent_c36/identity.json')=={'protocol':PARENT_PROTOCOL,'settings':C36_SETTINGS,'source_hashes':read(Path(root)/'C36_SOURCE_PINS.json')},'Imported C36 identity differs')
 require(sha(out/'parent_c36/C36_FROZEN.json')==SETTINGS['expected_parent_freeze_sha256'],'C36 freeze changed')
 for n,h in read(out/'CONTEXT_MEMBERS.json').items():require(sha(out/'c35'/n)==h==sha(out/'parent_c36/c35'/n),'Execution context differs: '+n)
 require(sha(out/'c35/COEFFICIENTS_FROZEN.json')==SETTINGS['expected_writer_freeze_sha256'],'Frozen C35 differs')
 return out/'c35'

def resume_score(path,root):
 with zipfile.ZipFile(path) as z:
  verify(z,True);require(zread(z,'identity.json')==identity(root),'Other C37 source/settings')
  require(zread(z,'PREPARED.json')['complete'],'Incomplete parent import')
  ns=z.namelist();return (int('SUMMARY.json' in ns and zread(z,'SUMMARY.json')['complete']),int('C37_FROZEN.json' in ns),sum(n.startswith('c35/results/') and '__c37_new' in n and n.endswith('/COMPLETE.json') for n in ns))
