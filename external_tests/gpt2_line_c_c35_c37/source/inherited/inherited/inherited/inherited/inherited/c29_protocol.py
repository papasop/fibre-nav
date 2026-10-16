import hashlib,json,os,time,zipfile,shutil
from pathlib import Path
from archive_c26 import require,read_json,file_sha,verify_archive,complete,replay_equal,HEAD_SHA,CASES,STARTS,GATES
from c27_protocol import dump,metrics_counts,REVISION,PARENT_CP_SHA
from c28_protocol import PARENT_MATRIX_SHA,PARENT_BRANCH_CP_SHA
PROTOCOL='GPT2_C29_FROZEN_DIRECTION_DIAGNOSIS_A100_V1'
RESULT='gpt2_c29_frozen_direction_diagnosis_results'
DRIVE_NAME=PROTOCOL+'_RECOVERY'
PARENT_PROTOCOL='GPT2_C28_RESPONSE_AWARE_FOUR_ROUTE_A100_V1'
ARMS=('pca_control','response_aware')
PINS={'pca_control':{'checkpoint_sha256':'55df1663935c4b7d4c67f55f36e31d3f7f1e3c6607a6bbea45455d45c87463a4','matrices_sha256':'974d6e12834789aaf96b3b5507332908392665b44b6f66d88ad6b19a286f5aa3'},'response_aware':{'checkpoint_sha256':'789e1bf36a28f0a04a85f421f585d820496ad053d5af78a99a9d01e23539a269','matrices_sha256':'39c80d954f528afd41bc56504be99cc842d0ab0816651f906ecedfe78e8fda4a'}}
SETTINGS={'training_states':76,'inspected_development_states':16,'output_scale':.07025910168886185,
 'response_budget':.002,'kl_budget':.005,'margin_gate':.05,'metric_comparison_tolerance':2e-6,'diagnostic_change_tolerance':1e-6,
 'diagnostic_half_step':.5,'half_step_cohort':'all16 inspected states, both fixed C28 arms','half_step_is_not_selected_or_accepted':True,
 'training_Jacobians_reused':76,'new_Jacobian_cohort':'16 already inspected C25 development states, evaluated at the fixed C27 baseline endpoints',
 'all_writers_frozen':True,'new_head_training_steps':0,'new_candidates':0,'new_heads_committed':0,
 'fresh_evaluation':False,'new_state_generation':False,'internal_parameter_update':False,'controller_dispatch_tested':False,
 'archive_every_completed_states':8,'scope':'External diagnostic of fixed C27/C28 endpoints on76 training and16 inspected states. Actual float32 parameter displacements, float64 J-delta arithmetic and one predeclared midpoint probe. No refitting, candidate selection, correction, new writer or LC1 acceptance claim.'}

def source_hashes(root):return {p.relative_to(root).as_posix():file_sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
def check_inherited(root):
 pins=json.loads((Path(root)/'C28_SOURCE_PINS.json').read_text())
 for n,h in pins.items():require(file_sha(Path(root)/'inherited'/n)==h,'Inherited C28 source differs: '+n)
 return pins

def parent_probe(z,root,check_source=True):
 i=read_json(z,'identity.json');s=read_json(z,'SUMMARY.json');pins=json.loads((Path(root)/'C28_SOURCE_PINS.json').read_text())
 require(i['protocol']==s['protocol']==PARENT_PROTOCOL and s['complete'],'Completed C28 result required')
 require(i['source_hashes']==pins,'Exact C28 source required')
 if check_source:
  for n,h in pins.items():require(hashlib.sha256(z.read('source/'+n)).hexdigest()==h,'Archived source differs: '+n)
 require(s['parent_C27_matrix_sha256']==PARENT_MATRIX_SHA and s['parent_C27_unchanged'] and s['all76_training_passes_preserved'],'C28 fixed baseline differs')
 for arm in ARMS:
  fr=read_json(z,'arms/'+arm+'/FROZEN.json')
  require(all(fr[k]==v for k,v in PINS[arm].items()),'Recorded frozen arm differs: '+arm)
  require(hashlib.sha256(z.read('arms/'+arm+'/FINAL_CHECKPOINT.pt')).hexdigest()==PINS[arm]['checkpoint_sha256'],'Actual arm checkpoint differs')
  require(s['training'][arm]['passed']==76 and s['training'][arm]['valid']==76,'C28 training denominator differs')
 return i,s

def prepare_parent(path,out,root):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);provenance={}
 with zipfile.ZipFile(path) as z:
  manifest=verify_archive(z);i,s=parent_probe(z,root)
  def cp(old,new):
   p=out/new;p.parent.mkdir(parents=True,exist_ok=True)
   with z.open(old) as src,p.open('wb') as dest:shutil.copyfileobj(src,dest,8<<20)
   provenance[new]={'member':old,'sha256':file_sha(p)}
  for n in ('C25_FINAL_CHECKPOINT.pt','C9_CHECKPOINT.pt','MODEL_PINS.json','SPLIT.json','C27_BRANCH.pt','C27_FEATURE_MAP.pt','C27_FROZEN.json'):cp('parent/'+n,'parent/'+n)
  cp('SUMMARY.json','parent/C28_SUMMARY.json');cp('BASELINE_REPLAY.json','parent/C28_BASELINE_REPLAY.json')
  split=read_json(z,'parent/SPLIT.json')
  for cohort,ts in [('training',split['training']),('inspected',split['prospective_new_states'])]:
   for t in ts:
    sid=t['id'];b='data/'+cohort+'/'+sid+'/'
    for n in ('prepared.pt','INPUT.pt','REFERENCE.json'):cp(b+n,b+n)
    if cohort=='training':cp('geometry/'+sid+'.pt','geometry/'+sid+'.pt')
    else:cp('evaluation/'+sid+'/COMPLETE.json',b+'C28_COMPLETE.json')
  for arm in ARMS:
   for n in ('FINAL_CHECKPOINT.pt','FROZEN.json'):cp('arms/'+arm+'/'+n,'parent/'+arm+'/'+n)
   for case in CASES:cp('arms/'+arm+'/'+case+'/FEATURE_MAP.pt','parent/'+arm+'/'+case+'/FEATURE_MAP.pt')
  dump(out/'PARENT_PROVENANCE.json',{'parent_zip_sha256':file_sha(path),'verified_members':len(manifest),'members':provenance,'full_parent_embedded':False})
 dump(out/'PREPARED.json',{'complete':True,'training':76,'inspected':16,'training_geometry_reused':76});print('C29 compact immutable C28 data verified.',flush=True)

def snapshot(out,dd=''):
 out=Path(out);result=out.parent/(RESULT+'.zip');tmp=result.with_suffix('.zip.tmp')
 m={p.relative_to(out).as_posix():file_sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'MANIFEST.json' and '__pycache__' not in p.parts and not p.name.endswith('.tmp')}
 dump(out/'MANIFEST.json',m)
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for n in m:z.write(out/n,n)
  z.write(out/'MANIFEST.json','MANIFEST.json')
 with zipfile.ZipFile(tmp) as z:verify_archive(z,heartbeat=lambda *a,**k:None)
 os.replace(tmp,result)
 if dd:
  d=Path(dd);d.mkdir(parents=True,exist_ok=True);dest=d/(RESULT+'_'+str(time.time_ns())+'.zip');t=dest.with_suffix('.zip.tmp');shutil.copyfile(result,t);require(file_sha(t)==file_sha(result),'Drive copy differs');os.replace(t,dest)
  valid=[]
  for p in sorted(d.glob(RESULT+'_*.zip'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
   try:
    with zipfile.ZipFile(p) as z:verify_archive(z,heartbeat=lambda *a,**k:None);require(read_json(z,'identity.json')['protocol']==PROTOCOL,'Wrong protocol')
    valid.append(p)
   except Exception:continue
  for p in valid[3:]:p.unlink()
  print('DRIVE_BACKUP_OK',dest,flush=True)
 print('RESULT_ARCHIVE='+str(result),flush=True);return result

def resume_score(path,root):
 with zipfile.ZipFile(path) as z:
  verify_archive(z,heartbeat=lambda *a,**k:None);i=read_json(z,'identity.json');require(i['protocol']==PROTOCOL and i['settings']==SETTINGS and i['source_hashes']==source_hashes(root),'C29 resume source/settings differ');ns=z.namelist()
  return (int('SUMMARY.json' in ns),sum(n.startswith('diagnosis/') and n.endswith('/COMPLETE.json') for n in ns),sum(n.startswith('geometry/inspected/') and n.endswith('.pt') for n in ns))
