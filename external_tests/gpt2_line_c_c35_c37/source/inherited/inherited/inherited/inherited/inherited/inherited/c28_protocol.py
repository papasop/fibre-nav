import hashlib,json,os,time,zipfile,shutil
from pathlib import Path
from archive_c26 import require,read_json,file_sha,verify_archive,complete,replay_equal,HEAD_SHA,CASES,STARTS,GATES
from c27_protocol import dump,metrics_counts,REVISION,PARENT_CP_SHA
PROTOCOL='GPT2_C28_RESPONSE_AWARE_FOUR_ROUTE_A100_V1'
RESULT='gpt2_c28_response_aware_four_route_results'
DRIVE_NAME=PROTOCOL+'_RECOVERY'
PARENT_PROTOCOL='GPT2_C27_STATE_RESIDUAL_COMPARISON_A100_V1'
PARENT_MATRIX_SHA='d0e605744ece637c71cc26000d2b67435069a6c3c42158cfda5572bae9bbc8da'
PARENT_BRANCH_CP_SHA='def063964cf067532c1244abc6b376768eedf3a5cc754c5e88a67b79f7155666'
ARMS=('pca_control','response_aware')
SETTINGS={'training_states':76,'states_per_route':19,'inspected_development_states':16,'routes':list(CASES),
 'components':8,'features_plus_bias':9,'response_indices':list(range(3077,3083)),'scale_floor':.001,
 'response_budget':.002,'kl_budget':.005,'margin_gate':.05,'fit_uses_only_training':True,
 'ridge_grid':[1e-6,1e-8],'increment_radius_grid':[.05,.1,.2],'alpha_grid':[1.,.5,.25],
 'target_response_interval':.0012,'layer_radius':.5,'max_rounds_per_route':1,'n_specs_per_route':18,
 'interval_solver_max_iterations':8192,'interval_solver_tolerance':1e-10,'interval_solver_polish_iterations':128,
 'output_scale':.07025910168886185,'parent_training_passes_protected':76,'archive_every_candidates':6,
 'geometry':'Fresh offline response Jacobians at the fixed C27 writer endpoints on all76 training states; no development-state geometry',
 'cached_preparation_features_used':True,'online_proposals':1,'online_jacobians':0,'online_SVD':0,'online_correction':0,'online_backtracking':0,
 'teacher_fallback':False,'fresh_evaluation':False,'new_state_generation':False,'requested_output_bit_supplied':True,'internal_parameter_update':False,
 'scope':'Four-route development comparison of training-only PCA8 versus six declared pre-write response features plus two training-only PCA features. Frozen C27 base. Existing16 inspected C25 states evaluated only after both arms freeze; no fresh-generalisation or LC1-closure claim.'}

def source_hashes(root):return {p.relative_to(root).as_posix():file_sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
def check_inherited(root):
 pins=json.loads((Path(root)/'C27_SOURCE_PINS.json').read_text())
 for n,h in pins.items():require(file_sha(Path(root)/'inherited'/n)==h,'Inherited C27 source changed: '+n)
 return pins

def specs():
 return [{'id':'candidate_%03d'%(a*6+b*3+c),'radius_index':a,'ridge_index':b,'alpha_index':c,'increment_radius':r,'ridge':l,'alpha':v}
  for a,r in enumerate(SETTINGS['increment_radius_grid']) for b,l in enumerate(SETTINGS['ridge_grid']) for c,v in enumerate(SETTINGS['alpha_grid'])]

def route_guard(before,after,ids):
 require(set(before)==set(after)==set(ids),'Route denominator differs')
 bc=metrics_counts(before);ac=metrics_counts(after)
 bscore=sum((r['metrics']['drift']/.002)**2 for r in before.values());ascore=sum((r['metrics']['drift']/.002)**2 for r in after.values())
 reasons=[]
 if bc['valid']!=len(ids) or ac['valid']!=len(ids):reasons.append('INVALID_REFERENCE')
 if bc['passed']!=len(ids) or ac['passed']!=len(ids):reasons.append('LOST_COMPLETE_PASS')
 if ac['maximum_R']>bc['maximum_R']+1e-12:reasons.append('WORST_RESPONSE_INCREASED')
 if ascore>=bscore-1e-10:reasons.append('NO_RESPONSE_SCORE_IMPROVEMENT')
 return {'accepted':not reasons,'reasons':reasons,'before':bc,'after':ac,'before_response_score':bscore,'after_response_score':ascore}

def parent_probe(z,root,check_source=True):
 i=read_json(z,'identity.json');s=read_json(z,'SUMMARY.json');fr=read_json(z,'arms/state_bypass/FROZEN.json')
 require(i['protocol']==s['protocol']==PARENT_PROTOCOL and s['complete'],'Completed C27 result required')
 require(i['source_hashes']==json.loads((Path(root)/'C27_SOURCE_PINS.json').read_text()),'Exact C27 FIX1 parent source required')
 if check_source:
  for n,h in i['source_hashes'].items():require(file_sha_bytes(z.read('source/'+n))==h,'Archived C27 source bytes differ: '+n)
 require(s['base_head_sha256']==HEAD_SHA and s['base_head_unchanged'] and s['all_initial74_passes_preserved'],'C27 base identity/preservation differs')
 require(s['training']['state_bypass']['passed']==76 and s['training']['state_bypass']['valid']==76,'C27 76/76 reference required')
 require(fr['accepted_refinement'] and fr['matrix_sha256']==PARENT_MATRIX_SHA and fr['checkpoint_sha256']==PARENT_BRANCH_CP_SHA,'Actual C27 residual identity differs')
 require(file_sha_bytes(z.read('arms/state_bypass/FINAL_CHECKPOINT.pt'))==PARENT_BRANCH_CP_SHA,'Actual C27 branch checkpoint digest differs')
 require(file_sha_bytes(z.read('arms/state_bypass/FEATURE_MAP.pt'))==fr['feature_map_sha256'],'C27 feature map digest differs')
 return i,s,fr

def file_sha_bytes(b):return hashlib.sha256(b).hexdigest()

def prepare_parent(path,out,root):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);provenance={}
 with zipfile.ZipFile(path) as z:
  manifest=verify_archive(z);i,s,fr=parent_probe(z,root)
  def cp(old,new):
   p=out/new;p.parent.mkdir(parents=True,exist_ok=True)
   with z.open(old) as src,p.open('wb') as dest:shutil.copyfileobj(src,dest,8<<20)
   provenance[new]={'member':old,'sha256':file_sha(p)}
  for old,new in [('parent/C25_FINAL_CHECKPOINT.pt','parent/C25_FINAL_CHECKPOINT.pt'),('parent/C9_CHECKPOINT.pt','parent/C9_CHECKPOINT.pt'),('parent/MODEL_PINS.json','parent/MODEL_PINS.json'),('parent/SPLIT.json','parent/SPLIT.json'),('SUMMARY.json','parent/C27_SUMMARY.json'),('arms/state_bypass/FINAL_CHECKPOINT.pt','parent/C27_BRANCH.pt'),('arms/state_bypass/FEATURE_MAP.pt','parent/C27_FEATURE_MAP.pt'),('arms/state_bypass/FROZEN.json','parent/C27_FROZEN.json')]:cp(old,new)
  split=read_json(z,'parent/SPLIT.json')
  for cohort,ss in [('training',split['training']),('inspected',split['prospective_new_states'])]:
   for t in ss:
    sid=t['id'];b='data/'+cohort+'/'+sid+'/'
    for n in ('prepared.pt','INPUT.pt','REFERENCE.json'):cp(b+n,b+n)
    if cohort=='inspected':cp('evaluation/'+sid+'/COMPLETE.json',b+'C27_COMPLETE.json')
  dump(out/'PARENT_PROVENANCE.json',{'parent_zip_sha256':file_sha(path),'parent_protocol':PARENT_PROTOCOL,'verified_members':len(manifest),'members':provenance,'full_parent_embedded':False})
 dump(out/'PREPARED.json',{'complete':True,'training':76,'inspected':16,'C27_matrix_sha256':PARENT_MATRIX_SHA})
 print('C28 immutable compact C27 import complete; no old history embedded.',flush=True)

def snapshot(out,dd=''):
 out=Path(out);result=out.parent/(RESULT+'.zip');temp=result.with_suffix('.zip.tmp')
 files={p.relative_to(out).as_posix():file_sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'MANIFEST.json' and '__pycache__' not in p.parts and not p.name.endswith('.tmp')}
 dump(out/'MANIFEST.json',files)
 with zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for n in files:z.write(out/n,n)
  z.write(out/'MANIFEST.json','MANIFEST.json')
 with zipfile.ZipFile(temp) as z:verify_archive(z,heartbeat=lambda *a,**k:None)
 os.replace(temp,result)
 if dd:
  dest=Path(dd);dest.mkdir(parents=True,exist_ok=True);target=dest/(RESULT+'_'+str(time.time_ns())+'.zip');tmp=target.with_suffix('.zip.tmp');shutil.copyfile(result,tmp);require(file_sha(tmp)==file_sha(result),'Drive copy differs');os.replace(tmp,target)
  valid=[]
  for p in sorted(dest.glob(RESULT+'_*.zip'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
   try:
    with zipfile.ZipFile(p) as z:verify_archive(z,heartbeat=lambda *a,**k:None);require(read_json(z,'identity.json')['protocol']==PROTOCOL,'Wrong protocol')
    valid.append(p)
   except Exception:continue
  for p in valid[3:]:p.unlink()
  print('DRIVE_BACKUP_OK',target,flush=True)
 print('RESULT_ARCHIVE='+str(result),flush=True);return result

def resume_score(path,root):
 with zipfile.ZipFile(path) as z:
  verify_archive(z,heartbeat=lambda *a,**k:None);i=read_json(z,'identity.json');require(i['protocol']==PROTOCOL and i['settings']==SETTINGS and i['source_hashes']==source_hashes(root),'C28 source/settings differ')
  names=z.namelist();return (int('SUMMARY.json' in names),int('WRITERS_FROZEN.json' in names),sum(n.startswith('evaluation/') and n.endswith('/COMPLETE.json') for n in names),sum(n.startswith('arms/') and n.endswith('/FROZEN.json') for n in names),sum('/candidates/' in n and n.endswith('/COMPLETE.json') for n in names),sum(n.startswith('geometry/') and n.endswith('.pt') for n in names))
