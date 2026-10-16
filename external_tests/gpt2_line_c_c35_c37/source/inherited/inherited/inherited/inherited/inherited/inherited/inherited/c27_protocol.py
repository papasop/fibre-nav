"""C27 activation audit and fixed residual-feature comparison, development only."""
import hashlib,json,os,time,zipfile,shutil
from pathlib import Path
from archive_c26 import require,read_json,file_sha,verify_archive,guard,quality,complete,replay_equal,HEAD_SHA,CASES,STARTS,GATES
PROTOCOL='GPT2_C27_STATE_RESIDUAL_COMPARISON_A100_V1'
RESULT='gpt2_c27_state_residual_comparison_results'
PARENT_SHA='6a918540cfe15e229daf17c9c62ce6c29f8d379e3e8e62df9f9a39f7a773137e'
PARENT_CP_SHA='5f09423a93b7c1db9e52cd7b95593aecd36eb4897f8f4657259d0a194f3371a5'
REVISION='607a30d783dfa663caf39e06633721c8d4cfcd7e'
ARMS=('hidden_control','state_bypass')
SETTINGS={'route':'xor_10','feature_components':8,'feature_scale_floor':.001,'fit_feature_maps_on_training_only':True,
 'n_training':76,'n_route_training':19,'n_inspected_development':16,'max_offline_rounds_per_arm':1,
 'ridge_grid':[1e-6,1e-8],'increment_radius_grid':[.05,.10,.20],'alpha_grid':[1.,.5,.25],
 'layer_radius':.5,'target_response_interval':.0018,'response_budget':.002,'kl_budget':.005,'margin_gate':.05,
 'interval_solver_max_iterations':8192,'interval_solver_tolerance':1e-10,'interval_solver_polish_iterations':128,
 'output_scale':.07025910168886185,'archive_every_candidates':5,'feature_components_plus_bias':9,
 'all_candidates_retained':True,'all76_physical_audit_before_commit':True,'all_initial74_passes_protected':True,
 'saved_Jacobians_only':True,'new_model_backward_calls':0,'fresh_evaluation':False,'new_state_generation':False,
 'online_write_Jacobians':0,'online_write_SVD':0,'online_correction':0,'teacher_fallback':False,
 'external_runtime_commit':True,'requested_output_bit_supplied':True,'controller_dispatch_tested':False,'internal_parameter_update':False,
 'scope':'Same frozen C25 GPT2/model/chart, xor10-only residual output branch; fixed 19-state training geometry and76-state guards; original16 C25 inspected development states evaluated only after both arms freeze. No unseen-class, fresh-state, independent or native-update claim.'}


def source_hashes(root):
 return {p.relative_to(root).as_posix():file_sha(p) for p in sorted(Path(root).rglob('*')) if p.is_file() and '__pycache__' not in p.parts}


def check_legacy(root):
 for n,h in json.loads((Path(root)/'C25_SOURCE_PINS.json').read_text()).items():
  require(file_sha(Path(root)/'legacy'/n)==h,'Pinned legacy source changed: '+n)


def dump(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp')
 tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False));os.replace(tmp,path)


def specs():
 return [{'id':'candidate_%03d'%(r*6+j*3+k),'radius_index':r,'ridge_index':j,'alpha_index':k,
          'increment_radius':radius,'ridge':ridge,'alpha':alpha} for r,radius in enumerate(SETTINGS['increment_radius_grid'])
         for j,ridge in enumerate(SETTINGS['ridge_grid']) for k,alpha in enumerate(SETTINGS['alpha_grid'])]


def metrics_counts(rows):
 return {'n':len(rows),'valid':sum(bool(r['reference_valid']) for r in rows.values()),'passed':sum(complete(r) for r in rows.values()),
         'gate_counts':{g:sum(r['metrics']['gates'][g] for r in rows.values()) for g in GATES},
         'maximum_R':max(r['metrics']['drift'] for r in rows.values())}


def decide(train,dev):
 if any(x['valid']!=x['n'] for x in list(train.values())+list(dev.values())):return 'C27_STATE_REFERENCE_FAILURE'
 if train['state_bypass']['passed']!=76:return 'C27_RESIDUAL_TRAINING_INCOMPLETE'
 if dev['state_bypass']['passed']!=16:return 'C27_INSPECTED_STATE_GENERALISATION_FAILURE'
 return 'C27_INSPECTED_STATE_RESIDUAL_PASS'


def snapshot(out,drive_dir=''):
 out=Path(out);session=out.parent;path=session/(RESULT+'.zip');tmp=path.with_suffix('.zip.tmp')
 manifest={p.relative_to(out).as_posix():file_sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='MANIFEST.json' and not p.name.endswith('.tmp')}
 # Nested original source manifests must remain covered.
 for p in out.rglob('MANIFEST.json'):
  if p!=out/'MANIFEST.json':manifest[p.relative_to(out).as_posix()]=file_sha(p)
 dump(out/'MANIFEST.json',manifest)
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for n in sorted(manifest):z.write(out/n,n)
  z.write(out/'MANIFEST.json','MANIFEST.json')
 with zipfile.ZipFile(tmp) as z:verify_archive(z,heartbeat=lambda *a,**k:None)
 os.replace(tmp,path)
 if drive_dir:
  dd=Path(drive_dir);dd.mkdir(parents=True,exist_ok=True);dst=dd/(RESULT+'_'+str(time.time_ns())+'.zip');dt=dst.with_suffix('.zip.tmp')
  shutil.copyfile(path,dt);require(file_sha(dt)==file_sha(path),'Drive transfer digest mismatch');os.replace(dt,dst)
  valid=[]
  for old in sorted(dd.glob(RESULT+'_*.zip'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
   try:
    with zipfile.ZipFile(old) as z:
     verify_archive(z,heartbeat=lambda *a,**k:None);require(read_json(z,'identity.json')['protocol']==PROTOCOL,'Wrong result protocol')
    valid.append(old)
   except Exception as e:print('保留未校验C27备份',old.name,str(e),flush=True)
  for old in valid[3:]:old.unlink()
  print('DRIVE_BACKUP_OK',dst,flush=True)
 print('RESULT_ARCHIVE='+str(path),flush=True);return path


def resume_score(path,root):
 with zipfile.ZipFile(path) as z:
  verify_archive(z,heartbeat=lambda *a,**k:None);i=read_json(z,'identity.json')
  require(i['protocol']==PROTOCOL and i['settings']==SETTINGS and i['source_hashes']==source_hashes(root),'C27 resume source/settings differ')
  require(i['parent_C25_sha256']==PARENT_SHA,'C27 parent differs')
  names=z.namelist()
  return (int('SUMMARY.json' in names),int('WRITERS_FROZEN.json' in names),sum(n.startswith('evaluation/') and n.endswith('/COMPLETE.json') for n in names),
          sum(n.startswith('arms/') and n.endswith('/FROZEN.json') for n in names),sum('/candidates/' in n and n.endswith('/COMPLETE.json') for n in names))
