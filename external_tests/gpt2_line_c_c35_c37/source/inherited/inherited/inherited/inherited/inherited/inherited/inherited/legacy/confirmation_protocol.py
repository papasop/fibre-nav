"""Prospective local-state validation of the exact frozen C9 head; no fitting."""
import hashlib, json, os, shutil, time, zipfile
from pathlib import Path
from c9_protocol import sha, verified_zip, verify_open_zip, CASES, STATES, REVISION, C6_SHA, C8_SHA, endpoint_gate, episode_gate, target_from_reads

PROTOCOL='GPT2_C9_CONFIRMATION_A100_V1'
RESULT='gpt2_c9_confirmation_results'
PARENT_PROTOCOL='GPT2_C9_STATE_GENERALISATION_A100_V1'
C9_SHA='f0f5abd2197311438bf23588dbcb1bedef1649380873b04f86b5e7b1340fce3b'
HEAD_SHA='835793a66ec435a8cae6c879b65351e808152919af1a000607426404f6c955ab'
CHECKPOINT='heads/functional/checkpoint_000240.pt'
CHECKPOINT_SHA='c0ded5051fe40b7be05b4570dd3b57bf393fc619362e8ae24d0c827ada29b4a7'
ARMS=('learned_raw','learned_corrected','correction_only','random_corrected','nearest_teacher_corrected')
SETTINGS={'response_budget':.002,'kl_budget':.005,'margin_gate':.05,
 'correction_ridge':1e-6,'correction_max_steps':8,'correction_backtracking':6,
 'new_state_seed_base':215031,'random_control_seed_base':315031,'norms':[.375,.75],
 'replicates_per_norm':2,'n_requested':16,'archive_every_states':2,
 'feature_identity_tolerance':1e-4,'proposal_replay_tolerance':2e-6,
 'requested_output_bit_supplied':True,'new_training_steps':0,
 'fresh_state_seeds':True,'fresh_evaluation':False,'independent_model_seed':False,
 'same_access_prompts':True,'same_four_trained_input_classes':True,
 'native_parameter_update':False,'controller_dispatch_tested':False,
 'state_replacement_allowed':False,'test_based_checkpoint_selection':False,
 'correction_only_expected_noop_on_admissible_preparation':True,
 'nearest_baseline':'minimum standardised feature distance over all twelve C9 training references; tie breaks by fixed training order; transported original teacher increment',
 'correction':'C9 response-only SVD ridge step and fixed six backtracking values; content is checked afterwards, never supplied to repair',
 'random_baseline':'one fixed isotropic direction per state, norm matched to learned proposal; no best-of selection',
 'primary_scope':'new local states of known input classes in the same model/chart, not unseen tasks or independent-model replication',
 'timing':'CUDA synchronised observed seconds; saved completed operations only; interrupted partial work and archive IO excluded'}

def state_specs():
 return [{'id':case+'__confirm%d'%i,'case':case,'seed':SETTINGS['new_state_seed_base']+100*c+i,
          'random_seed':SETTINGS['random_control_seed_base']+100*c+i,'norm':SETTINGS['norms'][i//2],
          'cohort':'prospective_local_state_validation'}
         for c,case in enumerate(CASES) for i in range(4)]

def arm_order(index):
 k=index%len(ARMS);return ARMS[k:]+ARMS[:k]

def correction_accept(old,new,alpha):
 import math
 return math.isfinite(new) and new<old*(1-1e-4*alpha)

def validate_parent(path):
 if sha(path)!=C9_SHA:raise ValueError('Requires the exact complete C9 result archive, not C10/C12 or source ZIP')
 with zipfile.ZipFile(path) as z:
  ident=verify_open_zip(z,PARENT_PROTOCOL);s=json.loads(z.read('SUMMARY.json'));f=json.loads(z.read('WRITER_FROZEN.json'))
  if not s.get('complete') or s.get('decision')!='C9_LOCAL_STATE_GENERALISATION_PASS':raise ValueError('Complete positive C9 required')
  if s.get('writer_head_sha256')!=HEAD_SHA or f.get('head_sha256')!=HEAD_SHA or f.get('selected_checkpoint')!=CHECKPOINT:raise ValueError('Frozen C9 writer changed')
  if hashlib.sha256(z.read(CHECKPOINT)).hexdigest()!=CHECKPOINT_SHA:raise ValueError('Actual C9 checkpoint changed')
  if hashlib.sha256(z.read('parent_c8.zip')).hexdigest()!=C8_SHA:raise ValueError('Nested C8 changed')
  return ident

def source_hashes(root):
 return {p.name:sha(p) for p in Path(root).iterdir() if p.is_file() and not p.name.startswith('build_')}

def inherited_check(root):
 for name,digest in json.loads((Path(root)/'INHERITED_SOURCE_HASHES.json').read_text()).items():
  if sha(Path(root)/name)!=digest:raise ValueError('Inherited C9 source changed: '+name)

def aggregate(rows):
 if len(rows)!=16 or [r['id'] for r in rows]!=[s['id'] for s in state_specs()]:raise ValueError('All sixteen fixed requested states must be reported in order')
 ans={'n_requested':16,'valid_state_references':sum(bool(r['reference_valid']) for r in rows),'arms':{}}
 for arm in ARMS:
  r=[x['arms'][arm] for x in rows]
  ans['arms'][arm]={'n_requested':16,'passed':sum(bool(x['protocol_pass']) for x in r),
   'endpoint_passed_including_invalid_references':sum(bool(x['metrics'].get('episode_pass')) for x in r),
   'gate_counts':{g:sum(bool(x['metrics'].get('gates',{}).get(g)) for x in r) for g in ('bits_match','output_flip','unselected_preserved','exact_read_margin','development_read_margin','response','KL')},
   'online_elapsed_seconds_total':sum(x['online_seconds'] for x in r),
   'accepted_correction_steps':sum(x['correction_steps'] for x in r),
   'correction_Jacobian_builds':sum(x.get('correction_Jacobian_builds',0) for x in r),
   'correction_SVD_calls':sum(x.get('correction_SVD_calls',0) for x in r),
   'correction_backtracking_trials':sum(x.get('correction_backtracking_trials',0) for x in r)}
 # Paired advantage is descriptive: these states share a model/chart and known classes.
 ans['paired_comparisons']={}
 for arm in ARMS[2:]:
  pairs=[(bool(x['arms']['learned_corrected']['protocol_pass']),bool(x['arms'][arm]['protocol_pass'])) for x in rows]
  ans['paired_comparisons'][arm]={'learned_only':sum(a and not b for a,b in pairs),'control_only':sum(b and not a for a,b in pairs),
   'both_pass':sum(a and b for a,b in pairs),'neither_pass':sum(not a and not b for a,b in pairs)}
 return ans

def decide(counts,replay):
 if len(replay)!=8 or not all(x.get('replay_pass') for x in replay):return 'C9_CONFIRMATION_PARENT_REPLAY_FAILURE'
 if counts['valid_state_references']!=16:return 'C9_CONFIRMATION_STATE_REFERENCES_INCONCLUSIVE'
 return 'C9_LOCAL_STATE_CONFIRMATION_'+('PASS' if counts['arms']['learned_corrected']['passed']==16 else 'FAIL')

def resume_score(path):
 ident=verified_zip(path,PROTOCOL)
 if ident.get('settings')!=SETTINGS or ident.get('parent_c9_sha256')!=C9_SHA or ident.get('source_hashes')!=source_hashes(Path(__file__).parent):raise ValueError('Incompatible confirmation source/settings')
 with zipfile.ZipFile(path) as z:
  if hashlib.sha256(z.read('parent_c9.zip')).hexdigest()!=C9_SHA:raise ValueError('Parent C9 digest mismatch')
  names=z.namelist();return (int('SUMMARY.json' in names),sum(n.startswith('evaluation/') and n.endswith('/COMPLETE.json') for n in names),sum(n.startswith('evaluation/') and n.endswith('/RESULT.json') for n in names),sum(n.startswith('replay/') and n.endswith('.json') for n in names))

def snapshot(out,drive=''):
 out=Path(out)
 if not (out/'identity.json').exists():return None
 m={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and not p.name.endswith('.tmp')}
 tmp=out/'MANIFEST.json.tmp';tmp.write_text(json.dumps(m,indent=2));os.replace(tmp,out/'MANIFEST.json')
 dest=out.parent/(RESULT+'.zip');tmp=dest.with_suffix('.partial')
 with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
  for n in list(m)+['MANIFEST.json']:z.write(out/n,n,compress_type=zipfile.ZIP_STORED if n.endswith('.zip') else zipfile.ZIP_DEFLATED)
 verified_zip(tmp,PROTOCOL);os.replace(tmp,dest)
 if drive:
  d=Path(drive);d.mkdir(parents=True,exist_ok=True);q=d/(RESULT+'_'+str(time.time_ns())+'.zip');temp=q.with_suffix('.partial')
  shutil.copyfile(dest,temp)
  if sha(temp)!=sha(dest):raise RuntimeError('Drive copy digest mismatch')
  verified_zip(temp,PROTOCOL);os.replace(temp,q);print('DRIVE_BACKUP_OK',q,flush=True)
  valid=[]
  for old in sorted(d.glob(RESULT+'_*.zip')):
   try:verified_zip(old,PROTOCOL);valid.append(old)
   except Exception:continue
  for old in valid[:-3]:old.unlink()
 print('RESULT_ARCHIVE='+str(dest),flush=True);return dest
