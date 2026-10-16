"""Actual all-grid physical search, restoring the round origin after every trial."""
import json
from pathlib import Path
import numpy as np
import torch
from c24_protocol import candidate_specs,SETTINGS,quality,guard,merge_route,check_replay,candidate_key
from c24_identity import head_hash,tensor_hash
from output_layer import layer_matrix,set_layer_matrix
from raw_writer import one_raw_proposal
from radius_calibration import scale_direction
from joint_calibration import response_prediction

def scan_round(head,route,group,states,system,bases,current,round_dir,output_scale,evaluate,measure_training,invariant_check,dump,snapshot):
 round_dir=Path(round_dir);old_layer=layer_matrix(head,route);before_hash=head_hash(head);group_ids=[t['id'] for t in group]
 if len(group_ids)!=15 or len(set(group_ids))!=15 or len(states)!=15 or any(t['case']!=SETTINGS['scan_case'] for t in group):raise ValueError('All15 selected-route states required')
 def restore():
  set_layer_matrix(head,route,old_layer);invariant_check()
  if head_hash(head)!=before_hash:raise RuntimeError('Round origin not restored')
 records=[];eligible=[];seen={};unique_audits=set();audits=[]
 for index,spec in enumerate(candidate_specs()):
  folder=round_dir/'candidates'/spec['id'];folder.mkdir(parents=True,exist_ok=True);complete=folder/'COMPLETE.json'
  D,info=scale_direction(system,bases[spec['ridge_index']],spec['ridge'],SETTINGS['layer_frobenius_radius'],spec['increment_radius'])
  trial=(old_layer.double()+spec['alpha']*torch.from_numpy(D)).float()
  try:
   set_layer_matrix(head,route,trial);invariant_check();trial_hash=head_hash(head)
   if complete.exists():
    record=json.loads(complete.read_text())
    if record['spec']!=spec or record['start_head_sha256']!=before_hash or record['head_sha256']!=trial_hash:raise RuntimeError('Cached physical candidate identity changed')
   else:
    displacement=[]
    for t,r in zip(group,states):
     delta=one_raw_proposal(head,t['input'],t['route'],output_scale).cpu();displacement.append(((t['prep']+delta).double()-r['theta'].double()).numpy())
    displacement=np.stack(displacement);pred=response_prediction(system['J'],system['E'],displacement);alias=seen.get(trial_hash)
    if alias is not None:rows=alias['route_rows'];alias_of=alias['spec']['id']
    else:
     rows={};alias_of=None
     for t in group:
      f=folder/'states'/(t['id']+'.json');delta=one_raw_proposal(head,t['input'],t['route'],output_scale).cpu();theta=t['prep']+delta
      if f.exists():row=json.loads(f.read_text())
      else:row={'reference_valid':t['reference_valid'],'metrics':evaluate(theta,t)};dump(f,row)
      if row['metrics']['theta_sha256']!=tensor_hash(theta):raise RuntimeError('Cached trial parameter state changed')
      rows[t['id']]=row
    g=guard(current,merge_route(current,rows,group_ids));actual=np.array([rows[t['id']]['metrics']['response_residual'] for t in group])
    record={'spec':spec,'head_sha256':trial_hash,'start_head_sha256':before_hash,'route_rows':rows,'route_guard':g,'route_guard_eligible':g['accepted'],
     'full60_verified':False,'byte_identical_alias_of':alias_of,'direction_info':info,'prediction':{'maximum_drift':float(np.max(np.abs(pred))),
      'maximum_prediction_error':float(np.max(np.abs(actual-pred))),'actual_layer_step_norm':float((trial.double()-old_layer.double()).norm()),
      'maximum_actual_memory_step_norm':float(np.max(np.linalg.norm(displacement,axis=1)))}}
    dump(complete,record)
   if set(record['route_rows'])!=set(group_ids):raise RuntimeError('Cached route denominator changed')
   for t in group:
    delta=one_raw_proposal(head,t['input'],t['route'],output_scale).cpu()
    if record['route_rows'][t['id']]['metrics']['theta_sha256']!=tensor_hash(t['prep']+delta):raise RuntimeError('Cached complete candidate state changed')
   g=guard(current,merge_route(current,record['route_rows'],group_ids))
   if g!=record['route_guard'] or g['accepted']!=record['route_guard_eligible']:raise RuntimeError('Cached candidate guard mismatch')
   records.append(record);seen.setdefault(trial_hash,record)
   if g['accepted']:eligible.append({**spec,'rows':merge_route(current,record['route_rows'],group_ids),'record':record})
   print('C24 PHYSICAL SCAN',index+1,'/90',spec['id'],'alpha',spec['alpha'],'eligible',g['accepted'],'R',g['after']['maximum_drift'],flush=True)
  finally:restore()
  if (index+1)%SETTINGS['archive_every_candidates']==0:snapshot()
 winner=None
 for candidate in sorted(eligible,key=candidate_key):
  record=candidate['record'];spec=record['spec']
  if record['head_sha256'] in unique_audits:continue
  unique_audits.add(record['head_sha256']);folder=round_dir/'candidate_audits'/spec['id'];folder.mkdir(parents=True,exist_ok=True);f=folder/'AUDIT.json'
  D,_=scale_direction(system,bases[spec['ridge_index']],spec['ridge'],SETTINGS['layer_frobenius_radius'],spec['increment_radius']);trial=(old_layer.double()+spec['alpha']*torch.from_numpy(D)).float()
  try:
   set_layer_matrix(head,route,trial);invariant_check()
   if head_hash(head)!=record['head_sha256']:raise RuntimeError('Ranked candidate changed')
   if f.exists():a=json.loads(f.read_text())
   else:
    measured=measure_training(head);error=None
    try:check_replay(candidate['rows'],measured)
    except ValueError as e:error=str(e)
    g=guard(current,measured);a={'spec':spec,'start_head_sha256':before_hash,'head_sha256':head_hash(head),'rows':measured,'guard':g,'replay_error':error,'full60_verified':bool(error is None and g['accepted'])};dump(f,a)
   if a['spec']!=spec or a['start_head_sha256']!=before_hash or a['head_sha256']!=record['head_sha256'] or a['guard']!=guard(current,a['rows']):raise RuntimeError('Cached full60 audit mismatch')
   error=None
   try:check_replay(candidate['rows'],a['rows'])
   except ValueError as e:error=str(e)
   if a['replay_error']!=error or a['full60_verified']!=bool(error is None and a['guard']['accepted']):raise RuntimeError('Cached full60 acceptance mismatch')
   audits.append(a)
   if a['full60_verified']:
    winner={'spec':spec,'head_sha256':a['head_sha256'],'layer':trial.clone(),'rows':a['rows'],'guard':a['guard'],'full60_verified':True};break
  finally:restore()
 restore()
 return winner,{'n_requested_candidate_specs':90,'n_completed_candidate_specs':len(records),'n_unique_candidate_heads':len(seen),'n_route_guard_eligible_specs':len(eligible),'n_full60_audits':len(audits),'start_head_sha256':before_hash,'origin_restored':True,'selected_spec':winner['spec'] if winner else None}
