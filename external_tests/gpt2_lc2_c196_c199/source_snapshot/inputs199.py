"""Bind all44 originals and independently reconstruct original C198 arithmetic."""
from core199 import *
from inputs198 import load as load198
from worker198 import frozen_digest
from audit147 import source_check
def audit198(path,ctx,out,c,budget=lambda:None):
 out=Path(out);out.mkdir(parents=True,exist_ok=True);d=ctx['data']['d'];p=ctx['C196_final_head'];groups={};preps={};features={};points={};ends={};replays=[];repeats=[]
 with contextlib.closing(Archive(path)) as z:
  members=z.verify_manifest();source_check(z,c['source198'],'C198');s=z.json('SUMMARY.json');compare(s['protocol'],parent.PROTOCOL);compare(s['complete'],True);compare(s['synthetic_fixture'],not c['production']);compare(s['real_model_execution_claim'],c['production']);compare(s['all43_originals_independently_bound'],c['production']);compare(s['errors'],[]);compare(s['stage'],'complete');compare(s['fixed_final_head_digest'],parameter_hash(p));compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),parent.design());compare(z.json('ALL672_REQUESTS_FROZEN.json'),parent.requests());compare(z.json('FROZEN_PIPELINE_DIGESTS.json'),frozen_digest(ctx));compare(s['original_inputs_unchanged'],{k:True for k in parent.INPUT_KEYS});links=z.json('ORIGINAL_INPUTS.json');compare(set(links),set(parent.INPUT_KEYS))
  for k in parent.INPUT_KEYS:compare(links[k]['sha256'],c['pins'][k]['sha256'],'C198 ancestry/'+k)
  for k,v in parent.no_training().items():compare(s[k],v,'C198 no training/'+k)
  check_head(z.arrays('FIXED_C196_FINAL_HEAD.npz')['P'],c['production']);exact(z.arrays('FIXED_C196_FINAL_HEAD.npz')['P'],p)
  refs=z.arrays('FIXED_ARCHIVED_REFERENCES.npz');compare(set(refs),{'response','logits'});exact(refs['response'],np.asarray(ctx['baseline']['FP32_cached_response_reference'],np.float32));exact(refs['logits'],ctx['baseline_logits']['FP32_logits_cast64'])
  prepared=z.arrays('ALL84_PREPARED_STATES.npz');compare(set(prepared),set(IDS))
  for sid in IDS:exact(prepared[sid],d['points'][sid]['prepared'])
  for name in ('SEALED_BEFORE_MODEL.json','FRESH_PROPOSALS_SEALED_BEFORE_ENDPOINTS.json'):
   seal=z.json(name)
   for n,h in seal['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'C198 saved seal/'+n)
  seal=z.json('FRESH_PROPOSALS_SEALED_BEFORE_ENDPOINTS.json');compare(seal['requests_consumed'],336);compare(seal['records_completed'],336);compare(seal['postwrite_endpoint_measurements'],0)
  rawrows=z.raw('ALL672_RECORDS.jsonl').splitlines();rows=[clean_json(line) for line in rawrows];compare([dict(stage=r['stage'],state_id=r['state_id']) for r in rows],parent.requests(),'Original C198672 row order')
  for r in rows:groups.setdefault(r['stage'],[]).append(r)
  saved=z.arrays('ALL168_FRESH_PROPOSALS_BEFORE_ENDPOINTS.npz');compare(set(saved),{'round%d__%s'%(j,sid) for j in (0,1) for sid in IDS});checked_features=[]
  for rnd in (0,1):
   key=str(rnd);preps[key]=groups['prepare'+key];features[key]=groups['features'+key];points[key]={}
   for i,sid in enumerate(IDS):
    budget();q=d['points'][sid];case=sid.split('__')[0];prep=preps[key][i];compare(prep,dict(stage='prepare'+key,state_id=sid,round=rnd,**metric_record(prep['metrics'],q['prepared'],q['prepared'],case,True,False)),'C198 preparation arithmetic')
    n='prewrite/round%d/%s.npz'%(rnd,sid);a=z.arrays(n);compare(set(a),{'prepared','capture3147','full3211','prewrite_observed','raw','weights','effective','point'});exact(a['prepared'],q['prepared']);capture=farray(a['capture3147'],(3147,));o=farray(a['prewrite_observed'],(22,));compare(float(capture[3076]),float(int(case[-2])^int(case[-1])));want=fresh_proposal(d,p,q['prepared'],capture)
    for name in ('raw','full','weights','effective','point'):exact(a['full3211' if name=='full' else name],want[name],'Recomputed C198 fresh '+name)
    point=a['point'];need(point.dtype==np.float32,'Actual C198 FP32 point');exact(point,saved['round%d__%s'%(rnd,sid)]);points[key][sid]=point.copy()
    f=dict(stage='features'+key,state_id=sid,round=rnd,prepared_sha256=safe.tensor_hash(q['prepared']),fresh_point_sha256=safe.tensor_hash(point),capture_Linf_vs_archive=float(np.max(abs(capture-q['features'][:3147]))),full3211_Linf_vs_archive=float(np.max(abs(want['full']-q['features']))),raw_Linf_vs_archive=float(np.max(abs(want['raw'].astype(float)-q['raw'].astype(float)))),routing_Linf_vs_archive=float(np.max(abs(want['weights']-d['weights'][sid]))),point_Linf_vs_C196=float(np.max(abs(point.astype(float)-ctx['C196_final_points'][sid].astype(float)))),point_coordinates_changed_vs_C196=int(np.count_nonzero(point!=ctx['C196_final_points'][sid])),feature_vs_preparation_observed_Linf=float(np.max(abs(o-np.asarray(prep['metrics']['observed'])))),endpoint_used_for_proposal=False);compare(features[key][i],f,'Full C198 fresh input/route/point diagnostic');checked_features.append(f)
  valid={sid:all(preps[str(j)][i]['qualification']['complete_pass'] for j in (0,1)) for i,sid in enumerate(IDS)}
  for arm in ('archived','fresh'):
   for rnd in (0,1):
    key=arm+str(rnd);ends[key]=groups[key]
    for i,sid in enumerate(IDS):
     budget();r=ends[key][i];pt=ctx['C196_final_points'][sid] if arm=='archived' else points[str(rnd)][sid];want=dict(stage=key,state_id=sid,round=rnd,arm=arm,**metric_record(r['metrics'],pt,d['points'][sid]['prepared'],sid.split('__')[0],valid[sid]));delta=replay_pair(ctx['C196_final_native'][str(rnd)][i]['metrics'],r['metrics']);want['comparison_with_original_C196']=delta;compare(r,want,'C198 full native endpoint')
     if arm=='archived':replays.append(dict(state_id=sid,round=rnd,**delta))
     if rnd==1:repeats.append(dict(arm=arm,state_id=sid,**replay_pair(ends[arm+'0'][i]['metrics'],r['metrics'])))
    compare(z.json(key+'_ALL84.json'),ends[key],'C198 endpoint duplicate')
  for i,sid in enumerate(IDS):repeats.append(dict(arm='preparation',state_id=sid,**replay_pair(preps['0'][i]['metrics'],preps['1'][i]['metrics'])))
  result=parent.outcome(preps,ends,replays,repeats);parity=all(f['feature_vs_preparation_observed_Linf']<=.0002 for f in checked_features);result['feature_preparation_parity_consistent']=parity
  if not parity:result.update(frozen_training_replay_passed=False,decision='C198_COMPLETE_REPLAY_OR_TRAINING_CONSTRAINT_FAILURE_NO_GENERALISATION')
  for k,v in result.items():compare(s[k],v,'C198 reconstructed outcome/'+k)
  need(result['frozen_training_replay_passed'],'Original C198 full training replay prerequisite')
  expected=dict(archived_endpoint_summaries={str(j):parent.summaries(ends['archived'+str(j)]) for j in (0,1)},fresh_endpoint_summaries={str(j):parent.summaries(ends['fresh'+str(j)]) for j in (0,1)},preparation_summaries={str(j):parent.summaries(preps[str(j)]) for j in (0,1)},archived_replay_mismatch_records=[r for r in replays if not pair_pass(r)],repeat_mismatch_records=[r for r in repeats if not pair_pass(r)],fresh_feature_diagnostics={name:max(r[name] for r in checked_features) for name in ('capture_Linf_vs_archive','full3211_Linf_vs_archive','raw_Linf_vs_archive','routing_Linf_vs_archive','point_Linf_vs_C196','feature_vs_preparation_observed_Linf')})
  for k,v in expected.items():compare(s[k],v,'C198 reconstructed summary/'+k)
  for k,v in dict(records_completed=672,requests_consumed=672,new_model_forwards=1513,new_feature_records=168,archived_endpoint_records=168,fresh_endpoint_records=168,preparation_records=168,preparation_replay_performed=True,all_fresh_proposals_sealed_before_endpoints=True,FP64_full_acceptance_evaluated=False,independent_laboratory_replication=False).items():compare(s[k],v,'C198 full coverage/'+k)
  compare(s['runtime_calls'],dict(model_forwards=1513,initialization_forwards=1,feature_attempts=168,feature_calls=168,metric_attempts=504,metric_calls=504))
  for n,obj in [('ALL168_ORIGINAL_ENDPOINT_REPLAY_CHECKS.json',replays),('ALL252_REPEAT_CHECKS.json',repeats),('ALL168_FRESH_FEATURE_DIAGNOSTICS.json',features),('ALL168_PREPARATION_RECORDS.json',preps)]:compare(z.json(n),obj,'C198 auxiliary reconstruction/'+n)
  if c['production']:
   a=z.arrays('model_runtime/INITIALIZATION_BASELINE_DIAGNOSTIC.npz');exact(a['fixed_logits'],ctx['baseline_logits']['FP32_logits_cast64'].astype(np.float32));exact(a['fixed_response'],np.asarray(ctx['baseline']['FP32_cached_response_reference'],np.float32));r=z.json('model_runtime/REFERENCE_POLICY.json');compare(r,dict(protocol=parent.PROTOCOL,fresh_zero_logits_gap_Linf=float(np.max(abs(a['fresh_logits']-a['fixed_logits']))),fresh_zero_response_gap_Linf=float(np.max(abs(a['fresh_response']-a['fixed_response']))),fresh_baseline_used_for_scoring=False,reference_recentered=False,archived_logits_sha256=array_digest(a['fixed_logits']),archived_response_sha256=array_digest(a['fixed_response'])));pins=z.json('model_runtime/MODEL_PINS.json')
   for k in SEMANTIC_PINS:compare(pins[k],ctx['modelpins'][k],'C198 semantic model pins/'+k)
 receipt=dict(protocol=parent.PROTOCOL,decision=s['decision'],whole_zip_sha256=sha(path),members_hashed=members,feature_records_reconstructed=168,native_records_reconstructed=504,ordered_records_reconstructed=672,model_outputs_independently_recomputed=False,new_model_forwards=0,source_executed_from_archive=False)
 dump(out/'C198_ORIGINAL_REBOUND.json',receipt);dump(out/'C198_RECOMPUTED_SUMMARIES.json',expected);return dict(receipt=receipt,points=points,preps=preps,ends=ends,features=features)
def load(paths,out,c=None,budget=lambda:None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);compare(set(paths),set(INPUT_KEYS))
 for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed original before payload/'+k)
 for n,h in c['source198'].items():compare(sha(ROOT/'vendor198'/n),h,'Immutable delivered C198 source/'+n)
 with archive155.prior.no_models(),safe.no_solvers():
  ctx=load198({k:paths[k] for k in parent.INPUT_KEYS},out/'forty_three_originals',c['parent'],budget);r=audit198(paths['C198'],ctx,out/'C198_reconstruction',c,budget)
 ctx.update(C198_receipt=r['receipt'],C198_fresh_points=r['points'],C198_preps=r['preps'],C198_fresh_native={str(j):r['ends']['fresh'+str(j)] for j in (0,1)})
 dump(out/'ALL44_ORIGINALS_BOUND.json',dict(all_originals_bound=c['production'],head_digest=parameter_hash(ctx['C196_final_head']),C198_decision=r['receipt']['decision'],new_model_forwards=0));return ctx
