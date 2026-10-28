"""Two distinct frozen paths, no replacement of freshly acquired writer inputs."""
import argparse,resource,traceback
from core198 import *
from inputs198 import load
def run(paths,out,c=None,input_loader=None,runtime_factory=None,proposal_function=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();stage='registered';errors=[];ctx=None;rt=None;records=[];seals={};at=0;preps={};features={};fresh={};ends={};replays=[];repeats=[]
 calls=dict(model_forwards=0,initialization_forwards=0,feature_attempts=0,feature_calls=0,metric_attempts=0,metric_calls=0)
 s=dict(protocol=PROTOCOL,complete=False,decision='C198_INCOMPLETE_NO_REPLAY_ACCEPTANCE',synthetic_fixture=not c['production'],real_model_execution_claim=False,all43_originals_independently_bound=False,scope=SCOPE,training_states_requested=84,preparation_replay_performed=False,all_fresh_proposals_sealed_before_endpoints=False,**no_training())
 def checkpoint():
  s.update(stage=stage,records_completed=len(records),requests_consumed=at,new_model_forwards=calls['model_forwards'],runtime_calls=dict(calls),elapsed_worker_seconds=time.monotonic()-started,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 def budget():
  need(time.monotonic()-started<TIMEOUT,'C198 fixed240-minute worker budget')
  for n,h in seals.items():compare(sha(out/n),h,'Frozen C198 artifact/'+n)
  if ctx is not None:compare(frozen_digest(ctx),frozen,'Full frozen proposal pipeline and archived data')
 def save(r):
  records.append(r)
  with (out/'ALL672_RECORDS.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
  if len(records)%12==0:checkpoint()
 def authorize(st,sid):
  nonlocal at
  budget();compare(requests()[at],dict(stage=st,state_id=sid),'Exact672 request registry');at+=1
 try:
  need(not(out/'SUMMARY.json').exists(),'Fresh output only');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());seals['DESIGN_FROZEN_BEFORE_MODEL.json']=sha(out/'DESIGN_FROZEN_BEFORE_MODEL.json');checkpoint();compare(set(paths),set(INPUT_KEYS),'Exactly43 fixed originals')
  for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Whole original before payload/'+k)
  for n,h in c['source197'].items():compare(sha(ROOT/'vendor197'/n),h,'Trusted source/'+n)
  stage='original_archive_binding';ctx=(load if input_loader is None else input_loader)(paths,out/'binding',c,lambda:need(time.monotonic()-started<TIMEOUT,'C198 binding budget'));frozen=frozen_digest(ctx);check_head(ctx['C196_final_head'],c['production']);d=ctx['data']['d'];p=ctx['C196_final_head'];compare(set(d['points']),set(IDS));compare(set(ctx['C196_final_points']),set(IDS));s['all43_originals_independently_bound']=input_loader is None
  dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});dump(out/'ALL672_REQUESTS_FROZEN.json',requests());dump(out/'FROZEN_PIPELINE_DIGESTS.json',frozen);np.savez_compressed(out/'FIXED_C196_FINAL_HEAD.npz',P=p);np.savez_compressed(out/'FIXED_ARCHIVED_REFERENCES.npz',response=np.asarray(ctx['baseline']['FP32_cached_response_reference'],np.float32),logits=ctx['baseline_logits']['FP32_logits_cast64']);np.savez_compressed(out/'ALL84_PREPARED_STATES.npz',**{sid:d['points'][sid]['prepared'] for sid in IDS})
  for n in ('ORIGINAL_INPUTS.json','ALL672_REQUESTS_FROZEN.json','FROZEN_PIPELINE_DIGESTS.json','FIXED_C196_FINAL_HEAD.npz','FIXED_ARCHIVED_REFERENCES.npz','ALL84_PREPARED_STATES.npz'):seals[n]=sha(out/n)
  dump(out/'SEALED_BEFORE_MODEL.json',dict(artifact_seals=dict(seals),new_model_forwards=0,new_training_steps=0));seals['SEALED_BEFORE_MODEL.json']=sha(out/'SEALED_BEFORE_MODEL.json');budget();stage='pinned_CPU_model_initialization';checkpoint()
  if runtime_factory is None:
   from runtime198 import factory
   runtime_factory=factory
  rt=runtime_factory(out/'model_runtime',ctx,calls);s['real_model_execution_claim']=bool(c['production'] and input_loader is None);s['fixed_final_head_digest']=parameter_hash(p);budget()
  with safe.no_solvers():
   for rnd in (0,1):
    stage='prepare'+str(rnd);rr=[]
    for sid in IDS:
     authorize(stage,sid);point=d['points'][sid]['prepared'];farray(point,(DIM,));need(point.dtype==np.float32,'Original physical preparation');r=dict(stage=stage,state_id=sid,round=rnd,**metric_record(rt.metrics(point,sid.split('__')[0],False),point,point,sid.split('__')[0],True,False));rr.append(r);save(r)
    preps[str(rnd)]=rr;stage='features'+str(rnd);ff=[]
    for i,sid in enumerate(IDS):
     authorize(stage,sid);q=d['points'][sid];case=sid.split('__')[0];desired=int(case[-2])^int(case[-1]);capture,observed=rt.features(q['prepared'],desired);farray(capture,(3147,));farray(observed,(22,));compare(float(capture[3076]),float(desired),'Requested bit feature');proposal=(fresh_proposal if proposal_function is None else proposal_function)(d,p,q['prepared'],capture);pt=proposal['point'];need(pt.dtype==np.float32,'Actual fresh physical proposal');farray(pt,(DIM,));fresh[(rnd,sid)]=pt.copy()
     filename='prewrite/round%d/%s.npz'%(rnd,sid);path=out/filename;path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,prepared=q['prepared'],capture3147=capture,full3211=proposal['full'],prewrite_observed=observed,raw=proposal['raw'],weights=proposal['weights'],effective=proposal['effective'],point=pt);seals[filename]=sha(path)
     r=dict(stage=stage,state_id=sid,round=rnd,prepared_sha256=safe.tensor_hash(q['prepared']),fresh_point_sha256=safe.tensor_hash(pt),capture_Linf_vs_archive=float(np.max(abs(capture-q['features'][:3147]))),full3211_Linf_vs_archive=float(np.max(abs(proposal['full']-q['features']))),raw_Linf_vs_archive=float(np.max(abs(proposal['raw'].astype(float)-q['raw'].astype(float)))),routing_Linf_vs_archive=float(np.max(abs(proposal['weights']-d['weights'][sid]))),point_Linf_vs_C196=float(np.max(abs(pt.astype(float)-ctx['C196_final_points'][sid].astype(float)))),point_coordinates_changed_vs_C196=int(np.count_nonzero(pt!=ctx['C196_final_points'][sid])),feature_vs_preparation_observed_Linf=float(np.max(abs(observed-np.asarray(preps[str(rnd)][i]['metrics']['observed'])))),endpoint_used_for_proposal=False);ff.append(r);save(r)
    features[str(rnd)]=ff;checkpoint()
   np.savez_compressed(out/'ALL168_FRESH_PROPOSALS_BEFORE_ENDPOINTS.npz',**{'round%d__%s'%(j,sid):fresh[(j,sid)] for j in (0,1) for sid in IDS});seals['ALL168_FRESH_PROPOSALS_BEFORE_ENDPOINTS.npz']=sha(out/'ALL168_FRESH_PROPOSALS_BEFORE_ENDPOINTS.npz');dump(out/'FRESH_PROPOSALS_SEALED_BEFORE_ENDPOINTS.json',dict(artifact_seals=dict(seals),requests_consumed=at,records_completed=len(records),postwrite_endpoint_measurements=0));seals['FRESH_PROPOSALS_SEALED_BEFORE_ENDPOINTS.json']=sha(out/'FRESH_PROPOSALS_SEALED_BEFORE_ENDPOINTS.json');s.update(all_fresh_proposals_sealed_before_endpoints=True,preparation_replay_performed=True);budget()
   valid={sid:all(preps[str(j)][i]['qualification']['complete_pass'] for j in (0,1)) for i,sid in enumerate(IDS)}
   point_ids={(j,sid):safe.tensor_hash(pt) for (j,sid),pt in fresh.items()}
   for arm in ('archived','fresh'):
    for rnd in (0,1):
     stage=arm+str(rnd);rr=[]
     for i,sid in enumerate(IDS):
      authorize(stage,sid);point=ctx['C196_final_points'][sid] if arm=='archived' else fresh[(rnd,sid)];expected=ctx['C196_final_native'][str(rnd)][i]['metrics']['theta_sha256'] if arm=='archived' else point_ids[(rnd,sid)];compare(safe.tensor_hash(point),expected,'Sealed point consumed');met=rt.metrics(point,sid.split('__')[0],True);r=dict(stage=stage,state_id=sid,round=rnd,arm=arm,**metric_record(met,point,d['points'][sid]['prepared'],sid.split('__')[0],valid[sid]))
      reference=ctx['C196_final_native'][str(rnd)][i]['metrics'];delta=replay_pair(reference,met);r['comparison_with_original_C196']=delta
      if arm=='archived':replays.append(dict(state_id=sid,round=rnd,**delta))
      if rnd==1:repeats.append(dict(arm=arm,state_id=sid,**replay_pair(ends[arm+'0'][i]['metrics'],met)))
      rr.append(r);save(r)
     ends[stage]=rr;dump(out/(stage+'_ALL84.json'),rr);checkpoint();print('C198_ENDPOINT',stage,summaries(rr),flush=True)
   for i,sid in enumerate(IDS):repeats.append(dict(arm='preparation',state_id=sid,**replay_pair(preps['0'][i]['metrics'],preps['1'][i]['metrics'])))
  compare(at,672);compare(len(records),672);compare(calls['feature_calls'],168);compare(calls['metric_calls'],504);compare(calls['model_forwards'],FORWARDS);rt.verify(True);budget();result=outcome(preps,ends,replays,repeats);parity=all(r['feature_vs_preparation_observed_Linf']<=.0002 for ff in features.values() for r in ff);result['feature_preparation_parity_consistent']=parity
  if not parity:result.update(frozen_training_replay_passed=False,decision='C198_COMPLETE_REPLAY_OR_TRAINING_CONSTRAINT_FAILURE_NO_GENERALISATION')
  s.update(result,complete=True,archived_endpoint_summaries={str(j):summaries(ends['archived'+str(j)]) for j in (0,1)},fresh_endpoint_summaries={str(j):summaries(ends['fresh'+str(j)]) for j in (0,1)},preparation_summaries={str(j):summaries(preps[str(j)]) for j in (0,1)},archived_replay_mismatch_records=[r for r in replays if not pair_pass(r)],repeat_mismatch_records=[r for r in repeats if not pair_pass(r)],fresh_feature_diagnostics={name:max(r[name] for ff in features.values() for r in ff) for name in ('capture_Linf_vs_archive','full3211_Linf_vs_archive','raw_Linf_vs_archive','routing_Linf_vs_archive','point_Linf_vs_C196','feature_vs_preparation_observed_Linf')},new_feature_records=168,archived_endpoint_records=168,fresh_endpoint_records=168,preparation_records=168,FP64_full_acceptance_evaluated=False,independent_laboratory_replication=False)
  for n,obj in [('ALL168_ORIGINAL_ENDPOINT_REPLAY_CHECKS.json',replays),('ALL252_REPEAT_CHECKS.json',repeats),('ALL168_FRESH_FEATURE_DIAGNOSTICS.json',features),('ALL168_PREPARATION_RECORDS.json',preps)]:dump(out/n,obj)
  stage='complete';(out/'REPORT_ZH.md').write_text('# C198：冻结C196的模型级重放\n\n'+json.dumps({k:s[k] for k in result}|dict(archived_endpoint_summaries=s['archived_endpoint_summaries'],fresh_endpoint_summaries=s['fresh_endpoint_summaries']),ensure_ascii=False,indent=2)+'\n\n全部84态已暴露。归档端点控制与新采特征全流程分别报告；完成不等于通过，失败不删、不改头、不修复。固定C155参考不重置。未测新确认/排除输入，LC2仍开放。\n')
 except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True);s.update(complete=False,decision='C198_INCOMPLETE_NO_REPLAY_ACCEPTANCE')
 finally:
  if rt is not None:
   try:rt.close()
   except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C198_INCOMPLETE_NO_REPLAY_ACCEPTANCE')
  checks={}
  for k in INPUT_KEYS:
   try:checks[k]=sha(paths[k])==c['pins'][k]['sha256']
   except Exception:checks[k]=False
  s['original_inputs_unchanged']=checks
  if not all(checks.values()):errors.append('Original input missing or changed');s.update(complete=False,decision='C198_INCOMPLETE_NO_REPLAY_ACCEPTANCE')
  checkpoint()
 return dict(s,errors=errors)
def frozen_digest(ctx):
 d=ctx['data']['d'];arrays={sid+'__'+k:v for sid,q in d['points'].items() for k,v in q.items() if isinstance(v,np.ndarray)};arrays.update({'weights__'+sid:w for sid,w in d['weights'].items()});arrays.update({'router__'+k:v for k,v in d['router'].items()});arrays.update(V=d['V'],head=ctx['C196_final_head']);arrays.update({'archived_point__'+sid:pt for sid,pt in ctx['C196_final_points'].items()})
 return dict(state_basis_router_head=safe.head_digest(arrays),raw_writer=safe.head_digest(d['writer']),raw_control=safe.head_digest(d['control']),baseline=hashlib.sha256(json.dumps(ctx['baseline'],sort_keys=True).encode()).hexdigest(),baseline_logits=array_digest(ctx['baseline_logits']['FP32_logits_cast64']))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
 p.add_argument('--out',required=True);a=p.parse_args();s=run({k:getattr(a,k.lower()) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
 if not s['complete']:raise SystemExit(1)
