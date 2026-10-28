"""Known84 preflight, then fixed48 prospective confirmation; no fallback."""
import argparse,resource,traceback
from core199 import *
from inputs199 import load
from novelty199 import inventory
from worker198 import frozen_digest as prior_digest
class PreflightStop(Exception):pass
def frozen_digest(ctx):
 r=prior_digest(ctx);r['C198_point_controls']=safe.head_digest({'r'+j+'__'+sid:pt for j,points in ctx['C198_fresh_points'].items() for sid,pt in points.items()});r['C198_metric_controls']=hashlib.sha256(json.dumps(ctx['C198_fresh_native'],sort_keys=True,allow_nan=False).encode()).hexdigest();return r
def run(paths,out,c=None,input_loader=None,runtime_factory=None,history_loader=None):
 c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic();stage='registered';errors=[];seals={};ctx=None;rt=None;records=[];at=0;known={};known_points={};prep={};features={};newpoints={};endpoints={};repeats=[];new_exposed=set()
 calls=dict(model_forwards=0,initialization_forwards=0,feature_attempts=0,feature_calls=0,metric_attempts=0,metric_calls=0)
 s=dict(protocol=PROTOCOL,complete=False,decision='C199_INCOMPLETE_NO_CONFIRMATION',synthetic_fixture=not c['production'],real_model_execution_claim=False,all44_originals_independently_bound=False,history_inventory_bound44_originals=False,known_preflight_passed=False,confirmation_states_requested=48,confirmation_states_evaluated=0,new_states_prepared=0,new_centers_prepared=0,confirmation_states_exposed=0,local_confirmation_passed=False,confirmation_evaluation_complete=False,all48_coordinates_sealed_before_model=False,all96_proposals_sealed_before_new_endpoints=False,programme_wide_nonaccess_proved=False,chronology_proved=False,scope=SCOPE,**no_training())
 def checkpoint():
  s.update(stage=stage,records_completed=len(records),requests_consumed=at,new_model_forwards=calls['model_forwards'],runtime_calls=dict(calls),confirmation_states_exposed=len(new_exposed),elapsed_worker_seconds=time.monotonic()-started,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss));dump(out/'SUMMARY.json',dict(s,errors=errors))
 def budget():
  need(time.monotonic()-started<TIMEOUT,'C199 fixed240-minute worker budget')
  for n,h in seals.items():compare(sha(out/n),h,'Frozen C199 artifact/'+n)
  if ctx is not None:compare(frozen_digest(ctx),frozen,'Frozen proposal pipeline and C198 controls')
 def authorize(st,sid):
  nonlocal at
  budget();need(at<len(requests()),'No unregistered request');compare(requests()[at],dict(stage=st,state_id=sid),'Exact conditional624 registry');at+=1
 def save(r):
  records.append(r)
  with (out/'ALL_RECORDS.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
  if len(records)%12==0:checkpoint()
 def seal(n):seals[n]=sha(out/n)
 def store_proposal(prefix,sid,point,capture,observed):
  q=fresh_proposal(d,head,point,capture);n=prefix+'/'+sid+'.npz';p=out/n;p.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(p,prepared=point,capture3147=capture,full3211=q['full'],prewrite_observed=observed,raw=q['raw'],weights=q['weights'],effective=q['effective'],point=q['point']);seal(n);return q
 try:
  need(not(out/'SUMMARY.json').exists(),'Fresh output only');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());seal('DESIGN_FROZEN_BEFORE_MODEL.json');checkpoint();compare(set(paths),set(INPUT_KEYS),'Exactly44 fixed originals')
  for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Whole input before payload/'+k)
  for n,h in c['source198'].items():compare(sha(ROOT/'vendor198'/n),h,'Immutable delivered source/'+n)
  stage='original_binding';ctx=(load if input_loader is None else input_loader)(paths,out/'binding',c,lambda:need(time.monotonic()-started<TIMEOUT,'Binding budget'));frozen=frozen_digest(ctx);d=ctx['data']['d'];head=ctx['C196_final_head'];check_head(head,c['production']);s['all44_originals_independently_bound']=input_loader is None;dump(out/'ORIGINAL_INPUTS.json',{k:dict(path=str(v),sha256=sha(v)) for k,v in paths.items()});dump(out/'FROZEN_PIPELINE_DIGESTS.json',frozen);np.savez_compressed(out/'FIXED_C196_FINAL_HEAD.npz',P=head)
  for n in ('ORIGINAL_INPUTS.json','FROZEN_PIPELINE_DIGESTS.json','FIXED_C196_FINAL_HEAD.npz'):seal(n)
  stage='bound_history_inventory';history=(inventory if history_loader is None else history_loader)(paths,ctx,out,budget);s['history_inventory_bound44_originals']=history_loader is None
  if (out/'BOUND_HISTORY_NOVELTY_INVENTORY.json').exists():seal('BOUND_HISTORY_NOVELTY_INVENTORY.json')
  dump(out/'WRITER_FROZEN_BEFORE_NEW_COORDINATES.json',dict(artifact_seals=dict(seals),head_digest=parameter_hash(head),model_forwards_so_far=0));seal('WRITER_FROZEN_BEFORE_NEW_COORDINATES.json');points,bindings=generate(d,history);s['new_states_prepared']=48;np.savez_compressed(out/'ALL48_PREPARED_COORDINATES_BEFORE_MODEL.npz',**points);dump(out/'ALL48_COORDINATE_BINDINGS.json',bindings);dump(out/'ALL48_CONFIRMATION_REGISTRY.json',registry());dump(out/'ALL624_REQUESTS_FROZEN.json',requests())
  for n in ('ALL48_PREPARED_COORDINATES_BEFORE_MODEL.npz','ALL48_COORDINATE_BINDINGS.json','ALL48_CONFIRMATION_REGISTRY.json','ALL624_REQUESTS_FROZEN.json'):seal(n)
  dump(out/'NEW_COORDINATES_SEALED_BEFORE_MODEL.json',dict(artifact_seals=dict(seals),model_forwards_so_far=0,new_states=48));seal('NEW_COORDINATES_SEALED_BEFORE_MODEL.json');s.update(all48_coordinates_sealed_before_model=True,fixed_final_head_digest=parameter_hash(head));budget();stage='pinned_CPU_model_initialization';checkpoint()
  if runtime_factory is None:
   from runtime199 import factory
   runtime_factory=factory
  rt=runtime_factory(out/'model_runtime',ctx,calls);s['real_model_execution_claim']=bool(c['production'] and input_loader is None)
  with safe.no_solvers():
   stage='known_prepare';rr=[]
   for sid in IDS:
    authorize(stage,sid);point=d['points'][sid]['prepared'];r=dict(stage=stage,state_id=sid,case=sid.split('__')[0],**metric_record(rt.metrics(point,sid.split('__')[0],False),point,point,sid.split('__')[0],True,False));rr.append(r);save(r)
   known[stage]=rr;stage='known_features';ff=[]
   for i,sid in enumerate(IDS):
    authorize(stage,sid);case=sid.split('__')[0];capture,o=rt.features(d['points'][sid]['prepared'],int(case[-2])^int(case[-1]));farray(capture,(3147,));farray(o,(22,));compare(float(capture[3076]),float(int(case[-2])^int(case[-1])));q=store_proposal('known_prewrite',sid,d['points'][sid]['prepared'],capture,o);known_points[sid]=q['point'].copy();r=dict(stage=stage,state_id=sid,case=case,point_sha256=safe.tensor_hash(q['point']),feature_preparation_observed_Linf=float(np.max(abs(o-np.asarray(known['known_prepare'][i]['metrics']['observed'])))),point_Linf_vs_C198=float(np.max(abs(q['point'].astype(float)-ctx['C198_fresh_points']['0'][sid].astype(float)))));ff.append(r);save(r)
   known[stage]=ff;np.savez_compressed(out/'ALL84_KNOWN_FRESH_PROPOSALS.npz',**known_points);seal('ALL84_KNOWN_FRESH_PROPOSALS.npz');dump(out/'KNOWN_PROPOSALS_SEALED_BEFORE_ENDPOINTS.json',dict(artifact_seals=dict(seals),known_postwrite_measurements=0));seal('KNOWN_PROPOSALS_SEALED_BEFORE_ENDPOINTS.json');replay=[]
   for arm in ('known_archived','known_fresh'):
    stage=arm;rr=[]
    for i,sid in enumerate(IDS):
     authorize(stage,sid);point=ctx['C198_fresh_points']['0'][sid] if arm=='known_archived' else known_points[sid];expected=safe.tensor_hash(point);m=rt.metrics(point,sid.split('__')[0],True);compare(safe.tensor_hash(point),expected,'Actual known point unchanged after metric');r=dict(stage=stage,state_id=sid,case=sid.split('__')[0],**metric_record(m,point,d['points'][sid]['prepared'],sid.split('__')[0],known['known_prepare'][i]['qualification']['complete_pass']));r['C198_comparison']=replay_pair(ctx['C198_fresh_native']['0'][i]['metrics'],m)
     if arm=='known_archived':replay.append(dict(state_id=sid,**r['C198_comparison']))
     rr.append(r);save(r)
    known[stage]=rr
   controlpass=all(r['qualification']['complete_pass'] for st in ('known_prepare','known_archived','known_fresh') for r in known[st]) and all(pair_pass(r) for r in replay) and all(r['feature_preparation_observed_Linf']<=.0002 for r in known['known_features']);s.update(known_preflight_complete=True,known_preflight_passed=bool(controlpass),known_preparation_summary=summarize(known['known_prepare'],IDS),known_archived_summary=summarize(known['known_archived'],IDS),known_fresh_summary=summarize(known['known_fresh'],IDS),known_replay_mismatches=[r for r in replay if not pair_pass(r)]);dump(out/'KNOWN84_PREFLIGHT_RECORDS.json',known);dump(out/'KNOWN84_REPLAY_CHECKS.json',replay);budget();rt.verify(True)
   if not controlpass:
    s.update(complete=True,decision='C199_COMPLETE_PREFLIGHT_FAILURE_NO_NEW_STATE_MEASUREMENT',next_branch='RESOLVE_KNOWN_REPLAY_BEFORE_NEW_CONFIRMATION',confirmation_evaluation_complete=False);stage='preflight_stopped_before_new_measurements';checkpoint();raise PreflightStop()
   newids=[r['state_id'] for r in registry()];byid={r['state_id']:r for r in registry()}
   for rnd in (0,1):
    stage='prepare'+str(rnd);rr=[]
    for sid in newids:
     authorize(stage,sid);new_exposed.add(sid);point=points[sid];case=byid[sid]['case'];r=dict(stage=stage,state_id=sid,case=case,round=rnd,**metric_record(rt.metrics(point,case,False),point,point,case,True,False));rr.append(r);save(r)
    prep[str(rnd)]=rr;stage='features'+str(rnd);ff=[]
    for i,sid in enumerate(newids):
     authorize(stage,sid);case=byid[sid]['case'];capture,o=rt.features(points[sid],int(case[-2])^int(case[-1]));farray(capture,(3147,));farray(o,(22,));compare(float(capture[3076]),float(int(case[-2])^int(case[-1])));q=store_proposal('new_prewrite/round%d'%rnd,sid,points[sid],capture,o);newpoints[(rnd,sid)]=q['point'].copy();r=dict(stage=stage,state_id=sid,case=case,round=rnd,prepared_sha256=safe.tensor_hash(points[sid]),point_sha256=safe.tensor_hash(q['point']),feature_preparation_observed_Linf=float(np.max(abs(o-np.asarray(prep[str(rnd)][i]['metrics']['observed'])))),endpoint_used_for_proposal=False);ff.append(r);save(r)
    features[str(rnd)]=ff;checkpoint()
   np.savez_compressed(out/'ALL96_NEW_PROPOSALS_BEFORE_ENDPOINTS.npz',**{'round%d__%s'%(j,sid):newpoints[(j,sid)] for j in (0,1) for sid in newids});seal('ALL96_NEW_PROPOSALS_BEFORE_ENDPOINTS.npz');dump(out/'NEW_PROPOSALS_SEALED_BEFORE_ENDPOINTS.json',dict(artifact_seals=dict(seals),new_postwrite_measurements=0,requests_consumed=at,records_completed=len(records)));seal('NEW_PROPOSALS_SEALED_BEFORE_ENDPOINTS.json');s['all96_proposals_sealed_before_new_endpoints']=True;budget();valid={sid:all(prep[str(j)][i]['qualification']['complete_pass'] for j in (0,1)) for i,sid in enumerate(newids)};newhashes={(j,sid):safe.tensor_hash(point) for (j,sid),point in newpoints.items()}
   for rnd in (0,1):
    stage='endpoint'+str(rnd);rr=[]
    for sid in newids:
     authorize(stage,sid);case=byid[sid]['case'];point=newpoints[(rnd,sid)];compare(safe.tensor_hash(point),newhashes[(rnd,sid)],'Actual sealed new proposal');m=rt.metrics(point,case,True);compare(safe.tensor_hash(point),newhashes[(rnd,sid)],'New proposal unchanged after metric');r=dict(stage=stage,state_id=sid,case=case,round=rnd,**metric_record(m,point,points[sid],case,valid[sid]));rr.append(r);save(r)
    endpoints[str(rnd)]=rr;dump(out/('NEW_ENDPOINTS_ROUND%d.json'%rnd),rr);s['confirmation_states_evaluated']=len({r['state_id'] for rr in endpoints.values() for r in rr});checkpoint();print('C199_NEW_ENDPOINT',rnd,summarize(rr,newids),flush=True)
   for kind,table in [('preparation',prep),('endpoint',endpoints)]:
    for a,b in zip(table['0'],table['1']):repeats.append(dict(kind=kind,state_id=a['state_id'],**replay_pair(a['metrics'],b['metrics'])))
  compare(at,624);compare(len(records),624);compare(calls['feature_calls'],180);compare(calls['metric_calls'],444);compare(calls['model_forwards'],FORWARDS);rt.verify(True);budget();parity=all(r['feature_preparation_observed_Linf']<=.0002 for rr in features.values() for r in rr);repeated=all(pair_pass(r) for r in repeats);passed=all(valid.values()) and all(summarize(endpoints[str(j)],newids)['passed']==48 for j in (0,1)) and repeated and parity
  s.update(complete=True,decision='C199_FROZEN_LOCAL_CONFIRMATION_ALL48_PASS_NO_EXCLUDED_INPUT' if passed else 'C199_COMPLETE_LOCAL_CONFIRMATION_FAILURE_NO_EXCLUDED_INPUT',local_confirmation_passed=bool(passed),confirmation_evaluation_complete=True,preparation_summaries={str(j):summarize(prep[str(j)],newids) for j in (0,1)},endpoint_summaries={str(j):summarize(endpoints[str(j)],newids) for j in (0,1)},per_case_endpoint_summaries={case:{str(j):summarize([r for r in endpoints[str(j)] if r['case']==case],[sid for sid in newids if byid[sid]['case']==case]) for j in (0,1)} for case in ('xor_00','xor_01','xor_11')},valid_preparations_both_rounds=sum(valid.values()),native_repeats_consistent=bool(repeated),repeat_mismatches=[r for r in repeats if not pair_pass(r)],feature_preparation_parity_consistent=bool(parity),next_branch='PREREGISTER_EXCLUDED_INPUT_PROVENANCE_AND_EVALUATION' if passed else 'RETAIN_FAILURE_AND_EXPOSURE_NO_ADAPTATION_WITHIN_CONFIRMATION',all_C199_measured_states_exposed_after_run=True,FP64_full_acceptance_evaluated=False)
  for n,obj in [('ALL96_NEW_PREPARATIONS.json',prep),('ALL96_NEW_FEATURE_DIAGNOSTICS.json',features),('ALL96_NEW_ENDPOINTS.json',endpoints),('ALL96_NEW_REPEAT_CHECKS.json',repeats)]:dump(out/n,obj)
  stage='complete';(out/'REPORT_ZH.md').write_text('# C199：冻结写入器48态新方向确认\n\n'+json.dumps({k:s[k] for k in ('decision','local_confirmation_passed','endpoint_summaries','preparation_summaries','per_case_endpoint_summaries','repeat_mismatches','next_branch')},ensure_ascii=False,indent=2)+'\n\n固定12已知中心、两档原幅度、每档两方向；共48态，非新中心或排除类。完整失败保留。后续若使用这些观测适配，不能重称未见测试。LC2仍开放。\n')
 except PreflightStop:pass
 except BaseException:errors.append(traceback.format_exc());print(errors[-1],flush=True);s.update(complete=False,decision='C199_INCOMPLETE_NO_CONFIRMATION')
 finally:
  if rt is not None:
   try:rt.close()
   except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C199_INCOMPLETE_NO_CONFIRMATION')
  checks={}
  for k in INPUT_KEYS:
   try:checks[k]=sha(paths[k])==c['pins'][k]['sha256']
   except Exception:checks[k]=False
  s['original_inputs_unchanged']=checks
  if not all(checks.values()):errors.append('Original input missing or changed');s.update(complete=False,decision='C199_INCOMPLETE_NO_CONFIRMATION')
  checkpoint()
 return dict(s,errors=errors)
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
 p.add_argument('--out',required=True);a=p.parse_args();s=run({k:getattr(a,k.lower()) for k in INPUT_KEYS},a.out);print(json.dumps(s,ensure_ascii=False,indent=2),flush=True)
 if not s['complete']:raise SystemExit(1)
