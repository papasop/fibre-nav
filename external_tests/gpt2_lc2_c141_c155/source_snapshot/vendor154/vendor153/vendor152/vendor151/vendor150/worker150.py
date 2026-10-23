"""Fixed two-arm full84 offline experiment; frozen endpoints retain every failure."""
import argparse,traceback,resource
from core150 import *
from inputs150 import load

def run(paths,out,c=None,runtime_factory=None,backend_factory=None,input_loader=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);start=time.monotonic();stage='registered';d=None;backend=None;pre=[];zeros=[];warms=[];updates=[];records=[];seals={};errors=[];stats_by_arm={arm:dict(steps=0,elapsed_training_seconds=0.) for arm in TRAIN_ARMS}
    calls=dict(model_forwards=0,initialisation_model_forwards=0,preflight_metric_attempts=0,preflight_metrics=0,path_metric_attempts=0,path_metrics=0)
    s=dict(protocol=PROTOCOL,complete=False,decision='C150_INCOMPLETE_COMPARISON_NOT_QUALIFIED',synthetic_fixture=not c['production'],real_model_execution_claim=False,training_states=84,all84_remain_exposed_training=True,requested_training_steps_per_arm=12,requested_model_forwards=6553,requested_scalar_backwards=4032,requested_native_preflight=336,requested_native_endpoints=672,new_states_prepared=0,confirmation_states_evaluated=0,excluded_input_states_evaluated=0,new_model_features_acquired=0,new_Jacobians=0,new_SVD_calls=0,linear_systems_solved=0,GPT2_optimizer_updates=0,LoRA_A_optimizer_updates=0,base_writer_updates=0,teacher_calls=0,backtracking_steps=0,online_response_repair_steps=0,archived_source_executed=False,thresholds_changed=False,parent_decision_reclassified=False,candidate_selected=False,writer_deployed=False,best_checkpoint_selected=False,AD_accuracy_independently_certified=False,router_label_access=False,router_parameters_trained=False,geometric_atlas_established=False,fixed_coupling_offset=1,coupling_pattern_selected=False,parameter_count_confound_explicit=True,LC2_closed=False,LC2_single_fold_passed=False,training_qualified={arm:False for arm in TRAIN_ARMS},scope=design()['scope'])
    def budget():need(time.monotonic()-start<TIMEOUT,'C150 fixed240-minute worker budget')
    def checkpoint(full=False):
        bs=backend.stats() if backend is not None else {};s.update(stage=stage,elapsed_worker_seconds=time.monotonic()-start,process_peak_RSS_KiB=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),runtime_calls=calls,gradient_stats=bs,new_model_forwards=calls['model_forwards'],new_backward_calls=bs.get('scalar_backward_calls',0),new_training_steps=len(updates),new_head_fits=sum(x['steps']>0 for x in stats_by_arm.values()),per_arm_training=stats_by_arm,preflight_records=len(pre),zero_parity_records=len(zeros),warm_parity_records=len(warms),final_endpoint_records=len(records));dump(out/'SUMMARY.json',dict(s,errors=errors))
        if full:
            for name,obj in [('PREFLIGHT_PROGRESS.json',pre),('ZERO_FUNCTIONAL_PROGRESS.json',zeros),('WARM_FUNCTIONAL_PROGRESS.json',warms),('TRAINING_PROGRESS.json',updates),('FINAL_ENDPOINT_PROGRESS.json',records)]:dump(out/name,obj)
    def seal(name):seals[name]=sha(out/name)
    def unchanged():
        for name,h in seals.items():compare(sha(out/name),h,'Frozen artifact/'+name)
        if d is not None:compare(fixed_digest(d),d['fixed_identities'],'Actual frozen input/head/router identities')
    try:
        need(not (out/'SUMMARY.json').exists(),'Fresh C150 output');dump(out/'DESIGN_FROZEN_BEFORE_MODEL.json',design());seal('DESIGN_FROZEN_BEFORE_MODEL.json');checkpoint();stage='original_archive_audit_and_training_only_router'
        d=(load if input_loader is None else input_loader)(paths,out,c);d['fixed_identities']=fixed_digest(d);budget()
        for name in ('INPUTS_FROZEN.json','ALL84_TRAINING_REGISTRY_FROZEN.json','BASELINE_ALL84_ROUTING_AND_FAILURE_COMPARISON.json','FROZEN_TRAINING_ONLY_ROUTER.npz','ALL84_PREWRITE_ROUTING.npz','ALL84_TRAINING_INPUTS.npz','FROZEN_SHARED_ROW_BASIS.npz','INITIAL_C148_ROUTED_CONTROL.npz','FROZEN_ORIGINAL_WRITER.npz','FROZEN_ORIGINAL_CONTROL.npz','MODEL_PINS_REFERENCE.json','FROZEN_INPUT_IDENTITIES.json','PARENT_AUDIT_RECEIPT.json'):need((out/name).is_file(),'Missing bound input/'+name);seal(name)
        need(set(d['points'])=={r['state_id'] for r in registry()},'All84 exact inputs');rank=len(d['initial_a']);s['trainable_parameters']={TRAIN_ARMS[0]:3*rank,TRAIN_ARMS[1]:6*rank};phase=Phase(d);params={arm:initial_parameters(d,arm) for arm in TRAIN_ARMS};mom={arm:(np.zeros_like(p),np.zeros_like(p)) for arm,p in params.items()};zero=np.zeros((3,rank));zero_e=np.zeros(rank)
        if runtime_factory is None:
            from runtime150 import factory
            runtime_factory=factory
        actual=backend_factory is None and input_loader is None and c['production'];stage='model_initialisation';rt=runtime_factory(out/'model_runtime',d,phase,calls);s['real_model_execution_claim']=bool(actual);compare({k:v for k,v in rt.pins.items() if k!='torch'},{k:v for k,v in d['modelpins'].items() if k!='torch'},'Pinned GPT2 identity');stage='all84_native_preflight';native={}
        for sp in registry():
            sid=sp['state_id'];p=d['points'][sid]
            for kind in ('prepared','raw','C146','C148'):
                budget();point=preflight_point(d,sid,kind);write=kind!='prepared';phase.authorize((sid,kind),point,sp['case'],write);met=rt.metrics(point,sp['case'],write);g,_,_=native_gates(met,sp['case'],point,write)
                if not write:need(all(g.values()),'Invalid preparation retained; blocks training')
                else:endpoint_gates(met,sp['case'],point,p['prepared'],True)
                qq=[repeat_check(met,ref) for ref in d['refs'][(sid,kind)]];need(len(qq)>0 and all(q['same_point_identity'] and q['within_replay_tolerance'] for q in qq),'Original native replay drift blocks before backward');pre.append(dict(**sp,kind=kind,metrics=met,comparisons=qq));native[(sid,kind)]=met;checkpoint()
            print('C150_PREFLIGHT',sid,'OK',flush=True)
        unchanged();phase.begin_zero();stage='zero_functional_parity'
        if backend_factory is None:
            from functional150 import factory
            backend_factory=factory
        backend=backend_factory(rt,d,phase,calls)
        def check_parity(sp,p,arm,kind):
            r=backend.parity(sp,p,arm);compare(r['state_id'],sp['state_id']);compare(r['arm'],arm);need(0<=r['point_parity_error_Linf']<=2e-6,'Functional point parity');e=close(r['observed'],native[(sp['state_id'],kind)]['observed'],'Functional/native observations',2e-6);k=abs(r['KL']-native[(sp['state_id'],kind)]['KL']);need(k<=2e-7,'Functional/native KL parity');return dict(r,observed_parity_Linf=e,KL_parity_absolute_error=k)
        for sp in registry():budget();zeros.append(check_parity(sp,zero,TRAIN_ARMS[0],'raw'));checkpoint()
        need(backend.stats()['scalar_backward_calls']==0,'All zero parity before backward');phase.begin_warm();stage='both_initial_heads_functional_parity'
        for arm in TRAIN_ARMS:
            for sp in registry():budget();warms.append(check_parity(sp,params[arm],arm,'C148'));checkpoint()
        need(backend.stats()['scalar_backward_calls']==0,'All warm parity before backward');backend.verify(rehash=True);unchanged();phase.begin_training();s['both_warm_parities_complete_before_backward']=True;stage='fixed12_each_full84_training';checkpoint(True)
        for step in range(STEPS):
            for arm in TRAIN_ARMS:
                budget();t=time.monotonic();p=params[arm];m,v=mom[arm];phase.begin_batch(step,p,arm);folder=out/'training'/arm/f'step{step+1:02d}';g,rows=backend.gradient(p,step,arm,folder,budget);gg,agg=aggregate(rows,p.shape);close(g,gg,'Full84 aggregate',1e-12)
                for sp,r in zip(registry(),rows):
                    for key,value in sp.items():compare(r[key],value,'State binding/'+key)
                    sid=sp['state_id'];a=effective(p,d['weights'][sid],arm);point=proposal(d['points'][sid]['prepared'],d['points'][sid]['raw'],d['V'],a)[0];compare(r['arm'],arm);compare(r['parameters_digest'],row_digest(p));compare(r['effective_control_digest'],row_digest(a));compare(r['routing_weights'],d['weights'][sid].tolist());compare(r['functional_point_sha256'],safe.tensor_hash(point));compare(r['objective_parts'],loss(r['observed'],r['KL'],point,d['points'][sid]['prepared'],a,sp['case']),atol=1e-10,rtol=1e-9)
                    for key in ('response_gradient','guard_gradient'):close(r[key],pullback(np.asarray(r['effective_'+key]),d['weights'][sid],arm),'Exact routing pullback',1e-12)
                phase.end_batch();new,mm,vv,info=update(p,g,m,v,step+1);np.savez_compressed(folder/'UPDATE.npz',before=p,before_m=m,before_v=v,gradient=g,after=new,after_m=mm,after_v=vv);dump(folder/'OBSERVATIONS.json',rows);dump(folder/'AGGREGATION.json',agg);updates.append(dict(arm=arm,**info,objective_before_update=agg['objective'],maximum_response_loss=agg['maximum_response_loss'],maximizing_state_ids=agg['maximizing_state_ids'],before_digest=row_digest(p),after_digest=row_digest(new),update_sha256=sha(folder/'UPDATE.npz'),n_training_records=N));params[arm]=new;mom[arm]=(mm,vv);stats_by_arm[arm]['steps']+=1;stats_by_arm[arm]['elapsed_training_seconds']+=time.monotonic()-t;unchanged();checkpoint(True);print('C150_UPDATE',arm,step+1,'/12 max_response_loss',agg['maximum_response_loss'],flush=True)
        need(backend.stats()['scalar_backward_calls']==4032 and backend.stats()['full84_gradient_batches']==24,'Both full12 schedules required');backend.verify(rehash=True);unchanged();stage='freeze_both_step12_heads';points={};outcomes=[]
        for arm in TRAIN_ARMS:np.savez_compressed(out/('FINAL_'+arm+'_STEP12.npz'),parameters=params[arm]);seal('FINAL_'+arm+'_STEP12.npz')
        for sp in registry():
            sid=sp['state_id'];p=d['points'][sid]
            for arm in EVAL_ARMS:
                a=zero_e if arm==EVAL_ARMS[0] else effective(d['initial_A'],d['weights'][sid],TRAIN_ARMS[0]) if arm==EVAL_ARMS[1] else effective(params[arm],d['weights'][sid],arm);point=p['raw'].copy() if arm==EVAL_ARMS[0] else proposal(p['prepared'],p['raw'],d['V'],a)[0]
                if arm in TRAIN_ARMS:
                    fresh,context_diag=context_proposal(p['prepared'],p['raw'],d['V'],params[arm],d['router'],arm);exact(fresh,point,'Cached/fresh prewrite routing point');compare(context_diag['routing_weights'],d['weights'][sid].tolist());close(backend.proposed(params[arm],sid,arm),point,'Frozen final NumPy/Torch map parity',2e-6)
                points[(sid,arm)]=point;outcomes.append(dict(**sp,arm=arm,routing_weights=d['weights'][sid].tolist(),effective_control=a.tolist(),point_sha256=safe.tensor_hash(point)))
        np.savez_compressed(out/'ALL336_FROZEN_FINAL_POINTS.npz',**{sid+'__'+arm:p for (sid,arm),p in points.items()});dump(out/'ALL336_FROZEN_FINAL_OUTCOMES.json',outcomes)
        for name in ('ALL336_FROZEN_FINAL_POINTS.npz','ALL336_FROZEN_FINAL_OUTCOMES.json'):seal(name)
        dump(out/'BOTH_STEP12_FREEZE_BEFORE_EVALUATION.json',dict(completed_steps=phase.completed,parameter_digests={a:row_digest(p) for a,p in params.items()},artifact_seals=dict(seals),native_final_measurements_so_far=0,no_arm_or_checkpoint_selection=True));seal('BOTH_STEP12_FREEZE_BEFORE_EVALUATION.json');phase.freeze(points);s['both_step12_heads_frozen_before_evaluation']=True;stage='two_round672_native_endpoints';first={};drifts=[];repeats=[]
        for rnd in (0,1):
            for sp in registry():
                sid=sp['state_id'];p=d['points'][sid]
                for arm in EVAL_ARMS:
                    budget();point=points[(sid,arm)];phase.authorize((rnd,sid,arm),point,sp['case'],True);met=rt.metrics(point,sp['case'],True);g=endpoint_gates(met,sp['case'],point,p['prepared'],True);a=zero_e if arm==EVAL_ARMS[0] else effective(d['initial_A'],d['weights'][sid],TRAIN_ARMS[0]) if arm==EVAL_ARMS[1] else effective(params[arm],d['weights'][sid],arm);r=dict(**sp,round=rnd,arm=arm,metrics=met,gates=g,training_objective_parts=loss(met['observed'],met['KL'],point,p['prepared'],a,sp['case']))
                    if arm in EVAL_ARMS[:2]:
                        q=repeat_check(met,native[(sid,'raw' if arm==EVAL_ARMS[0] else 'C148')]);r['control_replay']=q
                        if not(q['same_point_identity'] and q['within_replay_tolerance']):drifts.append(dict(state_id=sid,arm=arm,round=rnd,comparison=q))
                    if rnd==0:first[(sid,arm)]=met
                    else:
                        q=repeat_check(met,first[(sid,arm)]);r['repeat_comparison']=q
                        if not(q['same_point_identity'] and q['within_replay_tolerance']):repeats.append(dict(state_id=sid,arm=arm,comparison=q))
                    records.append(r);dump(out/'final_measurements'/f'record{len(records)-1:03d}.json',r);checkpoint()
                if len(records)%48==0:unchanged();checkpoint(True)
                print('C150_FINAL',rnd,sid,[(r['arm'],r['gates']['complete_pass']) for r in records[-4:]],flush=True)
        phase.finish();unchanged();backend.verify(rehash=True,check_chart=False);need(calls['model_forwards']==6553 and calls['preflight_metrics']==336 and calls['path_metrics']==672,'Full fixed model/native denominator');need(backend.stats()['functional_endpoint_calls']==2268,'Full functional denominator');sums,pairs=final_tables(records);consistent=not drifts and not repeats;qualified={arm:consistent and all(sums['all84'][str(r)][arm]['passed']==84 for r in (0,1)) for arm in TRAIN_ARMS}
        for name,obj in [('ALL672_FINAL_TRAINING_ENDPOINTS.json',records),('FINAL_TRAINING_SUMMARIES.json',sums),('PAIRED_ALL_ARMS.json',pairs),('CONTROL_REPLAY_FAILURES.json',drifts),('REPEATABILITY_FAILURES.json',repeats)]:dump(out/name,obj)
        s.update(complete=True,decision='C150_FIXED_FINAL_COMPARISON_COMPLETE_NO_CANDIDATE_SELECTED' if consistent else 'C150_COMPARISON_COMPLETE_REPLAY_DRIFT_NO_QUALIFICATION',training_qualified=qualified,repeatable=not repeats,controls_replay_consistent=not drifts,final_training_summaries=sums,paired_all_arms=pairs,next_branch='PREREGISTER_NEW_CONFIRMATION_FOR_QUALIFIED_ARMS' if any(qualified.values()) else 'RETAIN_TRAINING_FAILURE_NO_UNSEEN_EVALUATION');stage='complete'
        lines=['# C150：固定循环跨坐标耦合对照','','两臂使用同一个C148最终路由头初始化；耦合臂的新增交叉项从零开始。固定原路由、V、全部84训练态及原验收门槛；各12步，只验收固定最后一步。','','|臂|首轮通过|重复轮通过|首轮最坏响应漂移|','|---|---:|---:|---:|']
        for arm in EVAL_ARMS:lines.append(f"|{arm}|{sums['all84']['0'][arm]['passed']}/84|{sums['all84']['1'][arm]['passed']}/84|{sums['all84']['0'][arm]['maximum_response_drift']}|")
        lines+=['','判定：'+s['decision'],'训练资格：'+json.dumps(qualified),'','对照臂保持逐坐标缩放；实验臂增加b(u)*roll(u,1)。固定偏移1，不筛选耦合关系。参数从108增加至216，存在容量和优化几何差异；不把效果独立归因于耦合，不证明对角控制不可行。','','不只拟合五个失败态，保留全部84态与获救/破坏记录。未测新状态或排除类，不选臂、不部署。即使训练84/84，也仍须新的前瞻确认及严格隔离的排除输入评估。'];(out/'REPORT_ZH.md').write_text('\n'.join(lines)+'\n')
    except BaseException:errors.append(traceback.format_exc());s.update(complete=False,decision='C150_INCOMPLETE_COMPARISON_NOT_QUALIFIED',training_qualified={arm:False for arm in TRAIN_ARMS});print(errors[-1],flush=True)
    finally:
        checks={}
        for k,p in paths.items():
            try:checks[k]=sha(p)==c['pins'][k]['sha256']
            except Exception:checks[k]=False
        s['original_inputs_unchanged']=checks
        if set(checks)!=set(INPUT_KEYS) or not all(checks.values()):s.update(complete=False,decision='C150_INCOMPLETE_COMPARISON_NOT_QUALIFIED',training_qualified={arm:False for arm in TRAIN_ARMS});errors.append('Original input missing or changed')
        checkpoint(True)
    return dict(s,errors=errors)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in INPUT_KEYS:p.add_argument('--'+k.lower(),required=True)
    p.add_argument('--out',required=True);args=p.parse_args();s=run({k:Path(getattr(args,k.lower())) for k in INPUT_KEYS},args.out);console={k:v for k,v in s.items() if k not in ('paired_all_arms','final_training_summaries')};console['final_training_summaries']={'all84':s.get('final_training_summaries',{}).get('all84',{})};print(json.dumps(console,ensure_ascii=False,indent=2),flush=True)
    if not s['complete']:raise SystemExit(1)
