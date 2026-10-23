"""Reconstruct fixed C141 writes from original C140 tensors; never execute ZIP source."""
from core142 import *

def arrays_equal(actual,expected,label):
    need(set(actual)==set(expected),label+' tensor schema')
    for k in actual:exact(actual[k],expected[k],label+'/'+k)

def source_check(z,c):
    names={n[7:] for n in z.z.namelist() if n.startswith('source/')};need(names==set(c['source141']),'C141 delivered source inventory differs')
    for n,h in c['source141'].items():compare(hashlib.sha256(z.raw('source/'+n)).hexdigest(),h,'Trusted C141 source/'+n)

def audit(paths,out,c=None):
    c=contract() if c is None else c;out=Path(out);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    need(set(paths)=={'C141','C140'},'Exactly fixed C141 and C140 originals required')
    for k,p in paths.items():compare(sha(p),c['pins'][k]['sha256'],'Fixed input SHA256/'+k)
    dump(out/'INPUTS_FROZEN.json',{k:dict(path=str(p),sha256=sha(p)) for k,p in paths.items()})
    def budget():need(time.monotonic()-started<TIMEOUT,'C142 fixed20-minute audit budget exceeded')
    with contextlib.ExitStack() as stack:
        z=stack.enter_context(contextlib.closing(Archive(paths['C141'])));p=stack.enter_context(contextlib.closing(Archive(paths['C140'])))
        member_counts={k:a.verify_manifest() for k,a in [('C141',z),('C140',p)]};source_check(z,c);budget()
        s=z.json('SUMMARY.json');ps=p.json('SUMMARY.json');compare(s['protocol'],PARENT);compare(s['complete'],True);compare(ps['protocol'],prior.parent.PROTOCOL);compare(ps['complete'],True);compare(ps['decision'],'C140_FIXED_FINAL_ALL36_TRAINING_GATES_PASSED')
        if c['production']:
            compare(s['synthetic_fixture'],False);compare(s['real_model_execution_claim'],True);compare(ps['synthetic_fixture'],False);compare(ps['real_model_execution_claim'],True)
        for k in ('candidate_frozen_before_new_state_generation','all48_coordinates_frozen_before_model','all_proposals_frozen_before_new_endpoint_measurements'):compare(s[k],True,'Archived freeze declaration/'+k)
        for k in ('new_training_steps','new_backward_calls','new_head_fits','new_Jacobians','new_SVD_calls','linear_systems_solved','teacher_calls','backtracking_steps','online_response_repair_steps','excluded_input_states_evaluated'):compare(s[k],0)
        for k in ('candidate_selected','writer_deployed','thresholds_changed','parent_decision_reclassified','LC2_closed','LC2_single_fold_passed','archived_source_executed'):compare(s[k],False)
        compare(z.json('DESIGN_FROZEN_BEFORE_MODEL.json'),prior.design(),'Frozen C141 design')
        for k,v in prior.contract()['pins'].items():compare(z.json('INPUTS_FROZEN.json')[k]['sha256'],v['sha256'] if c['production'] else z.json('INPUTS_FROZEN.json')[k]['sha256'])
        compare(z.json('INPUTS_FROZEN.json')['C140']['sha256'],c['pins']['C140']['sha256'],'C140 linked original')
        frozen=z.arrays('FROZEN_SHARED_CONTROLS.npz');need(set(frozen)=={'V','C137_a','C140_a'},'Frozen shared control schema')
        v=farray(frozen['V'],(36 if c['production'] else len(frozen['V']),DIM));need(len(v)>0 and np.max(abs(v@v.T-np.eye(len(v))))<=1e-8,'Frozen row basis invalid')
        old=farray(frozen['C137_a'],(len(v),));new=farray(frozen['C140_a'],(len(v),));need(np.max(abs(old))<=.125 and np.max(abs(new))<=.125,'Frozen gain bound exceeded')
        exact(v,p.arrays('FROZEN_SHARED_ROW_BASIS.npz')['V'],'C140 original basis');exact(old,p.arrays('INITIAL_C137_STEP24_CONTROL.npz')['a'],'C137 fixed original vector');exact(new,p.arrays('FINAL_SHARED_CONTROL_STEP12.npz')['a'],'C140 fixed final vector');exact(new,p.arrays('training/step12/UPDATE.npz')['after_a'],'C140 actual update12 vector')
        freeze=p.json('STEP12_FREEZE_BEFORE_EVALUATION.json');compare(freeze['completed_steps'],12);compare(freeze['checkpoint_choice'],'Fixed C140 step12 only');compare(freeze['no_best_or_recovery_selection'],True);compare(freeze['control_digest'],row_digest(new));compare(freeze['control_sha256'],hashlib.sha256(p.raw('FINAL_SHARED_CONTROL_STEP12.npz')).hexdigest())
        writer=z.arrays('FROZEN_RAW_WRITER.npz');control=z.arrays('FROZEN_ORIGINAL_CONTROL.npz');arrays_equal(writer,p.arrays('FROZEN_ORIGINAL_WRITER.npz'),'Original raw writer');arrays_equal(control,p.arrays('FROZEN_ORIGINAL_CONTROL.npz'),'Original control')
        farray(writer['mean'],(3147,));farray(writer['scale'],(3147,));need(np.all(writer['scale']>0),'Nonpositive frozen feature scale');need(set(writer)>={'mean','scale','delta_mean','training_z','coefficients'},'Raw writer fields')
        farray(writer['delta_mean'],(DIM,));farray(writer['training_z'],(len(writer['training_z']),3147));farray(writer['coefficients'],(len(writer['training_z']),DIM));farray(control['delta_sketch'],(DIM,64))
        heads=dict(raw_writer=safe.head_digest(writer),control=safe.head_digest(control),V=row_digest(v),C137_a=row_digest(old),C140_a=row_digest(new))
        # Parent head field names are fixed by the delivered C141/core139 implementation.
        hd=z.json('CANDIDATE_FROZEN_BEFORE_NEW_STATES.json')['heads']
        data=dict(writer=writer,control=control,V=v,a=old,initial_a=old,a140=new)
        compare(hd,prior.head_digests(data),'Actual frozen tensor identities')
        compare(z.json('CANDIDATE_FROZEN_BEFORE_NEW_STATES.json')['checkpoint'],'C140 fixed step12 only')
        seals=z.json('PROPOSALS_SEALED_BEFORE_NEW_ENDPOINTS.json');compare(seals['heads'],hd);compare(seals['new_states'],48);compare(seals['proposal_points'],144);compare(seals['new_writer_endpoint_measurements_so_far'],0)
        expected_seals={'DESIGN_FROZEN_BEFORE_MODEL.json','INPUTS_FROZEN.json','BOUND_HISTORICAL_COORDINATE_HASHES.json','C140_BOUND_RECEIPT.json','FROZEN_RAW_WRITER.npz','FROZEN_ORIGINAL_CONTROL.npz','FROZEN_SHARED_CONTROLS.npz','CANDIDATE_FROZEN_BEFORE_NEW_STATES.json','ALL48_PREPARED_COORDINATES_BEFORE_MODEL.npz','ALL48_CONFIRMATION_REGISTRY.json','ALL144_PROPOSALS_BEFORE_ENDPOINTS.npz','ALL144_PROPOSAL_OUTCOMES.json'}
        need(set(seals['artifact_seals'])==expected_seals,'Frozen proposal seal coverage differs')
        for n,h in seals['artifact_seals'].items():compare(hashlib.sha256(z.raw(n)).hexdigest(),h,'Frozen pre-endpoint seal/'+n)
        all36=p.arrays('ALL36_TRAINING_INPUTS.npz');need(set(all36)=={r['state_id']+'__'+k for r in prior.parent.registry() for k in ('prepared','raw','features')},'C140 all36 input coverage')
        data['points']={r['state_id']:{k:farray(all36[r['state_id']+'__'+k],(3211,) if k=='features' else (DIM,)) for k in ('prepared','raw','features')} for r in prior.parent.registry()}
        data['historical_hashes']=set(z.json('BOUND_HISTORICAL_COORDINATE_HASHES.json')['hashes'])
        generated,reg=prior.fresh_coordinates(data);compare(z.json('ALL48_CONFIRMATION_REGISTRY.json'),reg,'Reconstructed48 registry/coordinates')
        saved=z.arrays('ALL48_PREPARED_COORDINATES_BEFORE_MODEL.npz');arrays_equal(saved,generated,'All48 actual float32 coordinates');budget()
        preps=z.json('ALL96_NEW_PREPARATIONS.json');compare(z.json('PREPARATION_PROGRESS.json'),preps);need(len(preps)==96,'All96 preparation records required');by={};repeat_fail=[]
        for rec,(rnd,sp) in zip(preps,[(j,r) for j in (0,1) for r in prior.registry()]):
            for k,x in dict(**sp,round=rnd).items():compare(rec[k],x,'Preparation identity/'+k)
            g,_,_=native_gates(rec['metrics'],sp['case'],saved[sp['state_id']],False);valid=bool(all(g.values()));compare(rec['preparation_valid'],valid)
            if rnd==1:
                rep=repeat_check(rec['metrics'],by[(0,sp['state_id'])]['metrics']);compare(rec['repeat_comparison'],rep)
                if not(rep['same_point_identity'] and rep['within_replay_tolerance']):repeat_fail.append(dict(kind='preparation',state_id=sp['state_id'],comparison=rep))
            by[(rnd,sp['state_id'])]=rec
        valids={r['state_id']:all(by[(j,r['state_id'])]['preparation_valid'] for j in (0,1)) for r in prior.registry()}
        psums={str(j):prior.preparation_summary([r for r in preps if r['round']==j]) for j in (0,1)};compare(z.json('PREPARATION_SUMMARIES.json'),psums);compare(s['preparation_summaries'],psums)
        props=z.arrays('ALL144_PROPOSALS_BEFORE_ENDPOINTS.npz');outcomes=[];maps={};features=[];full_errors=[]
        need(set(props)=={r['state_id']+'__'+a for r in prior.registry() for a in ARMS},'All144 proposal coverage')
        for index,sp in enumerate(prior.registry()):
            budget();sid=sp['state_id'];f=z.arrays('prewrite/'+sid+'.npz');need(set(f)=={'prepared','raw','C137','C140','capture3147','write_features3211','observed'},'Prewrite tensor schema')
            exact(f['prepared'],saved[sid],'Prewrite coordinate binding');x=farray(f['capture3147'],(3147,));farray(f['observed'],(22,));compare(float(x[3076]),float(int(sp['case'][-2])^int(sp['case'][-1])));close(f['observed'],by[(0,sid)]['metrics']['observed'],'Captured prewrite/native observation parity',.0002)
            raw,full=safe.reconstruct_raw(writer,control['delta_sketch'],saved[sid],x);farray(f['write_features3211'],(3211,));exact(f['write_features3211'][:3147],x,'Full3211 capture prefix');full_errors.append(close(f['write_features3211'],full,'Full3211 feature including64 tail',1e-10));exact(f['raw'],raw,'Original raw proposal')
            oldp,od=prior.parent.proposal(saved[sid],raw,v,old);newp,nd=prior.parent.proposal(saved[sid],raw,v,new)
            for arm,point,diag,key in [(ARMS[0],raw,dict(original_raw_rule=True),'raw'),(ARMS[1],oldp,od,'C137'),(ARMS[2],newp,nd,'C140')]:
                exact(f[key],point,'Actual frozen per-state proposal');exact(props[sid+'__'+arm],point,'Sealed combined proposal');maps[(sid,arm)]=point;outcomes.append(dict(**sp,arm=arm,point_sha256=safe.tensor_hash(point),preparation_valid_both_rounds=valids[sid],proposal_diagnostics=diag))
            features.append(dict(**sp,prepared_sha256=safe.tensor_hash(saved[sid]),raw_sha256=safe.tensor_hash(raw),C137_sha256=safe.tensor_hash(oldp),C140_sha256=safe.tensor_hash(newp),full_feature_digest=row_digest(full),preparation_valid_both_rounds=valids[sid]))
            if (index+1)%12==0:print('C142_MAP_AUDIT',index+1,'/48',flush=True)
        compare(z.json('ALL144_PROPOSAL_OUTCOMES.json'),outcomes);compare(z.json('PREWRITE_PROGRESS.json'),features)
        records=z.json('ALL288_CONFIRMATION_ENDPOINTS.json');compare(z.json('ENDPOINT_PROGRESS.json'),records);need(len(records)==288,'All288 endpoint records required');first={}
        for i,(rec,(rnd,sp,arm)) in enumerate(zip(records,[(j,r,a) for j in (0,1) for r in prior.registry() for a in ARMS])):
            sid=sp['state_id'];point=maps[(sid,arm)]
            for k,x in dict(**sp,round=rnd,arm=arm,status='EVALUATED',preparation_valid_both_rounds=valids[sid]).items():compare(rec[k],x,'Endpoint identity/'+k)
            compare(z.json(f'endpoints/record{i:03d}.json'),rec,'Individual endpoint copy');g=endpoint_gates(rec['metrics'],sp['case'],point,saved[sid],valids[sid]);compare(rec['recomputed_gates'],g,'Independent endpoint gates')
            if rnd==0:first[(sid,arm)]=rec['metrics']
            else:
                rep=repeat_check(rec['metrics'],first[(sid,arm)]);compare(rec['repeat_comparison'],rep)
                if not(rep['same_point_identity'] and rep['within_replay_tolerance']):repeat_fail.append(dict(kind='writer_endpoint',state_id=sid,arm=arm,comparison=rep))
        sums,pairs=summary_tables(records);legacy={k:v for k,v in sums.items() if not k.startswith('norm_')};legacy_pairs={k:v for k,v in pairs.items() if not k.startswith('norm_')}
        compare(z.json('CONFIRMATION_SUMMARIES.json'),legacy);compare(s['confirmation_summaries'],legacy);compare(z.json('PAIRED_VS_RAW_AND_C137.json'),legacy_pairs);compare(z.json('REPEATABILITY_FAILURES.json'),repeat_fail)
        passed=not repeat_fail and all(valids.values()) and all(sums['all48'][str(j)][ARMS[2]]['passed']==48 for j in (0,1))
        decision='C141_COMPLETE_REPEAT_DRIFT_NO_CONFIRMATION' if repeat_fail else ('C141_FROZEN_KERNEL_LOCAL_CONFIRMATION_ALL48_PASSED' if passed else PARENT_DECISION)
        branch='RESOLVE_REPEAT_DRIFT_NO_CONFIRMATION' if repeat_fail else ('PREREGISTER_SEPARATE_EXCLUDED_INPUT_PROVENANCE_AND_EVALUATION' if passed else 'STOP_FROZEN_CONFIRMATION_RETAIN_FAILURES_AND_EXPOSED_STATES')
        compare(s['decision'],decision);compare(s['local_confirmation_passed'],bool(passed));compare(s['repeatable'],not repeat_fail);compare(s['preparation_states_valid_both_rounds'],sum(valids.values()));compare(s['next_branch'],branch);compare(s['all_C141_states_exposed_after_this_run'],True)
        compare(s['runtime_calls'],dict(model_forwards=1309,initialisation_model_forwards=1,metric_attempts=528,metric_calls=528,feature_attempts=84,feature_calls=84))
        for k,x in dict(requested_confirmation_states=48,confirmation_states_evaluated=48,requested_preparation_records=96,requested_endpoint_records=288,known_control_records=36,preparation_records=96,new_feature_records=48,final_endpoint_records=288,new_model_forwards=1309).items():compare(s[k],x)
        # Rebind every known replay metric to C140's saved coordinates and endpoint observations.
        known=z.json('KNOWN_CONTROL_PROGRESS.json');need(len(known)==36,'All36 known controls required');c140refs={(r['state_id'],r['arm'],r['round']):r['metrics'] for r in p.json('ALL216_FINAL_TRAINING_ENDPOINTS.json')};need(len(c140refs)==216,'C140 all216 endpoint reference identities')
        preprefs={r['state_id']:r['metrics'] for r in p.json('PREFLIGHT_PROGRESS.json') if r['kind']=='prepared'};need(len(preprefs)==36,'C140 all36 preparation references')
        for rec,sp in zip(known,prior.parent.registry()):
            sid=sp['state_id'];base=data['points'][sid]
            for k,x in sp.items():compare(rec[k],x)
            for k in ('feature_error_Linf','full_feature_error_Linf','raw_map_error_Linf'):need(type(rec[k]) in (float,int) and 0<=rec[k]<=2e-6,'Invalid recorded known-feature parity')
            need(len(rec['measurements'])==4,'All144 known native controls')
            for r,kind in zip(rec['measurements'],('prepared','raw','C137','C140')):
                compare(r['kind'],kind);point=base['prepared'] if kind=='prepared' else (base['raw'] if kind=='raw' else prior.parent.proposal(base['prepared'],base['raw'],v,old if kind=='C137' else new)[0]);native_gates(r['metrics'],sp['case'],point,kind!='prepared')
                arm=prior.parent.EVAL_ARMS[0] if kind=='raw' else (prior.parent.EVAL_ARMS[1] if kind=='C137' else prior.parent.EVAL_ARMS[2]);refs=[preprefs[sid]] if kind=='prepared' else [c140refs[(sid,arm,j)] for j in (0,1)]
                # Prepared refs historically contain two C139 repeats, or one original replay. Their count is retained, not invented.
                expected=[repeat_check(r['metrics'],ref) for ref in refs]
                if kind=='prepared':
                    need(len(r['comparisons']) in (1,2),'Invalid known prepared comparison count')
                    for q in r['comparisons']:
                        need(q['same_point_identity'] and q['within_replay_tolerance'] and q['gate_identity'] and 0<=q['observed_error_Linf']<=.0002 and 0<=q['KL_absolute_error']<=5e-6,'Invalid archived preparation replay comparison')
                else:compare(r['comparisons'],expected)
                need(all(q['same_point_identity'] and q['within_replay_tolerance'] for q in expected),'Known native control drift')
        compare(s['known_controls_replay_consistent'],True);compare(s['original_inputs_unchanged'],{k:True for k in prior.INPUT_KEYS})
        diagnostics=[];recby={(r['round'],r['state_id'],r['arm']):r for r in records}
        for rnd in (0,1):
            for sp in prior.registry():
                sid=sp['state_id'];raw=np.array(recby[(rnd,sid,ARMS[0])]['metrics']['observed'][:6])
                for arm in ARMS[1:]:
                    rr=recby[(rnd,sid,arm)];dd=maps[(sid,arm)].astype(float)-maps[(sid,ARMS[0])].astype(float);d=residual_diagnostic(raw,rr['metrics']['observed'][:6],dd)
                    diagnostics.append(dict(**sp,round=rnd,arm=arm,complete_pass=rr['recomputed_gates']['complete_pass'],failed_gates=rr['recomputed_gates']['failed_gates'],**d))
        for n,obj in [('RECOMPUTED_CONFIRMATION_SUMMARIES.json',sums),('RECOMPUTED_PAIRED_EFFECTS.json',pairs),('RECOMPUTED_PREPARATION_SUMMARIES.json',psums),('ALL192_SIGNED_RESPONSE_EFFECT_DIAGNOSTICS.json',diagnostics),('C140_FAILURE_DIAGNOSTICS.json',[r for r in diagnostics if r['arm']==ARMS[2] and not r['complete_pass']]),('RECOMPUTED_REPEAT_FAILURES.json',repeat_fail)]:dump(out/n,obj)
        budget();return dict(parent_decision=decision,parent_decision_reclassified=False,local_confirmation_passed=bool(passed),members_hashed=member_counts,source_members_checked=len(c['source141']),sealed_artifacts_checked=len(expected_seals),states_audited=48,preparation_records_audited=96,proposal_maps_reconstructed=144,endpoint_records_audited=288,known_control_states_audited=36,known_native_records_audited=144,known_preparations_recompared_to_C140_native=True,legacy_known_preparation_comparison_scalars_independently_recomputed=False,full3211_feature_maximum_reconstruction_error=max(full_errors),actual_frozen_heads_rebound_to_original_C140=True,C140_step12_vector_checked=True,C140_all12_training_updates_reaudited=False,repeatable=not repeat_fail,preparation_states_valid_both_rounds=sum(valids.values()),confirmation_summaries=sums,paired_effects=pairs,C140_failures=sums['all48']['0'][ARMS[2]]['failed_ids'],all_C141_states_remain_exposed=True,small_norm_subset_cannot_replace_full_confirmation=True,elapsed_audit_seconds=time.monotonic()-started)
