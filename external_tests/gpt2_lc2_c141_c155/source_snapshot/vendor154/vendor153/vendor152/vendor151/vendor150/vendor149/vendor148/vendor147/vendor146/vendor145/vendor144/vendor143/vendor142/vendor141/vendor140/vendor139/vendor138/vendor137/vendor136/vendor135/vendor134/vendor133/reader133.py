"""Read C132 as data. No imported or executed Python from the scientific ZIP."""
from contextlib import closing
from core133 import *

def load(path,c,out):
    need(sha(path)==c['pin']['sha256'],'C132 whole ZIP hash mismatch')
    with closing(Archive(path)) as z:
        members=z.verify_manifest();parent=z.json('SUMMARY.json')
        need(parent.get('protocol')==PARENT and parent.get('complete') is True,'Completed C132 required')
        compare(parent['decision'],'C132_SHARED_KERNEL_FEASIBILITY_COMPLETE_NO_CANDIDATE')
        for k in ('candidate_selected','writer_deployed','LC2_closed','LC2_single_fold_passed','thresholds_changed','parent_decision_reclassified'):compare(parent[k],False,k)
        if c['production']:
            compare(parent['synthetic_fixture'],False);compare(parent['real_model_execution_claim'],True)
        ns={n[7:] for n in z.z.namelist() if n.startswith('source/')}
        need(ns==set(c['source']),'C132 delivered source inventory changed')
        for n,h in c['source'].items():need(hashlib.sha256(z.raw('source/'+n)).hexdigest()==h,'C132 delivered source changed: '+n)
        parentdesign=z.json('DESIGN_FROZEN_BEFORE_MODEL.json');compare(parentdesign['registry'],registry());compare(parentdesign['response_budget'],BUDGET);compare(parentdesign['proposal_radius'],RADIUS);compare(parentdesign['arms'],list(ARMS));compare(parentdesign['rounds'],2)
        seals=z.json('ALL_ARTIFACT_SEALS_BEFORE_ENDPOINTS.json')
        required={'DESIGN_FROZEN_BEFORE_MODEL.json','INPUTS_FROZEN.json','FIXED12_BINDINGS.json','FIXED12_ARCHIVED_INPUTS.npz','ORIGINAL_WRITER.npz','ORIGINAL_CONTROL.npz','RUNTIME_SOURCE_RECEIPT.json','ALL12_REPLAY.json','FROZEN_SHARED_OPERATORS.npz','OPERATOR_ALGEBRA.json','ALL_AVAILABLE_CANDIDATE_POINTS.npz','ALL36_CANDIDATE_OUTCOMES_FROZEN.json'}|{f'offline_geometry/jacobian_{i}.npz' for i in range(1,7)}
        need(set(seals)==required,'C132 pre-endpoint seal coverage changed')
        for n,h in seals.items():need(hashlib.sha256(z.raw(n)).hexdigest()==h,'C132 pre-endpoint seal changed: '+n)
        bindings=z.json('FIXED12_BINDINGS.json');need(len(bindings)==12,'All12 bindings required')
        archived=z.arrays('FIXED12_ARCHIVED_INPUTS.npz');need(set(archived)=={r['state_id']+'__'+k for r in registry() for k in ('prepared','raw','features')},'Archived input coverage differs')
        points={}
        for spec,binding in zip(registry(),bindings):
            for k,v in spec.items():compare(binding[k],v,'Fixed state identity/'+k)
            sid=spec['state_id'];p={k:farray(archived[sid+'__'+k],shape) for k,shape in (('prepared',(DIM,)),('raw',(DIM,)),('features',(3211,)))}
            need(p['prepared'].dtype==p['raw'].dtype==np.float32,'Archived coordinates must be float32')
            for k in p:compare(row_digest(p[k]),binding[k+'_digest'],'C132 archived '+k+' binding')
            points[sid]=dict(spec=spec,**p)
        replay=z.json('ALL12_REPLAY.json');need(len(replay)==12,'All12 replay records required')
        for spec,rec in zip(registry(),replay):
            for k,v in spec.items():compare(rec[k],v,'Replay identity/'+k)
            p=points[spec['state_id']];need(safe.metric_gates(rec['preparation'],spec['case'],p['prepared'],False),'Invalid prepared state')
            safe.metric_gates(rec['raw'],spec['case'],p['raw'],True,True)
            p['preparation']=rec['preparation'];p['raw_replay']=rec['raw']
        fit=[r for r in registry() if r['role']=='geometry_training'];compare(z.json('OFFLINE_JACOBIAN_PROGRESS.json'),fit,'Six local derivative identities')
        js={}
        for i,spec in enumerate(fit,1):
            f=z.arrays(f'offline_geometry/jacobian_{i}.npz');need(set(f)=={'J','prepared'},'Jacobian array inventory differs')
            exact(f['prepared'],points[spec['state_id']]['prepared'],'Jacobian prepared-coordinate binding differs')
            js[spec['state_id']]=farray(f['J'],(10,DIM))
        op=z.arrays('FROZEN_SHARED_OPERATORS.npz');info=z.json('OPERATOR_ALGEBRA.json');need(set(op)=={'V','A','B','response_stack'},'Operator inventory differs')
        v=farray(op['V'],(info['response_numeric_rank'],DIM));a=farray(op['A'],(4,DIM));b=farray(op['B'],(DIM,4));stack=farray(op['response_stack'],(36,DIM))
        exact(stack,np.stack([js[r['state_id']][:6] for r in fit]).reshape(36,DIM),'Response stack differs from local derivatives')
        close(a,np.stack([js[r['state_id']][6:10] for r in fit]).mean(0),'Common read map differs',1e-12)
        orth=float(np.max(abs(v@v.T-np.eye(len(v))))) if len(v) else 0.
        need(orth<=ALG_TOL,'Frozen response basis not orthonormal');compare(info['response_basis_orthogonality_error_Linf'],orth)
        span=float(np.max(abs(stack-(stack@v.T)@v)));need(span<=ALG_TOL*max(1.,float(np.max(abs(stack)))),'Frozen basis does not span response rows')
        available=info['read_preserved_arm_available'];need(type(available) is bool,'Invalid arm availability')
        ap=a-(a@v.T)@v;right=float(np.max(abs(ap@b-np.eye(4))));null=float(np.max(abs(v@b))) if len(v) else 0.;bnorm=float(np.linalg.norm(b))
        if available:
            compare(info['projected_read_numeric_rank'],4);need(right<=ALG_TOL and null<=ALG_TOL*max(1.,bnorm),'Read restoration algebra invalid')
            compare(info['read_checks'],dict(right_inverse_error_Linf=right,response_row_basis_times_B_Linf=null,B_Frobenius_norm=bnorm))
        algebra=dict(response_basis_orthogonality_error_Linf=orth,response_stack_span_residual_Linf=span,right_inverse_error_Linf=right,response_row_basis_times_B_Linf=null,read_preserved_arm_available=available,rank_recomputed_with_SVD=False,AD_values_independently_verified=False)
        candidates=z.arrays('ALL_AVAILABLE_CANDIDATE_POINTS.npz');diags=z.json('ALL36_CANDIDATE_OUTCOMES_FROZEN.json');need(len(diags)==36,'All36 candidate outcome records required')
        expected_names=set();maps={};checked=[]
        for (spec,arm),diag in zip([(r,arm) for r in registry() for arm in ARMS],diags):
            sid=spec['state_id'];p=points[sid]
            for k,vv in dict(**spec,arm=arm).items():compare(diag[k],vv,'Candidate identity/'+k)
            point,precap,postcap,computed=reconstruct(p['prepared'],p['raw'],arm,op,available)
            expected=dict(**spec,arm=arm,point_sha256=(safe.tensor_hash(point) if point is not None else None),**computed)
            compare(diag,expected,'Candidate algebra',atol=1e-10,rtol=1e-10)
            if point is not None:
                name=sid+'__'+arm;expected_names.add(name);exact(candidates[name],point,'Actual float32 candidate differs')
            maps[(sid,arm)]=dict(point=point,precap=precap,postcap=postcap);checked.append(expected)
        need(set(candidates)==expected_names,'Candidate tensor coverage differs')
        original=z.json('ALL72_REQUESTED_ENDPOINT_RECORDS.json');need(len(original)==72,'All72 endpoint records required');compare(z.json('ENDPOINT_PROGRESS.json'),original,'Endpoint progress copy')
        records=[];first={};repeat_failures=[]
        for (rnd,spec,arm),rec in zip([(rnd,r,a) for rnd in (0,1) for r in registry() for a in ARMS],original):
            sid=spec['state_id'];key=(sid,arm);point=maps[key]['point'];expected_id=dict(**spec,arm=arm,round=rnd)
            for k,vv in expected_id.items():compare(rec[k],vv,'Endpoint identity/'+k)
            compare(z.json(f'endpoints/{sid}/{arm}_round{rnd}.json'),rec,'Individual endpoint record')
            new=dict(rec)
            if point is None:
                compare(rec['status'],'UNAVAILABLE_COMMON_READ_OPERATOR');compare(rec['metrics'],None);compare(rec['recomputed_gates'],None)
            else:
                compare(rec['status'],'EVALUATED');g=gate_record(rec['metrics'],spec['case'],point,points[sid]['prepared']);compare(rec['recomputed_gates'],g,'Recomputed endpoint gates');new['recomputed_gates']=g
                if rnd==0:first[key]=rec['metrics']
                else:
                    rep=safe.repeat_comparison(rec['metrics'],first[key]);compare(rec['repeat_comparison'],rep,'Repeat arithmetic')
                    if not (rep['within_replay_tolerance'] and rep['same_point_identity']):repeat_failures.append(dict(state_id=sid,arm=arm,comparison=rep))
            records.append(new)
        summaries,pairs=endpoint_summaries(records);compare(z.json('ENDPOINT_SUMMARIES.json'),summaries,'Parent endpoint summary');compare(parent['endpoint_summaries'],summaries,'Parent embedded endpoint summary');compare(z.json('PAIRED_VS_RAW.json'),pairs,'Paired effects');compare(z.json('REPEATABILITY_FAILURES.json'),repeat_failures,'Repeat failure registry');compare(parent['repeatable'],not repeat_failures)
        ready=[arm for arm in ARMS[1:] if not repeat_failures and all(summaries['all12'][str(r)][arm]['passed']==summaries['all12'][str(r)][arm]['n_evaluated']==12 for r in (0,1))]
        improved=[arm for arm in ready if all(pairs['all12'][str(r)][arm]['rescued_ids'] and not pairs['all12'][str(r)][arm]['destroyed_ids'] for r in (0,1))];compare(parent['arms_meeting_historical_feasibility_gate'],ready);compare(parent['arms_supporting_new_center_study'],improved)
        raw_all=all(summaries['all12'][str(r)][ARMS[0]]['passed']==12 for r in (0,1));branch='PLAN_SEPARATE_NEW_CENTER_STUDY_WITH_PREREGISTERED_ARM_RULE' if improved else ('RAW_ALREADY_PASSES_THIS_SCOPE_NO_STRUCTURAL_GAIN_ESTABLISHED' if raw_all else 'SHARED_OPERATOR_NOT_QUALIFIED_REVISE_WRITE_PARAMETERISATION');compare(parent['next_branch'],branch)
        evaluated=sum(r['status']=='EVALUATED' for r in records)
        for k,value in dict(requested_states=12,geometry_training_states=6,historical_probe_states=6,requested_endpoint_records=72,evaluated_endpoint_records=evaluated,unavailable_endpoint_records=72-evaluated,endpoint_records=72).items():compare(parent[k],value,k)
        dump(out/'C132_ARCHIVE_AUDIT.json',dict(members_hashed=members,delivered_source_members=len(ns),pre_endpoint_seals_checked=len(seals),state_bindings_checked=12,local_Jacobians_bound=6,candidate_outcomes_checked=36,endpoint_records_checked=72,source_executed=False,ancestor_original_archives_reopened=False))
        dump(out/'RECOMPUTED_OPERATOR_ALGEBRA.json',algebra);dump(out/'RECONSTRUCTED_CANDIDATE_DIAGNOSTICS.json',checked);dump(out/'RECOMPUTED_ENDPOINT_RECORDS.json',records);dump(out/'RECOMPUTED_ENDPOINT_SUMMARIES.json',summaries);dump(out/'RECOMPUTED_PAIRED_VS_RAW.json',pairs)
        return dict(parent=parent,points=points,jacobians=js,maps=maps,records=records,summaries=summaries,repeatable=not repeat_failures,algebra=algebra)
