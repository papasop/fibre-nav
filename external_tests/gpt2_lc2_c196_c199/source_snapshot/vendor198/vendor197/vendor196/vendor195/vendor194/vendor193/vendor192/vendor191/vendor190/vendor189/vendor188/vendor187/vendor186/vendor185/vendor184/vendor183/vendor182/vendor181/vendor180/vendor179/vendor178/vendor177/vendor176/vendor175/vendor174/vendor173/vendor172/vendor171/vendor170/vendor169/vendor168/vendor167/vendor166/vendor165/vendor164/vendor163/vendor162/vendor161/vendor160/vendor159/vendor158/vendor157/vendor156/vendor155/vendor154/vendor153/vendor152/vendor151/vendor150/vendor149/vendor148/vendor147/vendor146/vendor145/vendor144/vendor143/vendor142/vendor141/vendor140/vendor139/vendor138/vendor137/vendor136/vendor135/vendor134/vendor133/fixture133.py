"""Synthetic archived nonlinear system. Never a GPT2 or LC2 result."""
import zipfile
from core133 import *

def pack(root,path):
    dump(root/'RESULT_MANIFEST.json',{p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file() and p.name!='RESULT_MANIFEST.json'})
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(root).as_posix())
    return path

def metric(theta,case,write):
    p=theta.astype(float);response=p[:6].copy();response[0]+=.1*(p[9]-p[10])**2
    o=np.r_[response,np.tile(p[6:10],4)];truth=safe.target_bits(case,write);bits=[int(x) for x in o[6:10]>=0];m=(2*np.array(truth*4)-1)*o[6:]
    g=dict(bits_match=bits==truth,response=bool(np.max(abs(response))<=BUDGET),KL=True,exact_read_margin=bool(min(m[:4])>=.05),development_read_margin=bool(min(m[4:])>=.05))
    if write:g.update(output_flip=bits[3]!=safe.target_bits(case,False)[3],unselected_preserved=bits[:3]==safe.target_bits(case,False)[:3])
    return dict(observed=o.tolist(),bits=bits,response_drift=float(np.max(abs(response))),KL=0.,exact_margins=m[:4].tolist(),development_margins=m[4:].tolist(),per_anchor_KL=[0.]*12,gates=g,**{'pass':all(g.values())},theta_sha256=safe.tensor_hash(theta))

def make_fixture(folder,unavailable=False):
    root=Path(folder)/'parent';root.mkdir(parents=True);rows=registry();fit=[r for r in rows if r['role']=='geometry_training'];source=b"raise AssertionError('Archived source must never execute')\n";(root/'source').mkdir();(root/'source/NEVER_EXECUTE.py').write_bytes(source)
    bindings=[];archived={};replay=[];points={};local=[]
    for i,spec in enumerate(rows):
        sid=spec['state_id'];theta=np.zeros(DIM,np.float32);theta[6:10]=(2*np.array(safe.target_bits(spec['case'],False))-1)*np.float32(.35);theta[10]=theta[9]
        raw=theta.copy();raw[9]=-raw[9];raw[0]=-.1*float(raw[9]-raw[10])**2+(.003 if spec['role']!='geometry_training' and spec['case']!='xor_01' else .001)
        features=np.zeros(3211,np.float64);features[:11]=theta[:11]
        for k,v in dict(prepared=theta,raw=raw,features=features).items():archived[sid+'__'+k]=v
        bindings.append(dict(**spec,prepared_digest=row_digest(theta),raw_digest=row_digest(raw),features_digest=row_digest(features)))
        replay.append(dict(**spec,preparation=metric(theta,spec['case'],False),raw=metric(raw,spec['case'],True)));points[sid]=(theta,raw)
        if spec['role']=='geometry_training':
            j=np.zeros((10,DIM));j[:,:10]=np.eye(10)
            if unavailable:j[0,9]=1 if len(local)%2 else -1
            local.append(j);np.savez_compressed(root/f'jac_tmp_{len(local)}.npz',J=j,prepared=theta)
    (root/'offline_geometry').mkdir()
    for i in range(1,7):(root/f'jac_tmp_{i}.npz').replace(root/f'offline_geometry/jacobian_{i}.npz')
    v=np.zeros((7 if unavailable else 6,DIM));v[:6,:6]=np.eye(6)
    if unavailable:v[6,9]=1
    a=np.zeros((4,DIM));a[:,6:10]=np.eye(4);b=np.zeros((DIM,4))
    if not unavailable:b[6:10]=np.eye(4)
    stack=np.stack(local)[:,:6].reshape(36,DIM);op=dict(V=v,A=a,B=b,response_stack=stack)
    info=dict(response_numeric_rank=len(v),response_basis_orthogonality_error_Linf=0.,projected_read_numeric_rank=3 if unavailable else 4,read_preserved_arm_available=not unavailable,read_checks={} if unavailable else dict(right_inverse_error_Linf=0.,response_row_basis_times_B_Linf=0.,B_Frobenius_norm=2.))
    candidate_points={};candidate_outcomes=[];mapped={}
    for spec in rows:
        sid=spec['state_id'];theta,raw=points[sid];d=raw.astype(float)-theta.astype(float)
        for arm in ARMS:
            if unavailable and arm==ARMS[2]:point=None;diag=dict(available=False,reason='COMMON_READ_RANK_OR_ALGEBRA_UNAVAILABLE')
            elif arm==ARMS[0]:point=raw.copy();diag=dict(available=True,radius_active=False,delta_norm=float(np.linalg.norm(d)),source='Original exact archived proposal')
            else:
                u=d.copy();u[:6]=0
                if unavailable:u[9]=0
                # This fixture has axis-aligned V, no cap and no read-restoration effect.
                point=(theta.astype(float)+u).astype(np.float32);actual=point.astype(float)-theta.astype(float)
                diag=dict(available=True,radius_active=False,preclip_norm=float(np.linalg.norm(u)),clip_factor=1.,delta_norm=float(np.linalg.norm(actual)),float32_commit_error_Linf=float(np.max(abs(actual-u))),preclip_stacked_response_action_Linf=float(np.max(abs(stack@u))),postclip_stacked_response_action_Linf=float(np.max(abs(stack@u))),actual_stacked_response_action_Linf=float(np.max(abs(stack@actual))),preclip_common_read_action_change_Linf=float(np.max(abs(a@u-a@d))),postclip_common_read_action_change_Linf=float(np.max(abs(a@u-a@d))),actual_common_read_action_change_Linf=float(np.max(abs(a@actual-a@d))),first_order_quantities_are_not_finite_endpoint_bounds=True)
            mapped[(sid,arm)]=point
            if point is not None:candidate_points[sid+'__'+arm]=point
            candidate_outcomes.append(dict(**spec,arm=arm,point_sha256=safe.tensor_hash(point) if point is not None else None,**diag))
    records=[];first={}
    for rnd in (0,1):
        for spec in rows:
            sid=spec['state_id'];theta,_=points[sid]
            for arm in ARMS:
                point=mapped[(sid,arm)];key=(sid,arm)
                if point is None:rec=dict(**spec,arm=arm,round=rnd,status='UNAVAILABLE_COMMON_READ_OPERATOR',metrics=None,recomputed_gates=None)
                else:
                    m=metric(point,spec['case'],True);g=safe.metric_gates(m,spec['case'],point,True,True);g.update(actual_delta_norm=float(np.linalg.norm(point.astype(float)-theta.astype(float))),proposal_radius_passed=True)
                    rec=dict(**spec,arm=arm,round=rnd,status='EVALUATED',metrics=m,recomputed_gates=g)
                    if rnd==0:first[key]=m
                    else:rec['repeat_comparison']=safe.repeat_comparison(m,first[key])
                records.append(rec);dump(root/'endpoints'/sid/f'{arm}_round{rnd}.json',rec)
    summaries,pairs=endpoint_summaries(records);evaluated=sum(r['status']=='EVALUATED' for r in records)
    summary=dict(protocol=PARENT,complete=True,decision='C132_SHARED_KERNEL_FEASIBILITY_COMPLETE_NO_CANDIDATE',synthetic_fixture=True,real_model_execution_claim=False,candidate_selected=False,writer_deployed=False,LC2_closed=False,LC2_single_fold_passed=False,thresholds_changed=False,parent_decision_reclassified=False,requested_states=12,geometry_training_states=6,historical_probe_states=6,requested_endpoint_records=72,evaluated_endpoint_records=evaluated,unavailable_endpoint_records=72-evaluated,endpoint_records=72,repeatable=True,arms_meeting_historical_feasibility_gate=[],arms_supporting_new_center_study=[],endpoint_summaries=summaries,next_branch='SHARED_OPERATOR_NOT_QUALIFIED_REVISE_WRITE_PARAMETERISATION')
    dump(root/'SUMMARY.json',summary);dump(root/'ENDPOINT_SUMMARIES.json',summaries);dump(root/'PAIRED_VS_RAW.json',pairs);dump(root/'REPEATABILITY_FAILURES.json',[]);dump(root/'ALL72_REQUESTED_ENDPOINT_RECORDS.json',records);dump(root/'ENDPOINT_PROGRESS.json',records)
    dump(root/'DESIGN_FROZEN_BEFORE_MODEL.json',dict(registry=rows,response_budget=BUDGET,proposal_radius=RADIUS,arms=list(ARMS),rounds=2));dump(root/'INPUTS_FROZEN.json',{'synthetic_only':True});dump(root/'RUNTIME_SOURCE_RECEIPT.json',[])
    dump(root/'FIXED12_BINDINGS.json',bindings);np.savez_compressed(root/'FIXED12_ARCHIVED_INPUTS.npz',**archived)
    for n in ('ORIGINAL_WRITER.npz','ORIGINAL_CONTROL.npz'):np.savez_compressed(root/n,synthetic=np.zeros(1))
    dump(root/'ALL12_REPLAY.json',replay);dump(root/'OFFLINE_JACOBIAN_PROGRESS.json',fit);np.savez_compressed(root/'FROZEN_SHARED_OPERATORS.npz',**op);dump(root/'OPERATOR_ALGEBRA.json',info);np.savez_compressed(root/'ALL_AVAILABLE_CANDIDATE_POINTS.npz',**candidate_points);dump(root/'ALL36_CANDIDATE_OUTCOMES_FROZEN.json',candidate_outcomes)
    sealed=['DESIGN_FROZEN_BEFORE_MODEL.json','INPUTS_FROZEN.json','FIXED12_BINDINGS.json','FIXED12_ARCHIVED_INPUTS.npz','ORIGINAL_WRITER.npz','ORIGINAL_CONTROL.npz','RUNTIME_SOURCE_RECEIPT.json','ALL12_REPLAY.json','FROZEN_SHARED_OPERATORS.npz','OPERATOR_ALGEBRA.json','ALL_AVAILABLE_CANDIDATE_POINTS.npz','ALL36_CANDIDATE_OUTCOMES_FROZEN.json']+[f'offline_geometry/jacobian_{i}.npz' for i in range(1,7)]
    dump(root/'ALL_ARTIFACT_SEALS_BEFORE_ENDPOINTS.json',{n:sha(root/n) for n in sealed});path=pack(root,Path(folder)/'C132_SYNTHETIC.zip')
    return path,dict(production=False,pin=dict(path=str(path),sha256=sha(path)),source={'NEVER_EXECUTE.py':hashlib.sha256(source).hexdigest()})
