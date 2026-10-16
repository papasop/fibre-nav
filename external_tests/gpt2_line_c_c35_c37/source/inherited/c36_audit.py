"""Safe tensor identity, numeric gate recount and prewrite proposal provenance."""
import numpy as np
import torch
from c36_protocol import *
from c35_fit import records,check_frozen,prior_geometry
from c35_math import compile_affine
from c35_parent import validate as validate_c35
from c35_protocol import specs as c35_specs
from worker_c35 import completed,proposals
from c31_runtime import load,vector,tensor_hash
from worker_c34 import check_metrics

def parent_specs(data):
    fixed=read(Path(data)/'FIXED_STATE_SPECS.json')
    return fixed['new_training']+fixed['prospective']+c35_specs()

def state_data(data,spec):
    data=Path(data)
    folder=data/('new_states' if '__c35_new' in spec['id'] else 'states/'+spec['cohort'])/spec['id']
    prep=vector(load(folder/'prepared.pt'));inp=load(folder/'INPUT.pt')
    require(inp['spec']==spec and inp['prepared_sha256']==tensor_hash(prep),'Parent input/preparation binding differs')
    return prep,inp

def writer_identity(data):
    from c27_features import state_hash
    from c27_protocol import PARENT_CP_SHA
    from c28_protocol import PARENT_BRANCH_CP_SHA,PARENT_MATRIX_SHA
    from c31_runtime import HEAD_SHA
    data=Path(data);cp=load(data/'parent/C25_FINAL_CHECKPOINT.pt');branch=load(data/'parent/C27_BRANCH.pt')
    require(sha(data/'parent/C25_FINAL_CHECKPOINT.pt')==PARENT_CP_SHA and state_hash(cp['head'])==HEAD_SHA,'Actual C25 head identity mismatch')
    require(sha(data/'parent/C27_BRANCH.pt')==PARENT_BRANCH_CP_SHA and tensor_hash(branch['residual_matrix'])==PARENT_MATRIX_SHA,'Actual C27 branch identity mismatch')
    return {'C25_head_sha256':state_hash(cp['head']),'C27_matrix_sha256':tensor_hash(branch['residual_matrix']),
            'authoritative_source_constants_checked':True}

def finite_tensors(value):
    if isinstance(value,torch.Tensor):
        require(bool(torch.isfinite(value).all()),'Nonfinite archived tensor')
        return 1
    if isinstance(value,dict):return sum(finite_tensors(v) for v in value.values())
    if isinstance(value,(list,tuple)):return sum(finite_tensors(v) for v in value)
    return 0

def audit(data,root,identity_checker=writer_identity):
    data=Path(data);validate_c35(data,Path(root)/'inherited');ss=parent_specs(data)
    summary=completed(data,ss);require(summary['complete'] and len(ss)==80,'Parent C35 did not finish all80')
    require(summary['decision']==SETTINGS['parent_expected_decision'],'Parent C35 decision changed')
    check_frozen(data);rr=records(data);tensor_rows=[]
    for p in sorted(data.rglob('*.pt')):
        if '__c36_new' in p.as_posix():continue
        value=load(p);n=finite_tensors(value)
        tensor_rows.append({'member':p.relative_to(data).as_posix(),'sha256':sha(p),'finite_tensors':n})
    for case,r in zip(CASES,rr):
        prior=prior_geometry(data,case)
        for key,expected in {**prior,**compile_affine(r)}.items():
            require(np.allclose(r[key],expected,rtol=1e-12,atol=1e-12),'Compiled/prior head does not match frozen construction: '+case+'/'+key)
    for spec in ss:
        sid=spec['id'];rec=read(data/'results'/sid/'COMPLETE.json');prep,inp=state_data(data,spec)
        value,point_file,ready=proposals(None,None,data,spec,prep,inp)
        require(value['core_forward_calls']==1 and not value['intermediate_candidate_forward'],'Parent proposal was not single feedforward')
        require(set(value['points'])==set(ARMS) and value['all_points_generated_before_endpoint_audit'] and not value['postwrite_response_used'],'Parent proposal scope differs')
        for arm in ARMS:
            m=rec['arms'][arm]['metrics'];check_metrics(m,spec['case'])
            require(m['theta_sha256']==tensor_hash(vector(value['points'][arm])),'Parent endpoint/physical tensor mismatch')
            require(rec['arms'][arm]['complete_accepted']==bool(rec['preparation_valid'] and m['pass']),'Parent acceptance flag inconsistent')
            a=read(data/'results'/sid/(arm+'.json'))
            require(a['ready_sha256']==sha(ready) and a['metrics']==m,'Parent arm audit/seal binding differs')
        if (len(tensor_rows)+ss.index(spec))%16==0:print('C36 ARCHIVAL ENDPOINT',sid,flush=True)
    ident=identity_checker(data)
    return {'complete':True,'numeric_gate_recount_matches_summary':True,'archived_states_audited':80,
            'archived_proposal_tensors_verified':240,'safe_loaded_pt_files':len(tensor_rows),
            'finite_tensor_count':sum(t['finite_tensors'] for t in tensor_rows),'tensor_files':tensor_rows,
            'actual_core_identity':ident,'all_four_compiled_affine_heads_rederived':True,
            'stored_seal_and_metric_bindings_checked':True,'does_not_independently_prove_historical_event_order':True,
            'not_a_full80_physical_forward_replay':True,'cohorts':summary['cohorts']}

def reproduce_all(rt,head,data,callback=None):
    """All archived80 proposal vectors must reproduce before new-state generation."""
    data=Path(data);rows=[]
    for i,spec in enumerate(parent_specs(data),1):
        prep,inp=state_data(data,spec);saved=load(data/'proposals'/spec['id']/'POINTS.pt')
        with torch.no_grad():
            points,_=head(prep.to(rt.device),inp['input'].to(rt.device),inp['route'].to(rt.device),inp['extras'][:6].to(rt.device).double()*.002)
        errors={a:float((points[a].cpu().double()-saved['points'][a].double()).abs().max()) for a in ARMS}
        require(all(v<=SETTINGS['proposal_reproduction_tolerance'] for v in errors.values()),'Parent proposal does not reproduce exactly: '+spec['id'])
        rows.append({'id':spec['id'],'maximum_absolute_errors':errors})
        if i%16==0:print('C36 PARENT PROPOSAL REPRODUCTION',i,'/80',flush=True)
    rt.invariant();return {'n':len(rows),'all_exact':True,'rows':rows,'model_endpoint_evaluations':0}

def compare_replay(actual,old,case):
    check_metrics(actual,case);check_metrics(old,case)
    require(actual['theta_sha256']==old['theta_sha256'] and actual['bits']==old['bits'] and actual['gates']==old['gates'],'Physical replay changed endpoint/gates')
    re=float(np.max(np.abs(np.asarray(actual['response_residual'])-old['response_residual'])))
    ke=abs(actual['kl']-old['kl'])
    me=max(abs(x-y) for k in ('exact_margins','heldout_margins') for x,y in zip(actual['evaluation'][k],old['evaluation'][k]))
    require(re<=SETTINGS['physical_replay_response_tolerance'] and ke<=SETTINGS['physical_replay_kl_tolerance'] and me<=SETTINGS['physical_replay_margin_tolerance'],'Physical replay exceeds fixed numeric tolerance')
    return {'response_error_Linf':re,'KL_error':ke,'margin_error_Linf':me,'gate_identity':True}
