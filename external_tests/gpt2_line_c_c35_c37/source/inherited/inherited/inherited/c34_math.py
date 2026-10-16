"""Small-row SVD basis construction. No claim of numerical-rank certification."""
import numpy as np
from c34_protocol import *

def construct(jacobians):
    js=[np.asarray(j,dtype=np.float64) for j in jacobians]
    require(len(js)==3 and len(set(j.shape for j in js))==1 and js[0].shape[0]==6,'Three6xd response Jacobians required')
    A=np.concatenate(js);require(np.isfinite(A).all() and A.shape[1]>0,'Finite geometry required')
    U,s,Vh=np.linalg.svd(A,full_matrices=False)
    cutoff=max(SETTINGS['absolute_singular_cutoff'],SETTINGS['relative_singular_cutoff']*float(s[0]))
    keep=s>cutoff;T=np.tile(np.eye(6),(3,1))
    B=(Vh[keep].T/s[keep])@(U[:,keep].T@T) if keep.any() else np.zeros((A.shape[1],6))
    b=(Vh[keep].T/s[keep])@(U[:,keep].T@np.ones(18)) if keep.any() else np.zeros(A.shape[1])
    require(np.isfinite(B).all() and np.isfinite(b).all(),'Nonfinite compensation basis')
    return {'response6':B,'common1':b[:,None]}, {'singular_values':s.tolist(),'cutoff':cutoff,'numerical_rank':int(keep.sum()),
        'response6_stack_target_error_Linf':float(np.max(np.abs(A@B-T))),
        'common1_stack_target_error_Linf':float(np.max(np.abs(A@b-1))),
        'response6_frobenius_norm':float(np.linalg.norm(B)), 'common1_norm':float(np.linalg.norm(b)),
        'common_direct_vs_sum_basis_numerical_difference_L2':float(np.linalg.norm(b-B@np.ones(6))),
        'numerical_rank_is_not_feasibility_certificate':True,'finite_preservation_not_certified':True}

def proposal(bases,arm,residual):
    e=np.asarray(residual,dtype=np.float64);require(e.shape==(6,) and np.isfinite(e).all(),'Finite six-coordinate measured residual required')
    require(arm in ('common1_oracle','response6_oracle'),'Unknown oracle arm')
    B=np.asarray(bases['common1' if arm=='common1_oracle' else 'response6'],dtype=np.float64)
    coef=np.array([-e.mean()]) if arm=='common1_oracle' else -e
    require(B.ndim==2 and B.shape[1]==len(coef) and np.isfinite(B).all(),'Invalid basis')
    raw=B@coef;n=float(np.linalg.norm(raw));radius=SETTINGS['maximum_parameter_increment_norm']
    scale=min(1.,radius/n) if n else 1.;delta=raw*scale
    require(np.isfinite(delta).all(),'Nonfinite compensation increment')
    return delta,{'oracle_coefficients':coef.tolist(),'unclipped_norm':n,'radial_scale':scale,
                  'clipped':scale<1.,'requested_increment_norm':float(np.linalg.norm(delta)),
                  'measured_raw_response_used':True,'post_write_correction_attempts':1,'backtracking_steps':0}

def aggregate(records):
    require(records,'Empty requested group')
    out={}
    for arm in ARMS:
        rs=[r['arms'][arm]['metrics'] for r in records]
        valid=[r['preparation_valid'] for r in records]
        raw=[r['arms']['raw']['metrics'] for r in records]
        out[arm]={'n_requested':len(records),'valid_preparations':sum(valid),
           'complete_passed':sum(bool(v and m['pass']) for v,m in zip(valid,rs)),
           'response_gate_passed_all':sum(m['gates']['response'] for m in rs),
           'gate_counts':{g:sum(m['gates'][g] for m in rs) for g in GATES},
           'maximum_response_drift':max(m['drift'] for m in rs),
           'median_response_drift':float(np.median([m['drift'] for m in rs])),
           'newly_complete_passed':sum(bool(v and not a['pass'] and b['pass']) for v,a,b in zip(valid,raw,rs)),
           'lost_complete_passes':sum(bool(v and a['pass'] and not b['pass']) for v,a,b in zip(valid,raw,rs)),
           'clipped_states':sum(r['arms'][arm].get('proposal',{}).get('clipped',False) for r in records),
           'oracle_response_correction_not_correction_free_writing':arm!='raw'}
    return out
