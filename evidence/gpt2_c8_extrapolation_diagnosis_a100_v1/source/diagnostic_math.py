"""Post hoc geometry measurements only. No fitted writer or candidate replacement."""
import numpy as np

def finite(a):
    a=np.asarray(a,dtype=np.float64)
    if not np.isfinite(a).all():raise ValueError('Nonfinite diagnostic input')
    return a

def span_basis(rows,rtol=1e-8,atol=1e-10):
    rows=finite(rows)
    if rows.ndim!=2 or rows.shape[0]==0:raise ValueError('Nonempty row matrix required')
    _,s,V=np.linalg.svd(rows,full_matrices=False)
    cutoff=max(atol,rtol*(float(s[0]) if len(s) else 0.))
    keep=s>cutoff
    return V[keep],{'rank_at_declared_cutoff':int(keep.sum()),'singular_values':s.tolist(),
                    'cutoff':cutoff,'rtol':rtol,'atol':atol,'certificate':False}

def span_residual(basis,v):
    v=finite(v);basis=finite(basis)
    if v.ndim!=1 or basis.ndim!=2 or basis.shape[1]!=v.size:raise ValueError('Span shape mismatch')
    residual=v-basis.T@(basis@v);size=float(np.linalg.norm(v));err=float(np.linalg.norm(residual))
    return {'norm':size,'residual_norm':err,'relative_residual':err/size if size>0 else None}

def feature_coverage(train,query,rtol=1e-8,atol=1e-10):
    train=finite(train);query=finite(query)
    if train.ndim!=2 or query.shape!=(train.shape[1],):raise ValueError('Feature shape mismatch')
    centre=train.mean(0);basis,svd=span_basis(train-centre,rtol,atol)
    d=np.linalg.norm(train-query,axis=1);nearest=int(d.argmin())
    # Leave-self-out training distance is reported separately; no arbitrary OOD gate is imposed.
    distances=np.linalg.norm(train[:,None,:]-train[None,:,:],axis=-1)
    np.fill_diagonal(distances,np.inf)
    reference=np.min(distances,axis=1) if len(train)>1 else np.array([])
    return {'affine_span':span_residual(basis,query-centre),'svd':svd,
      'nearest_training_index':nearest,'nearest_training_distance':float(d[nearest]),
      'training_leave_self_out_nearest_median':float(np.median(reference)) if len(reference) else None,
      'training_leave_self_out_nearest_max':float(np.max(reference)) if len(reference) else None,
      'span_inclusion_is_not_generalisation_certificate':True}

def delta_comparison(predicted,teacher,basis):
    predicted=finite(predicted);teacher=finite(teacher)
    if predicted.shape!=teacher.shape or predicted.ndim!=1:raise ValueError('Delta shape mismatch')
    pn=float(np.linalg.norm(predicted));tn=float(np.linalg.norm(teacher));err=float(np.linalg.norm(predicted-teacher))
    return {'predicted_norm':pn,'teacher_norm':tn,'norm_ratio':pn/tn if tn else None,
      'cosine_with_teacher':float(predicted@teacher/(pn*tn)) if pn and tn else None,
      'error_norm':err,'relative_teacher_error':err/tn if tn else None,
      'predicted_training_teacher_span':span_residual(basis,predicted),
      'teacher_training_teacher_span':span_residual(basis,teacher),
      'teacher_used_for_diagnosis_only':True}

def linear_vs_finite(D,delta,before,after):
    D=finite(D);delta=finite(delta);before=finite(before);after=finite(after)
    linear=D@delta;actual=after-before;remainder=actual-linear
    return {'linear_change':linear.tolist(),'actual_finite_change':actual.tolist(),
      'measured_remainder':remainder.tolist(),'linear_norm2':float(np.linalg.norm(linear)),
      'actual_norm2':float(np.linalg.norm(actual)),'remainder_norm2':float(np.linalg.norm(remainder)),
      'remainder_norm_inf':float(np.max(np.abs(remainder))),
      'curvature_bound_certified':False}

def committed_delta(prepared,candidate):
    # Subtract committed float32 states in float64. Do not analyse an uncommitted ideal proposal.
    a=np.asarray(prepared);b=np.asarray(candidate)
    if a.dtype!=np.float32 or b.dtype!=np.float32:raise ValueError('Committed float32 states required')
    return finite(b)-finite(a)
