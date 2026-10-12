"""Float64 CPU thin-SVD geometry; inferred numerical rank is not certification."""
import numpy as np

def checked_matrix(J):
    J=np.asarray(J,dtype=np.float64)
    if J.ndim!=2 or not np.isfinite(J).all():raise ValueError('Finite 2D Jacobian required')
    return J

def null_project(J,v,relative_cutoff=1e-8,absolute_cutoff=1e-10):
    J=checked_matrix(J);v=np.asarray(v,dtype=np.float64)
    if v.shape!=(J.shape[1],) or not np.isfinite(v).all():raise ValueError('Direction mismatch/nonfinite')
    _,s,vt=np.linalg.svd(J,full_matrices=False)
    threshold=max(absolute_cutoff,relative_cutoff*(float(s[0]) if len(s) else 0.))
    keep=s>threshold;basis=vt[keep]
    projected=v-basis.T@(basis@v)
    info={'singular_values':s.tolist(),'numerical_rank':int(keep.sum()),'rank_cutoff':threshold,
          'linear_leak_before':float(np.linalg.norm(J@v)),
          'linear_leak_after_float64':float(np.linalg.norm(J@projected)),
          'input_norm':float(np.linalg.norm(v)),'projected_norm':float(np.linalg.norm(projected)),
          'rank_certified':False,'finite_response_preservation_certified':False}
    return projected,info

def ridge_correction(J,e,ridge=1e-6):
    J=checked_matrix(J);e=np.asarray(e,dtype=np.float64)
    if ridge<=0 or e.shape!=(J.shape[0],) or not np.isfinite(e).all():raise ValueError('Invalid ridge/residual')
    left,s,vt=np.linalg.svd(J,full_matrices=False)
    delta=-vt.T@((s/(s*s+ridge))*(left.T@e))
    residual=e+J@delta
    info={'ridge':ridge,'singular_values':s.tolist(),'delta_norm':float(np.linalg.norm(delta)),
          'universal_delta_norm_bound':float(np.linalg.norm(e)/(2*np.sqrt(ridge))),
          'linear_residual_norm':float(np.linalg.norm(residual)),
          'actual_nonlinear_improvement_guaranteed':False}
    return delta,info
