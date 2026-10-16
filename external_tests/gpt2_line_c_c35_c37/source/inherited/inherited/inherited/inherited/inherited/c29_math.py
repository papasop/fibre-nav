import numpy as np

def vec(x):
 a=np.asarray(x,dtype=np.float64)
 if a.shape!=(6,) or not np.isfinite(a).all():raise ValueError('Finite six-response vector required')
 return a

def cosine(a,b):
 a=vec(a);b=vec(b);den=np.linalg.norm(a)*np.linalg.norm(b)
 return float(np.clip(a@b/den,-1,1)) if den>1e-14 else None

def common_mode_energy(a):
 a=vec(a);den=float(a@a)
 return float(min(1.,max(0.,6*float(a.mean())**2/den))) if den>1e-28 else None

def paired(e0,ea,J,delta,half=None,budget=.002,tol=1e-6):
 if not np.isfinite(budget) or budget<=0 or not np.isfinite(tol) or tol<0:raise ValueError('Positive budget and nonnegative diagnostic tolerance required')
 e0=vec(e0);ea=vec(ea);J=np.asarray(J,dtype=np.float64);delta=np.asarray(delta,dtype=np.float64)
 if J.shape!=(6,delta.size) or delta.ndim!=1 or not np.isfinite(J).all() or not np.isfinite(delta).all():raise ValueError('Finite Jacobian and actual displacement shapes required')
 linear=J@delta;actual=ea-e0;pred=e0+linear;remainder=actual-linear
 before=float(np.max(np.abs(e0)));after=float(np.max(np.abs(ea)));predicted=float(np.max(np.abs(pred)))
 pi=predicted<before-tol;ai=after<before-tol;pw=predicted>before+tol;aw=after>before+tol
 label='PREDICTED_AND_ACTUAL_IMPROVE' if pi and ai else ('PREDICTED_IMPROVES_ACTUAL_WORSENS' if pi and aw else ('FIRST_ORDER_AND_ACTUAL_WORSEN' if pw and aw else 'MIXED_OR_SMALL_CHANGE'))
 r={'baseline_residual':e0.tolist(),'endpoint_residual':ea.tolist(),'actual_parameter_displacement_norm':float(np.linalg.norm(delta)),
  'linear_response_change':linear.tolist(),'actual_response_change':actual.tolist(),'linear_predicted_residual':pred.tolist(),'remainder':remainder.tolist(),
  'baseline_drift':before,'endpoint_drift':after,'linear_predicted_drift':predicted,'remainder_infinity':float(np.max(np.abs(remainder))),
  'remainder_over_response_budget':float(np.max(np.abs(remainder)))/budget,'linear_alignment_with_needed_compensation':cosine(-e0,linear),'actual_alignment_with_needed_compensation':cosine(-e0,actual),
  'predicted_response_pass':predicted<=budget,'actual_response_pass':after<=budget,'predicted_pass_actual_fail':predicted<=budget and after>budget,
  'diagnostic_category':label,'baseline_common_mode_energy_fraction':common_mode_energy(e0),'endpoint_common_mode_energy_fraction':common_mode_energy(ea),
  'remainder_includes_floating_and_Jacobian_error':True,'finite_step_certificate':False}
 if half is not None:
  eh=vec(half['residual']);dh=np.asarray(half['delta'],dtype=np.float64)
  if dh.shape!=delta.shape or not np.isfinite(dh).all():raise ValueError('Actual midpoint displacement differs')
  hr=eh-e0-J@dh;h=float(np.max(np.abs(hr)));f=float(np.max(np.abs(remainder)))
  r['fixed_midpoint']={'actual_displacement_norm':float(np.linalg.norm(dh)),'residual':eh.tolist(),'drift':float(np.max(np.abs(eh))),'remainder_infinity':h,
   'full_over_half_remainder_ratio':f/h if h>1e-7 else None,'small_half_remainder_ratio_suppressed':h<=1e-7,'not_backtracking_or_writer_selection':True,'not_a_full_endpoint_acceptance_test':True}
 return r

def aggregate(rows):
 if not rows:raise ValueError('Nonempty diagnosis cohort required')
 cs={k:sum(r['diagnostic_category']==k for r in rows) for k in ('PREDICTED_AND_ACTUAL_IMPROVE','PREDICTED_IMPROVES_ACTUAL_WORSENS','FIRST_ORDER_AND_ACTUAL_WORSEN','MIXED_OR_SMALL_CHANGE')}
 vals=lambda k:[r[k] for r in rows if r[k] is not None]
 def med(k):v=vals(k);return float(np.median(v)) if v else None
 return {'n':len(rows),'categories':cs,'predicted_response_pass':sum(r['predicted_response_pass'] for r in rows),'actual_response_pass':sum(r['actual_response_pass'] for r in rows),
  'predicted_pass_actual_fail':sum(r['predicted_pass_actual_fail'] for r in rows),'median_linear_alignment':med('linear_alignment_with_needed_compensation'),'median_actual_alignment':med('actual_alignment_with_needed_compensation'),
  'maximum_remainder_infinity':max(r['remainder_infinity'] for r in rows),'median_remainder_over_response_budget':med('remainder_over_response_budget'),
  'maximum_endpoint_drift':max(r['endpoint_drift'] for r in rows),'no_causal_or_capacity_certificate':True}
