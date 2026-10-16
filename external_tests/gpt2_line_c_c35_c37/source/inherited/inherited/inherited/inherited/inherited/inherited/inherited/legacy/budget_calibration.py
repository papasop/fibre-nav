"""Offline interval-target dual fitting. Numerical solution, not a feasibility certificate.

For row-normalised A, solve the convex dual
  .5*c^T G c + ridge/2*||c||^2 + e^T*c + sum(b*|c|).
Here e=E/den, b=target_budget/den, D=A^T*c. Equivalently, the
primal penalises distance to a declared response interval rather than to zero.
Norm/trust caps and actual nonlinear gates are applied separately afterwards.
"""
import numpy as np
from radius_calibration import reconstruct_direction,scale_direction

def interval_coefficients(system,ridge,target_budget,max_iterations=8192,tolerance=1e-10,max_polish_iterations=128):
 if not np.isfinite(ridge) or ridge<=0 or not np.isfinite(target_budget) or target_budget<=0:raise ValueError('Positive ridge/target budget required')
 if type(max_iterations)!=int or max_iterations<1 or type(max_polish_iterations)!=int or max_polish_iterations<0 or not np.isfinite(tolerance) or tolerance<=0:raise ValueError('Invalid iteration limit/tolerance')
 values=system['values'];v=system['vectors'];den=system['den'];e=system['E'].reshape(-1)/den;b=target_budget/den
 G=(v*values)@v.T;Q=G+ridge*np.eye(len(e));L=float(values.max()+ridge)
 c=np.zeros_like(e);y=c.copy();momentum=1.;objective=float(np.dot(e,c));best=c.copy();best_obj=objective;converged=False
 def prox_gradient(x):
  grad=Q@x+e;u=x-grad/L
  return np.sign(u)*np.maximum(np.abs(u)-b/L,0.)
 def obj(x):return float(.5*x@(Q@x)+e@x+b@np.abs(x))
 for iteration in range(1,max_iterations+1):
  candidate=prox_gradient(y);candidate_obj=obj(candidate)
  # Monotone restart avoids accepting acceleration overshoot.
  if candidate_obj>objective:
   momentum=1.;y=c.copy();candidate=prox_gradient(y);candidate_obj=obj(candidate)
  next_momentum=(1.+np.sqrt(1.+4.*momentum*momentum))/2.
  y=candidate+((momentum-1.)/next_momentum)*(candidate-c);c=candidate;momentum=next_momentum;objective=candidate_obj
  if objective<best_obj:best=c.copy();best_obj=objective
  residual=L*np.max(np.abs(c-prox_gradient(c)));relative=residual/max(1.,float(np.max(np.abs(e))))
  if relative<=tolerance:converged=True;break
 c=best.copy();polish_count=0;accepted_polish=0
 # Orthant-restricted quadratic refinement uses a direct small-system solve.
 # Projection/backtracking accepts only an actual dual-objective decrease.
 # This avoids relying solely on slow first-order convergence in weak modes.
 for polish_count in range(1,max_polish_iterations+1):
  grad=Q@c+e;nonzero=c!=0.;kkt=np.where(nonzero,np.abs(grad+b*np.sign(c)),np.maximum(np.abs(grad)-b,0.))
  if np.max(kkt)/max(1.,float(np.max(np.abs(e))))<=tolerance:break
  active=np.flatnonzero(nonzero | (np.abs(grad)>b))
  if len(active)==0:break
  signs=np.where(c[active]!=0.,np.sign(c[active]),-np.sign(grad[active]))
  # Solve within an orthant. Newly activated zero coordinates whose Newton
  # direction has the wrong sign are dropped from this working set.
  raw=None
  for working_iteration in range(len(active)+1):
   if len(active)==0:break
   qa=Q[np.ix_(active,active)]
   try:raw=np.linalg.solve(qa,-e[active]-b[active]*signs)
   except np.linalg.LinAlgError:raw=None;break
   wrong=(c[active]==0.) & (raw*signs<=0.)
   if not wrong.any():break
   active=active[~wrong];signs=signs[~wrong];raw=None
  if raw is None:break
  direction=np.zeros_like(c);direction[active]=raw-c[active]
  crossing=(c!=0.) & (c*(c+direction)<=0.) & (direction!=0.)
  limits=np.full_like(c,np.inf);limits[crossing]=-c[crossing]/direction[crossing]
  first_boundary=min(1.,float(limits.min()));found=False
  for backtrack in range(20):
   alpha=first_boundary*2.**(-backtrack);trial=c+alpha*direction
   if backtrack==0:trial[np.isfinite(limits) & (limits<=first_boundary*(1.+1e-12))]=0.
   value=obj(trial)
   if np.isfinite(value) and value<best_obj:
    c=trial;best_obj=value;accepted_polish+=1;found=True;break
  if not found:
   trial=prox_gradient(c);value=obj(trial)
   if not np.isfinite(value) or value>=best_obj:break
   c=trial;best_obj=value;accepted_polish+=1
 grad=Q@c+e
 # Distance to the l1 subgradient stationarity condition.
 nonzero=c!=0.;kkt=np.where(nonzero,np.abs(grad+b*np.sign(c)),np.maximum(np.abs(grad)-b,0.))
 converged=bool(kkt.max()/max(1.,np.max(np.abs(e)))<=tolerance)
 coefficients=c/den
 if not np.isfinite(coefficients).all():raise ValueError('Nonfinite interval coefficients')
 predicted=system['E'].reshape(-1)+G@c*den
 info={'solver':'monotone restarted FISTA plus orthant quadratic refinement, convex interval dual','ridge':ridge,'target_response_interval':target_budget,
  'iterations':iteration,'iteration_limit':max_iterations,'polish_iterations':polish_count,'polish_iteration_limit':max_polish_iterations,'accepted_polish_updates':accepted_polish,'converged':converged,'dual_objective':best_obj,
  'maximum_dual_KKT_residual':float(kkt.max()),'relative_dual_KKT_residual':float(kkt.max()/max(1.,np.max(np.abs(e)))),
  'unscaled_solver_linear_maximum_drift':float(np.max(np.abs(predicted))),
  'unscaled_solver_linear_maximum_interval_excess':float(np.max(np.maximum(np.abs(predicted)-target_budget,0.))),
  'rank_certified':False,'linear_feasibility_certified':False,'finite_acceptance_certified':False,
  'ridge_allows_interval_penalty_residual':True,'norm_constraints_applied_after_solver':True}
 return coefficients,info
