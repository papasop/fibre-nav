"""Two fixed LP calls. Directions are offline diagnostic witnesses, never heads."""
import warnings,math
import numpy as np
from core167 import SOLVER_OPTIONS,need

def serial(result):
 def scalar(v):
  if isinstance(v,np.ndarray):return scalar(v.tolist())
  if isinstance(v,np.generic):return scalar(v.item())
  if isinstance(v,dict):return {k:scalar(x) for k,x in v.items()}
  if isinstance(v,(list,tuple)):return [scalar(x) for x in v]
  if isinstance(v,float) and not math.isfinite(v):return 'NaN' if math.isnan(v) else 'Infinity' if v>0 else '-Infinity'
  return v
 return scalar(dict(result))

def solve(A,free,backend=None,on_attempt=None,on_result=None):
 if backend is None:
  from scipy.optimize import linprog
  backend=linprog
 A=np.asarray(A);free=np.asarray(free);need(A.dtype==np.float64 and A.ndim==2 and free.dtype==np.bool_ and free.shape==(A.shape[1],),'Fixed LP shapes');need(np.isfinite(A).all(),'Finite solver matrix');n,m=A.shape
 if not n:return dict(direction=np.zeros(m),dual=np.empty(0),results=[],calls=0,warnings=[],primal_residual=None,dual_residual=None)
 c=np.zeros(m+1);c[-1]=-1.;bounds=[(-1.,1.) if f else (0.,0.) for f in free]+[(None,None)]
 dualc=np.r_[np.zeros(n),np.ones(m)];dualineq=np.block([[A.T,-np.eye(m)],[-A.T,-np.eye(m)]])
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  if on_attempt:on_attempt(1)
  p=backend(c,A_ub=np.c_[A,np.ones(n)],b_ub=np.zeros(n),bounds=bounds,method='highs-ds',options=dict(SOLVER_OPTIONS))
  if on_result:on_result('primal',serial(p))
  if on_attempt:on_attempt(2)
  q=backend(dualc,A_ub=dualineq,b_ub=np.zeros(2*m),A_eq=np.r_[np.ones(n),np.zeros(m)][None,:],b_eq=np.ones(1),bounds=[(0.,None)]*(n+m),method='highs-ds',options=dict(SOLVER_OPTIONS))
  if on_result:on_result('dual',serial(q))
 # A status code is logged, never treated as a certificate. Finite vectors can be verified even after a time limit.
 if p.x is None or q.x is None:return dict(direction=None,dual=None,results=[serial(p),serial(q)],calls=2,warnings=[str(w.message) for w in caught],primal_residual=None,dual_residual=None)
 px=np.asarray(p.x,dtype=np.float64);qx=np.asarray(q.x,dtype=np.float64);need(px.shape==(m+1,) and qx.shape==(n+m,) and np.isfinite(px).all() and np.isfinite(qx).all(),'Finite final solver vectors')
 d=np.clip(px[:m],-1.,1.);d[~free]=0.;lam=np.maximum(qx[:n],0.)
 if not np.any(lam):return dict(direction=None,dual=None,results=[serial(p),serial(q)],calls=2,warnings=[str(w.message) for w in caught],primal_residual=None,dual_residual=None)
 return dict(direction=d,dual=lam,results=[serial(p),serial(q)],calls=2,warnings=[str(w.message) for w in caught],primal_residual=dict(maximum_linear_violation=float(max(0.,np.max(A@px[:m]+px[-1]))),maximum_bound_violation=float(max(0.,np.max(abs(px[:m])-1.))),claimed_margin=float(px[-1])),dual_residual=dict(simplex_error=float(abs(qx[:n].sum()-1.)),minimum_raw_lambda=float(qx[:n].min()),maximum_inequality_violation=float(max(0.,np.max(dualineq@qx))),claimed_upper=float(q.fun)))
