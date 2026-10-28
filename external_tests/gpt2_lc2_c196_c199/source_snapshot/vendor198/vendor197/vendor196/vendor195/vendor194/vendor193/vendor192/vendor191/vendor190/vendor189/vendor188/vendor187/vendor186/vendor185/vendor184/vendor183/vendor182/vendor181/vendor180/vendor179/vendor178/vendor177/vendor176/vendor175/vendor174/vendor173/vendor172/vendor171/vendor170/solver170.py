"""One registered primal LP, no restarts/alternate solver or head search."""
from core170 import *
def solve(pr,backend=None,on_attempt=None,on_result=None):
 if not len(pr['active']):return dict(direction=np.zeros(pr['anchor'].size),calls=0,solver_result=None,warnings=[],empty_active_noop=True)
 if backend is None:
  import scipy
  need(scipy.__version__=='1.17.0','Fixed production SciPy1.17.0 required')
  from scipy.optimize import linprog
  backend=linprog
 from solver167 import serial
 import warnings
 A=pr['A'];m=A.shape[1];cost=np.r_[np.zeros(m),-1.];bounds=[(-1.,1.) if f else (0.,0.) for f in pr['free']]+[(None,None)]
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  if on_attempt:on_attempt()
  r=backend(cost,A_ub=np.c_[A,np.ones(len(A))],b_ub=np.zeros(len(A)),bounds=bounds,method='highs-ds',options=dict(SOLVER_OPTIONS))
  saved=serial(r)
  if on_result:on_result(saved)
 need(r.x is not None,'LP returned no direction; no retry/fallback');x=np.asarray(r.x,np.float64);need(x.shape==(m+1,) and np.isfinite(x).all(),'Finite final primal vector');d=np.clip(x[:m],-1.,1.);d[~pr['free']]=0.
 return dict(direction=d,calls=1,solver_result=saved,warnings=[str(w.message) for w in caught],empty_active_noop=False,raw_claimed_margin=float(x[-1]),direction_bound_adjustment_Linf=float(max(abs(d-x[:m]))))
