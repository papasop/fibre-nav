"""Single complete-coordinate primal LP; exact arithmetic decides usability."""
from core174 import *
def solve(pr,backend=None,on_attempt=None,on_result=None):
 if not len(pr['active']):return dict(direction=np.zeros_like(pr['anchor']),calls=0,solver_result=None,warnings=[],empty_active_noop=True)
 if backend is None:
  import scipy
  need(scipy.__version__=='1.17.0','Pinned SciPy1.17.0 required')
  from scipy.optimize import linprog
  backend=linprog
 from solver167 import serial
 import warnings
 m=pr['anchor'].size;cost=np.r_[np.zeros(m),-1.];bounds=[(-1.,1.) if f else (0.,0.) for f in pr['free']]+[(0.,None)]
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  if on_attempt:on_attempt()
  result=backend(cost,A_ub=np.c_[pr['A'],np.ones(1008)],b_ub=pr['rhs'],bounds=bounds,method='highs-ds',options=dict(SOLVER_OPTIONS));saved=serial(result)
  if on_result:on_result(saved)
 need(result.x is not None,'No finite primal vector: no fallback or infeasibility claim');x=farray(np.asarray(result.x,np.float64),(m+1,));d=np.clip(x[:-1],-1.,1.);d[~pr['free']]=0.;return dict(direction=d.reshape(pr['anchor'].shape),calls=1,solver_result=saved,warnings=[str(w.message) for w in caught],empty_active_noop=False,raw_claimed_normalized_margin=float(x[-1]),raw_maximum_linear_violation=float(max(0.,np.max(pr['A']@x[:-1]+x[-1]-pr['rhs']))),direction_bound_adjustment_Linf=float(np.max(abs(d-x[:-1]))),solver_optimality_or_infeasibility_certified=False)
